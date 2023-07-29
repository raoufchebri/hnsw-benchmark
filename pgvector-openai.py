import json
import psycopg2
import time
import numpy

class Pgvector:
    def __init__(self, connection, metric="angular", m = 64, ef_construction = 32):
        self._metric = metric

        self._m = m
        self._ef_construction = ef_construction

        self._opclass = {'angular': 'vector_cosine_ops', 'euclidean': 'vector_l2_ops'}[metric]
        self._op = {'angular': '<=>', 'euclidean': '<->'}[metric]
        self._table = 'documents' #'vectors_%s_%d' % (metric, lists)
        self._query = "SELECT _id FROM %s ORDER BY openai_vector %s %%s::vector LIMIT %%s" % (self._table, self._op)

        self._conn = connection
        self._conn.autocommit = True
        self._cur = self._conn.cursor()

    def create_index(self):
        self._cur.execute("SET maintenance_work_mem to '8000MB'")
        self._cur.execute('CREATE INDEX pgvector_hnsw_idx ON %s USING hnsw (openai_vector vector_cosine_ops) WITH (m = %d, ef_construction = %d)' % (self._table, self._m, self._ef_construction))

    def drop_index(self):
        self._cur.execute('DROP INDEX IF EXISTS pgvector_hnsw_idx')

    def set_query_arguments(self, ef_search):
        self._ef_search = ef_search
        self._cur.execute('SET hnsw.ef_search = %s', (str(ef_search),))
        self._cur.execute('SET effective_io_concurrency=10')
        self._cur.execute('SET enable_seqscan=off')

    def query(self, v, n, explain = False):
        query = self._query
        if explain:
            query = 'EXPLAIN ANALYZE ' + query
        self._cur.execute(query, (numpy.array(v).tolist(), n))
        res = self._cur.fetchall()
        return [r[0] for r in res]

    
    def insert_search_results(self, recall, exec_time, probes, lists):    
        self._cur.execute('INSERT INTO search_results (recall, exec_time, probes, lists) VALUES (%s, %s, %s, %s)', (recall, exec_time, probes, lists))

    def prewarm(self):
        self._cur.execute("SELECT pg_prewarm('pgvector_hnsw_idx', 'buffer')")

    def __str__(self):
        return 'Pgvector(m=%d, ef_construction=%d)' % (self._m, self._ef_construction)
    
    def set_maintenance_work_mem(self, value):
        self._cur.execute("SET maintenance_work_mem to '%sMB'" % value)
    
    def count(self):
        self._cur.execute('SELECT COUNT(*) FROM %s' % self._table)
        return self._cur.fetchone()[0]

def main():
    knn = 100 #  Number of nearest neighbours for the algorithm to return.',

    # import X_test from test_set_embeddings.json
    with open('test_set_embeddings.json', 'r') as f:
        X_test = json.load(f)

    # import X_train from test_results.json
    with open('test_results.json', 'r') as f:
        X_test_results = json.load(f)

    print('got %d queries' % len(X_test))

    m_list = [64]
    ef_construction_list = [32]
    ef_search_list = [1000]

    data_connection_string="postgres://"

    data_conn = psycopg2.connect(data_connection_string)
    connection = psycopg2.connect(
        user="postgres",
        password="password",
        host="localhost",
        port="5432"
    )

    data_conn.autocommit = True
    data_cur = data_conn.cursor()
    # create empty result dict with keys as i and values as recall and exec_time
    result = {}

    for m in m_list:
        for ef_construction in ef_construction_list:
            pgvector_algo = Pgvector(m=m, ef_construction=ef_construction, connection=connection)
            count = pgvector_algo.count()

            print('dropping index for %s' % pgvector_algo)
            pgvector_algo.drop_index()
            print('creating index for %s' % pgvector_algo)
            pgvector_algo.set_maintenance_work_mem(15000)
            start_time = time.time()
            pgvector_algo.create_index()
            build_time = time.time()
            print('Built index in', build_time - start_time)


            data_cur.execute('INSERT INTO build_times (extension, build_time, m, ef_construction) VALUES (%s, %s, %s, %s)', ("pgvector", build_time - start_time, m, ef_construction))
            
            print("prewarming index")
            pgvector_algo.prewarm()
            print("index prewarmed")
            for ef_search in ef_search_list:
                print('ef_search: %d' % (ef_search))

                
                for i, x in enumerate(X_test):
                    # run naive similarity search to get brute force results
                    result = X_test_results[str(i)]

                    # run similarity search with ivfflat with explain and analyze
                    pgvector_algo.set_query_arguments(ef_search=ef_search)
                    explain_pgvector = pgvector_algo.query(x, knn, explain=True)
                    exec_time = explain_pgvector[-1].split('Execution Time: ')[1].split(' ms')[0]

                    # get similarity search results
                    pgvector_algo.set_query_arguments(ef_search=ef_search)
                    pgvector_results = pgvector_algo.query(x, knn)

                    # compare ivfflat results with brute force results to calculate recall
                    recall = len(set(result).intersection(set(pgvector_results))) / len(result)
                    # print i, recall and exec_time
                    print('m: %s, ef_construction: %s, ef_search: %s, count: %f, i: %d, recall: %f, exec_time: %s' % (m, ef_construction, ef_search, count, i, recall, exec_time))

                    # insert recall and exec_time into table with data_cur, with m, ef_construction, ef_search and extension
                    data_cur.execute('INSERT INTO search_results (recall, exec_time, m, ef_construction, ef_search, extension) VALUES (%s, %s, %s, %s, %s, %s)', (recall, exec_time, m, ef_construction, ef_search, "pgvector"))