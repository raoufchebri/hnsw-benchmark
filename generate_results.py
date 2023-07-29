import json
import psycopg2
from urllib.parse import urlparse

# Parse the connection string
# connection_string = "postgres://raouf:SBYv16cFKWZD@ep-dark-bonus-301225.us-east-1.aws.neon.tech/neondb?sslmode=require"

# Connect to the database
print("Connecting to the database...")
conn = psycopg2.connect(
    user="postgres",
    password="mysecretpassword",
    host="localhost",
    port="5432"
)
cur = conn.cursor()
print("Connected to the database.")

# Load the embeddings from the test set
print("Loading the embeddings from the test set...")
with open("test_set_embeddings.json", "r") as file:
    embeddings = json.load(file)
print(f"Loaded {len(embeddings)} embeddings from the test set.")

# Define the number of nearest neighbors to retrieve
k = 100

# Define a dictionary to hold the results
results = {}

# Iterate over the embeddings
for i, embedding in enumerate(embeddings):
    print(f"Running query for embedding {i + 1} of {len(embeddings)}...")

    # Define the query
    query = "SELECT _id FROM documents ORDER BY openai_vector <=> %s::vector LIMIT %s"

    # Execute the query
    cur.execute(query, (embedding, k))

    # Fetch the results
    res = cur.fetchall()

    # Add the results to the dictionary
    results[i] = [r[0] for r in res]

    print(f"Query for embedding {i + 1} completed.")

# Save the results to a JSON file
print("Saving the results to a JSON file...")
with open("800k_results.json", "w") as file:
    json.dump(results, file)
print("Results saved to a JSON file.")

# Close the database connection
print("Closing the database connection...")
cur.close()
conn.close()
print("Database connection closed.")
