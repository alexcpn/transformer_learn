import lancedb
from sentence_transformers import SentenceTransformer
import numpy as np

# Connect to LanceDB (assuming you have already created the database and table)
db = lancedb.connect('./data/my_lancedb')
table = db.open_table('summaries')

# Load the same pre-trained transformer model for vectorization
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to vectorize the user query
def vectorize_query(query):
    return model.encode(query)

# Function to search for the most relevant summary
def search_summary(query):
    # Vectorize the user question
    query_vector = vectorize_query(query)
    
    # Perform similarity search using LanceDB and specify the vector column
    results = table.search(query_vector,vector_column_name="vectorS").limit(3).to_list()


    if results:
        for idx, result in enumerate(results):
            print(f"Result {idx + 1}:")
            print(f"Page Number: {result['page_number']}")
            print(f"Summary: {result['summary']}")
            print(f"Keywords: {result['keywords']}")
            print(f"Original Text: {result['original_content']}")
            print('-' * 80)
    else:
        print("No relevant summary found.")

# Example usage
user_question = "Not able to manage new devices"
search_summary(user_question)