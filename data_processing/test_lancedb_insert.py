import lancedb
from sentence_transformers import SentenceTransformer
import json
import pyarrow as pa
import numpy as np

# Sample data
summary_json = {
    "summary": "This passage discusses the importance of data science and its applications in various industries.",
    "keywords": "data science, applications, industries"
}
page_number = 5
original_content = "The full content of the passage goes here, which includes a lot of information on data science..."

# Convert summary_json to a string representation for vectorization
summary_text = summary_json['summary']

# Load a pre-trained transformer model for vectorization (e.g., using sentence-transformers)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Vectorize the summary text
vector = model.encode(summary_text)
# Connect to LanceDB (creates an on-disk database at specified path)
db = lancedb.connect('./data/my_lancedb')

# Define schema using PyArrow
schema = pa.schema([
    pa.field("page_number", pa.int64()),
    pa.field("original_content", pa.string()),
    pa.field("summary", pa.string()),
    pa.field("keywords", pa.string()),
    pa.field("vectorC", pa.list_(pa.float32(),384)),#all-MiniLM-L6-v2: This model has an embedding size of 384.
])

# Create or connect to a table with the schema
table = db.create_table('summaries', schema=schema, mode='overwrite')

# Store the summary vector with metadata
table.add([{
    "page_number": page_number,
    "original_content": original_content,
    "summary": summary_json['summary'],
    "keywords": summary_json['keywords'],
    "vectorC": vector  # The embedding/vector representation of the summary
}])

print("Summary stored successfully in LanceDB.")