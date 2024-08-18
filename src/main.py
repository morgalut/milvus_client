from text_extraction import extract_text_from_file
from embedding import generate_embeddings, insert_embeddings
from search_and_generate import search_documents, generate_response
from milvus_client import Collection
import os

folder_path = r"C:\Users\Mor\Desktop\work\doc"
documents = []
file_names = []

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    file_text = extract_text_from_file(file_path)
    if file_text:
        documents.append(file_text)
        file_names.append(file_name)

# Connect to Milvus and get collection
collection = Collection(name="document_collection")

# Generate and insert embeddings
embeddings = generate_embeddings(documents)
insert_embeddings(collection, documents, embeddings, file_names)

# Example query
query = "Tell me about the invention of the airplane."
query_embedding = generate_embeddings([query])[0]

# Search and generate response
search_results = search_documents(query_embedding)
if 'results' in search_results and search_results['results']:
    retrieved_docs = [doc['text'] for doc in search_results['results']]
    response = generate_response(query, retrieved_docs)
    print("Generated Response:", response)
else:
    print("No documents retrieved or unexpected response structure.")
