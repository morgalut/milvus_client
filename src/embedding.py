"""
This module handles the generation and insertion of embeddings for documents
using the SentenceTransformer model and a Milvus collection.
"""

import os
from sentence_transformers import SentenceTransformer
from milvus_client import setup_collection
from text_extraction import extract_text_from_file

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(docs):
    """
    Generate embeddings for the provided documents using the SentenceTransformer model.
    """
    embeddings = model.encode(docs)
    print("Embeddings Shape:", embeddings.shape)  # Print shape to verify
    return embeddings

def insert_embeddings(coll, docs, embs, filenames):
    """
    Insert document embeddings along with their metadata into the Milvus collection.
    
    Parameters:
    - coll: Milvus collection object.
    - docs: List of document texts.
    - embs: Embeddings generated for the documents.
    - filenames: List of file names corresponding to the documents.
    """
    ids = list(range(len(embs)))

    # Truncate documents to fit the max length allowed in the collection
    max_length = 512
    truncated_docs = [doc[:max_length] for doc in docs]
    
    data = [
        ids,
        embs.tolist(),
        truncated_docs,
        filenames
    ]

    # Insert the data into the Milvus collection
    coll.insert(data)

# Example usage
if __name__ == "__main__":
    # Define the folder containing your documents
    FOLDER_PATH = r"C:\Users\Mor\Desktop\work\doc"
    
    # Initialize lists to hold documents and corresponding file names
    doc_texts = []
    filenames = []

    # Extract text from each document in the folder
    for file_name in os.listdir(FOLDER_PATH):
        file_path = os.path.join(FOLDER_PATH, file_name)
        file_text = extract_text_from_file(file_path)
        if file_text:
            doc_texts.append(file_text)
            filenames.append(file_name)

    # Connect to Milvus and get the collection
    collection = setup_collection()

    # Generate embeddings for the documents
    embeddings = generate_embeddings(doc_texts)

    # Insert the embeddings and metadata into the collection
    insert_embeddings(collection, doc_texts, embeddings, filenames)
