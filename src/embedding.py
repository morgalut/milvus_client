import os
from sentence_transformers import SentenceTransformer
from milvus_client import Collection

model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(documents):
    return model.encode(documents)

def insert_embeddings(collection, documents, embeddings, file_names):
    ids = [i for i in range(len(embeddings))]
    data = [ids, embeddings, documents, file_names]
    collection.insert(data)
