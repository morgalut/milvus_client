import os
from sentence_transformers import SentenceTransformer
from milvus_client import Collection

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(documents):
    # Generate embeddings for the provided documents
    return model.encode(documents)

def insert_embeddings(collection, documents, embeddings, file_names):
    # Generate IDs for each embedding
    ids = [i for i in range(len(embeddings))]
    
    # Truncate documents to fit the max length
    max_length = 512
    truncated_documents = [doc[:max_length] for doc in documents]
    
    # Prepare the data to be inserted into the Milvus collection
    data = [ids, embeddings.tolist(), truncated_documents, file_names]
    collection.insert(data)

# Example usage
if __name__ == "__main__":
    # Example documents (replace this with actual documents in your use case)
    documents = ["Sample document text 1", "Sample document text 2"]
    
    # Generate embeddings
    embeddings = generate_embeddings(documents)
    print(embeddings.shape)  # This should print (number_of_documents, 384)
    
    # Initialize your Milvus collection (replace with actual collection setup)
    collection = Collection(name="document_collection")
    
    # Example file names (replace this with actual file names in your use case)
    file_names = ["doc1.txt", "doc2.txt"]
    
    # Insert the embeddings into the Milvus collection
    insert_embeddings(collection, documents, embeddings, file_names)
