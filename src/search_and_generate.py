"""
This module handles text embedding generation, document search, and response generation.
"""

import json
import requests
import torch
from transformers import AutoTokenizer, AutoModel, pipeline

# Define the API endpoints and headers
HEADERS = {
    'Authorization': 'Bearer api_key',
    'Content-Type': 'application/json'
}

def generate_embeddings(texts):
    """
    Generates embeddings for a list of texts using a pre-trained model.
    
    Args:
        texts (list of str): The texts to generate embeddings for.
    
    Returns:
        numpy.ndarray: The generated embeddings.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize and encode the texts
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Generate embeddings
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).numpy()

    return embeddings

def search_documents(query_vector, collection_name="document_collection", top_k=10):
    """
    Searches for documents in Milvus using the provided query vector.
    
    Args:
        query_vector (list or numpy.ndarray): The query vector for searching.
        collection_name (str): The name of the collection to search in.
        top_k (int): Number of top results to return.
    
    Returns:
        dict: The search results.
    """
    url = 'https://in03-9b9fce0682a5279.api.gcp-us-west1.zillizcloud.com/v2/vectordb/entities/search'
    
    query_vector_list = query_vector.tolist()
    
    data = {
        "collectionName": collection_name,
        "data": [query_vector_list],
        "annsField": "embedding",
        "limit": top_k,
        "outputFields": ["*"]
    }
    
    try:
        response = requests.post(url, headers=HEADERS, data=json.dumps(data), timeout=10)
        response.raise_for_status()
        response_json = response.json()
        print("Response JSON:", response_json)
        return response_json
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return {}

def generate_response(query, retrieved_docs):
    """
    Generates a response based on the query and retrieved documents using a text generation model.
    
    Args:
        query (str): The query to generate a response for.
        retrieved_docs (list of str): The documents retrieved from the search.
    
    Returns:
        str: The generated response.
    """
    generator = pipeline('text-generation', model='gpt-3.5-turbo')
    context = query + " " + " ".join(retrieved_docs)
    response = generator(context, max_length=150)
    return response[0]['generated_text']

def test_search():
    """
    Tests the search and response generation functionality.
    """
    query_text = "Tell me about the invention of the airplane."
    query_embedding = generate_embeddings([query_text])[0]
    
    # Perform the search
    search_results = search_documents(query_embedding)
    
    # Check and handle response structure
    if 'results' in search_results:
        if search_results['results']:
            retrieved_docs = [doc['text'] for doc in search_results['results']]
            print("Retrieved Documents:", retrieved_docs)
        else:
            print("No documents retrieved.")
    else:
        print("Unexpected response structure:", search_results)

test_search()
