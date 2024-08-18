import json
import requests
import torch
from transformers import AutoTokenizer, AutoModel, pipeline

# Define the API endpoints and headers
headers = {
    'Authorization': 'Bearer ',
    'Content-Type': 'application/json'
}

# Define a function to generate embeddings from a text query
def generate_embeddings(texts):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # or another model suitable for embedding generation
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize and encode the texts
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Generate embeddings
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).numpy()

    return embeddings

def search_documents(query_vector, collection_name="document_collection", top_k=10):
    url = 'https://in03-9b9fce0682a5279.api.gcp-us-west1.zillizcloud.com/v2/vectordb/entities/search'
    
    query_vector_list = query_vector.tolist()
    
    data = {
        "collection_name": collection_name,
        "data": [query_vector_list],
        "annsField": "embedding",
        "limit": top_k,
        "outputFields": ["*"]
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raise HTTPError for bad responses
        response_json = response.json()
        print("Response JSON:", response_json)  # Add logging to inspect the response
        return response_json
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return {}

def generate_response(query, retrieved_docs):
    generator = pipeline('text-generation', model='gpt-3.5-turbo')
    context = query + " " + " ".join(retrieved_docs)
    response = generator(context, max_length=150)
    return response[0]['generated_text']

# Define the query and generate the query embedding
query = "Tell me about the invention of the airplane."
query_embedding = generate_embeddings([query])[0]

# Perform the search
search_results = search_documents(query_embedding)

# Check and handle response structure
if 'results' in search_results:
    if search_results['results']:
        retrieved_docs = [doc['text'] for doc in search_results['results']]
        response = generate_response(query, retrieved_docs)
        print("Generated Response:", response)
    else:
        print("No documents retrieved.")
else:
    print("Unexpected response structure:", search_results)
