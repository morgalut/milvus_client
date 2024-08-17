# C:\Users\Mor\Desktop\work\src\search_and_generate.py

import json
import requests
from transformers import pipeline

# Define the API endpoints and headers
headers = {
    'Authorization': 'Bearer api_token',
    'Content-Type': 'application/json'
}

def search_documents(query_vector, top_k=10):
    url = 'https://in03-9b9fce0682a5279.api.gcp-us-west1.zillizcloud.com/v1/collections/mor/search'
    data = {
        "search": [
            {
                "data": [query_vector],
                "annsField": "vector",
                "limit": top_k,
                "outputFields": ["*"]
            }
        ]
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()

def generate_response(query, retrieved_docs):
    generator = pipeline('text-generation', model='gpt-3.5-turbo')
    context = query + " " + " ".join(retrieved_docs)
    response = generator(context, max_length=150)
    return response[0]['generated_text']
