# Retrieval-Augmented Generation (RAG) System

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system using Zilliz Cloud's Milvus for document retrieval and GPT-3.5 for response generation.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Configure Milvus connection in `milvus_client.py`
3. Place your documents in the `data/documents/` directory

## Usage
Run the `main.py` script to process documents, generate embeddings, and perform queries.

```bash
python src/main.py
```
## Test 
curl --request POST --url https://in03-9b9fce0682a5279.api.gcp-us-west1.zillizcloud.com/v2/vectordb/collections/list --header "accept: application/json" --header "authorization: Bearer 6b91b245bdc156fa023c50d04a5d2b6a1a7dcb4fe9cb6797b3efb59aa13797387b5e75c3f40e05a07f4afd06b98f3506c8ccbacc" --data "{}"
