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
