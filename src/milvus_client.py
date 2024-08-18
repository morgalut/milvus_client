"""
This module handles the connection to Milvus, sets up the collection schema,
and manages the collection.
"""

import logging
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def connect_to_milvus():
    """
    Connects to Milvus using Zilliz Cloud via HTTP protocol.
    """
    try:
        connections.connect(
            alias="default",
            uri="https://in03-9b9fce0682a5279.api.gcp-us-west1.zillizcloud.com",
            token="api_key"
        )
        logging.info("Successfully connected to Milvus.")
    except Exception as e:
        logging.error("Failed to connect to Milvus: %s", e)
        raise

# Define the Milvus collection schema
FIELDS = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),  # Ensure the dimension matches your embeddings
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=100),
]

SCHEMA = CollectionSchema(fields=FIELDS, description="Document collection for RAG")
COLLECTION_NAME = "document_collection"

def setup_collection():
    """
    Sets up the Milvus collection, including schema, index, and loading.
    """
    try:
        if utility.has_collection(COLLECTION_NAME):
            collection = Collection(name=COLLECTION_NAME)
            collection.drop()
            logging.info("Dropped existing collection: %s", COLLECTION_NAME)

        collection = Collection(name=COLLECTION_NAME, schema=SCHEMA)
        logging.info("Collection created: %s", COLLECTION_NAME)

        # Create index on the embedding field
        index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
        collection.create_index(field_name="embedding", index_params=index_params)
        logging.info("Index created for the collection.")

        # Load the collection explicitly after creation
        collection.load()
        logging.info("Collection loaded: %s", COLLECTION_NAME)

        return collection
    except Exception as e:
        logging.error("Failed to set up collection: %s", e)
        raise

# Run setup
connect_to_milvus()
collection = setup_collection()
