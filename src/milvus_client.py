import logging
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def connect_to_milvus():
    try:
        # Connect to Zilliz Cloud using HTTP protocol
        connections.connect(
            alias="default",
            uri="https://in03-9b9fce0682a5279.api.gcp-us-west1.zillizcloud.com",
            token=""
        )
        logging.info("Successfully connected to Milvus.")
    except Exception as e:
        logging.error(f"Failed to connect to Milvus: {e}")
        raise

# Define the Milvus collection schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),  # Ensure the dimension matches your embeddings
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=100),
]

schema = CollectionSchema(fields=fields, description="Document collection for RAG")
collection_name = "document_collection"

def setup_collection():
    try:
        # Check if the collection already exists, if so, drop it
        if utility.has_collection(collection_name):
            collection = Collection(name=collection_name)
            collection.drop()
            logging.info(f"Dropped existing collection: {collection_name}")

        # Create the collection
        collection = Collection(name=collection_name, schema=schema)
        logging.info(f"Collection created: {collection_name}")
        return collection
    except Exception as e:
        logging.error(f"Failed to set up collection: {e}")
        raise

# Run setup
connect_to_milvus()
collection = setup_collection()
