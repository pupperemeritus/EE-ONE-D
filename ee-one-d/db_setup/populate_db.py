import asyncio
import logging
import multiprocessing
import string

import nltk
import pandas as pd
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    db,
)
from pymilvus.model.dense import SentenceTransformerEmbeddingFunction

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Increase this value based on your system's capabilities
CHUNK_SIZE = 10000
BATCH_SIZE = 512


def encode_and_prepare_data():
    # Connect to Milvus
    connections.connect(alias="default", host="localhost", port=19530)

    # Create or switch to a database
    if "eeoned" not in db.list_database():
        db.create_database("eeoned")
    db.using_database("eeoned")

    # Initialize SentenceTransformer model for embeddings
    ef = SentenceTransformerEmbeddingFunction(
        device="cuda",
        trust_remote_code=True,
        batch_size=BATCH_SIZE,
    )

    # Load NLTK words
    words = nltk.corpus.words.words()

    # Process words
    processed_words = set()
    for word in words:
        # Convert to lowercase
        word = word.lower()
        # Remove punctuation
        word = word.translate(str.maketrans("", "", string.punctuation))
        # Remove words shorter than 2 characters
        if len(word) > 1:
            processed_words.add(word)

    # Convert set back to list
    processed_words = list(processed_words)

    logging.info(f"Original word count: {len(words)}")
    logging.info(f"Processed word count: {len(processed_words)}")

    # Encode words to obtain vectors
    vectors = ef.encode_documents(processed_words)

    logging.debug(f"Dim: {ef.dim}, {len(vectors[0])}")

    # Prepare data for upload
    data = {
        "id": list(range(len(processed_words))),
        "vector": vectors,
        "text": processed_words,
        "subject": ["words"] * len(processed_words),
    }

    return data, len(vectors[0])


async def bulk_insert_chunk(collection, chunk):
    df = pd.DataFrame(chunk)
    collection.insert(df)
    logging.debug(f"Inserted chunk of size {len(chunk['id'])}")


async def bulk_insert_concurrent(collection, data):
    chunk_size = CHUNK_SIZE
    tasks = []
    for i in range(0, len(data["id"]), chunk_size):
        chunk = {k: v[i : i + chunk_size] for k, v in data.items()}
        task = asyncio.create_task(bulk_insert_chunk(collection, chunk))
        tasks.append(task)
    await asyncio.gather(*tasks)


async def main():
    # Connect to Milvus
    connections.connect(alias="default", host="localhost", port=19530)

    # Use multiprocessing to process data
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        data, vector_dim = pool.apply(encode_and_prepare_data)

    # Connect to Milvus client
    client = MilvusClient(host="localhost", db_name="eeoned", port=19530, timeout=10**5)

    # Define collection name
    collection_name = "semantic_embeddings"
    db.using_database("eeoned")
    # Drop collection if exists and create new collection with ORM
    if client.has_collection(collection_name=collection_name):
        logging.debug("Dropping collection")
        client.drop_collection(collection_name=collection_name)

    # Define collection schema using CollectionSchema and FieldSchema
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=256),
        ]
    )

    # Create collection using Collection
    collection = Collection(name=collection_name, schema=schema)

    # Supported index params, modify these based on your Milvus version and support
    index_params = {
        "index_type": "GPU_IVF_FLAT",  # Change this if needed
        "params": {"nlist": 128},
        "metric_type": "L2",  # Change this if needed
    }

    try:
        collection.create_index(field_name="vector", index_params=index_params)
    except Exception as e:
        print(f"Failed to create index: {e}")
        return

    # Perform bulk insertion concurrently
    await bulk_insert_concurrent(collection, data)

    logging.info("Bulk insertion completed.")
    collection.flush()
    # Load the collection
    collection.load()

    print(f"Number of entities in collection: {collection.num_entities}")
    # Verify data insertion
    # Disconnect from Milvus
    connections.disconnect("default")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    asyncio.run(main())
