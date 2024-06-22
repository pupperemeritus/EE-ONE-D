import asyncio
import logging
import multiprocessing

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

    # Encode words to obtain vectors
    vectors = ef.encode_documents(words)

    logging.debug(f"Dim: {ef.dim}, {len(vectors[0])}")

    # Prepare data for upload
    data = {
        "id": list(range(len(words))),
        "vector": vectors,  # vectors is already a list of lists
        "text": words,
        "subject": ["words"] * len(words),
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

    # ... rest of the function remains the same ...

    # Define collection name
    collection_name = "semantic_embeddings"

    # Drop collection if exists and create new collection with ORM
    if client.has_collection(collection_name=collection_name):
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

    # Create index
    index_params = {
        "index_type": "GPU_IVF_FLAT",
        "params": {"nlist": 128},
        "metric_type": "L2",
    }
    collection.create_index(field_name="vector", index_params=index_params)

    # Perform bulk insertion concurrently
    await bulk_insert_concurrent(collection, data)

    logging.info("Bulk insertion completed.")

    # Load the collection
    collection.load()

    # Verify data insertion
    print(f"Number of entities in collection: {collection.num_entities}")
    collection.flush()
    # Disconnect from Milvus
    connections.disconnect("default")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    asyncio.run(main())
