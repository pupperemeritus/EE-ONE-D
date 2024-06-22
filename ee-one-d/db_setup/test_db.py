import logging
import time

from pymilvus import MilvusClient, connections, db, utility
from pymilvus.model.dense import SentenceTransformerEmbeddingFunction
from pymilvus.orm.collection import Collection, CollectionSchema, DataType, FieldSchema

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.running = False

    def start(self):
        if not self.running:
            self.start_time = time.time()
            self.running = True
            self.end_time = None
        else:
            print("Timer is already running.")

    def stop(self):
        if self.running:
            self.end_time = time.time()
            self.running = False
        else:
            print("Timer is not running.")

    def reset(self):
        self.start_time = None
        self.end_time = None
        self.running = False

    def elapsed(self):
        if self.running:
            return time.time() - self.start_time
        elif self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        else:
            return 0.0

    def __str__(self):
        return f"Elapsed time: {self.elapsed()} seconds"


# Connect to Milvus
connections.connect(alias="default", host="localhost", port=19530)

client = MilvusClient()

# Use the existing collection
collection_name = "semantic_embeddings"
if not utility.has_collection(collection_name):
    logging.error("Collection does not exist")
    exit(1)
else:
    logging.info("Using collection %s", collection_name)

# Load the collection
collection = Collection(collection_name)

# Initialize the SentenceTransformer embedding function
ef = SentenceTransformerEmbeddingFunction()
ef.device = "cuda"
ef.trust_remote_code = True

# Define the collection schema
schema = CollectionSchema(
    [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=256),
    ]
)


try:
    timer = Timer()
    query = "tea"
    encoded_query = ef.encode_queries([query])
    logging.info(f"Encoded query shape: {len(encoded_query[0])}")
    timer.start()
    results = collection.search(
        data=encoded_query,
        anns_field="vector",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=10,
        output_fields=["text", "subject"],
    )
    timer.stop()
    logging.debug(timer)
    if results:
        for hits in results:
            for hit in hits:
                print(
                    f"ID: {hit.id}, Distance: {hit.distance}, Text: {hit.entity.get('text')}"
                )
    else:
        print("No results found.")
except Exception as e:
    logging.error(f"Error performing search: {e}")

    logging.info("Script execution completed.")

# Disconnect from Milvus
connections.disconnect("default")