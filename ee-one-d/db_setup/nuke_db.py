import logging
import sys

from pymilvus import connections, db, utility

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def drop_all_collections():
    # Connect to Milvus
    connections.connect(alias="default", host="localhost", port=19530)

    # Switch to the "default" database
    db.using_database(sys.argv[1])

    # Get all collection names
    collection_names = utility.list_collections()

    if not collection_names:
        logging.info("No collections found in the 'default' database.")
    else:
        logging.info(f"Found {len(collection_names)} collections.")

        # Drop each collection
        for name in collection_names:
            try:
                utility.drop_collection(name)
                logging.info(f"Dropped collection: {name}")
            except Exception as e:
                logging.error(f"Failed to drop collection {name}: {e}")

    # Disconnect from Milvus
    connections.disconnect("default")


if __name__ == "__main__":
    drop_all_collections()
    logging.info("Script execution completed.")
