import logging
import logging.config
import os
import sys

from pymilvus import connections, db, utility

try:
    logging.config.fileConfig(os.path.join(os.getcwd(),"ee-one-d","logging.conf"))
except Exception as e:
    logging.error("Cwd must be root of project directory")
logger = logging.Logger(__name__)


def drop_all_collections():
    # Connect to Milvus
    connections.connect(alias="default", host="localhost", port=19530)

    # Switch to the "default" database
    db.using_database(sys.argv[1])

    # Get all collection names
    collection_names = utility.list_collections()

    if not collection_names:
        logger.info("No collections found in the 'default' database.")
    else:
        logger.info(f"Found {len(collection_names)} collections.")

        # Drop each collection
        for name in collection_names:
            try:
                utility.drop_collection(name)
                logger.info(f"Dropped collection: {name}")
            except Exception as e:
                logger.error(f"Failed to drop collection {name}: {e}")

    # Disconnect from Milvus
    connections.disconnect("default")


if __name__ == "__main__":
    drop_all_collections()
    logger.info("Script execution completed.")
