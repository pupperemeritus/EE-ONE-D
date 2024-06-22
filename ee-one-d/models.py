import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import nltk
import torch
import torch.nn.functional as F
from base import SearchClass
from nltk.tokenize import sent_tokenize
from pymilvus import MilvusClient, connections, db, utility
from pymilvus.model.dense import SentenceTransformerEmbeddingFunction
from pymilvus.orm.collection import Collection, CollectionSchema, DataType, FieldSchema
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.Logger(__name__)


class ModelQuery(ABC):
    def __init__(self, model: str, max_seq_length: int): ...

    def query(self, query: str, limit: int) -> List[dict]: ...


class QueryDBModel(ModelQuery):
    def __init__(
        self,
        model: str,
        max_seq_length: str,
        db_alias: str = "default",
        db_host: str = "localhost",
        db_name: str = "eeoned",
        db_collection_name: str = "semantic_embeddings",
        db_port=19530,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            connections.connect(alias=db_alias, host=db_host, port=db_port)
            self.client = MilvusClient()
            logger.debug(
                f"Connected to DB:\n Alias: {db_alias}\nHost: {db_host}\nPort: {db_port}"
            )

            db.using_database(db_name)
            logger.debug(f"Using Database {db_name}")

            self.collection = Collection(db_collection_name)

            self.embedding_func = SentenceTransformerEmbeddingFunction(
                device=self.device, trust_remote_code=True, batch_size=512
            )
            logger.debug(f"Initialized embedding function")

            schema = CollectionSchema(
                [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=256),
                    FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=256),
                ]
            )

        except Exception as e:
            logger.error(e)

    def query(self, query: str, limit: int = 10):
        encoded_query = self.embedding_func.encode_queries([query])
        results = self.collection.search(
            data=encoded_query,
            anns_field="vector",
            param={"metric_type": "L2"},
            limit=limit + 1,
            output_fields=["text"],
        )

        if logger.getEffectiveLevel() <= logging.DEBUG:
            for hits in results:
                for hit in hits:
                    logger.debug(
                        f"ID: {hit.id}, Distance: {hit.distance}, Text: {hit.entity.get('text')}"
                    )
        if results:
            (
                results[0].pop(0)
                if results[0][0].entity.get("text") == query
                else results[0].pop(-1)
            )

        return results[0]
