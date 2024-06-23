"""
Module: semantic_search.py
Author: Akil Krishna, M Rishikesh, Sri Guru Datta Pisupati,Simhadri Adhit


This module provides a class, SemanticSearch, for finding semantic neighbors of a given input string using pre-trained language models as the default metric.

Dependencies:
- nltk
- numpy
- torch
- transformers

Usage:
1. Initialize an instance of SemanticSearch with an input string.
2. Call the find_semantic_neighbors method to retrieve and display semantic neighbors.
"""

import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from base import SearchClass
from models import QueryDBModel

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SemanticSearch(SearchClass):
    def __init__(
        self,
        query: str = "",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the class with the provided model name, tokenizer name, and document.

        Parameters
        ----------
        model_name : str
            The name of the model to be loaded.

        """
        logger.debug("Initializing SemanticSearch class")


        self.model = QueryDBModel(model=model_name)

        self.query = query
        logger.debug("SemanticSearch initialized with model: %s", model_name)

    def __call__(self, limit: int = 5) -> List[Tuple[str, float]]:
        """
        Find semantic neighbors for a given query in the document.

        Parameters
        ----------
        query : str
            The query string for which to find semantic neighbors.
        k : int
            The number of semantic neighbors to retrieve.

        Returns
        -------
        List[Tuple[str, float]]
            A list of tuples containing the text of the semantic neighbor and its similarity score.
        """
        logger.debug("Finding semantic neighbors for query: %s", self.query)
        semantic_neighbors = self.model.query(self.query, limit)
        logger.debug("Found semantic neighbors for query: %s", self.query)
        semantic_neighbors_text = []
        for semantic_neighbor in semantic_neighbors:
            semantic_neighbors_text.append(semantic_neighbor.entity.get("text"))

        return semantic_neighbors_text


if __name__ == "__main__":
    query = "caffeine"
    semantic_search = SemanticSearch(query=query)

    # Test the find_semantic_neighbors method
    k = 10
    semantic_neighbors = semantic_search(k)

    for neighbor in semantic_neighbors:
        print(f"Neighbor: {neighbor}")
