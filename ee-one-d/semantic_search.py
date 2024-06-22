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
    """
    Class for finding semantic neighbors of a given input string using pre-trained language models as the default metric.

    Attributes:
        model : AutoModelForSequenceClassification
            The pre-trained model for semantic search.
        model_name : string
            The name of the pre-trained model to be used.
        max_seq_length : int
            The maximum sequence length.
        document : List[str]
            A list of strings representing the document.
        input_text : bool
            A flag indicating whether the input is text or not.
    """

    def __init__(
        self,
        query: str = "",
        model: str = "togethercomputer/m2-bert-80M-32k-retrieval",
        max_seq_length: Optional[int] = 1024,
    ):
        """
        Initialize the class with the provided model name, tokenizer name, and document.

        Parameters
        ----------
        model : str
            The name of the model to be loaded.
        tokenizer : str
            The name of the tokenizer to be loaded.
        document : List[str]
            A list of strings representing the document.
        max_seq_length : int, optional
            The maximum sequence length. Defaults to 1024.
        """
        logger.debug("Initializing SemanticSearch class")

        self.model = QueryDBModel(model=model, max_seq_length=max_seq_length)

        self.query = query
        logger.debug("SemanticSearch initialized with model: %s", model)

    def __call__(self, limit: int = 10) -> List[Tuple[str, float]]:
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
        logger.debug("Finding semantic neighbors for query: %s", query)
        semantic_neighbors = self.model.query(self.query, limit)
        logger.debug("Found semantic neighbors for query: %s", query)
        semantic_neighbors_text = []
        for semantic_neighbor in semantic_neighbors:
            semantic_neighbors_text.append(semantic_neighbor.entity.get("text"))

        return semantic_neighbors_text


if __name__ == "__main__":
    model_name = "togethercomputer/m2-bert-80M-32k-retrieval"
    tokenizer_name = "togethercomputer/m2-bert-80M-32k-retrieval"

    query = "caffeine"
    semantic_search = SemanticSearch(query=query, model=model_name)

    # Test the find_semantic_neighbors method
    k = 10
    semantic_neighbors = semantic_search(k)

    for neighbor in semantic_neighbors:
        print(f"Neighbor: {neighbor}")
