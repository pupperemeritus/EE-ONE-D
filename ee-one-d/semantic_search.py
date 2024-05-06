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
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class SemanticSearch:
    def __init__(
        self, model: str="togethercomputer/m2-bert-80M-32k-retrieval", tokenizer: str="bert-base-uncased", document: List[str]=[],
         input_text: bool = True, max_seq_length: Optional[int]=1024
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
        input_text : bool, optional
            A flag indicating whether the input is text or not. Defaults to True.
        max_seq_length : int, optional
            The maximum sequence length. Defaults to 1024.
        """
        logger.debug("Initializing SemanticSearch class")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_seq_length = max_seq_length
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model, trust_remote_code=True
        ).to(self.device)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer, trust_remote_code=True, model_max_length=self.max_seq_length
            )
        except:
            logging.error("Failed to load tokenizer, too large of a sequence length.")
        if input_text:
            self.document = sent_tokenize(document)
        else:
            self.document = document

        logger.debug("SemanticSearch initialized with model: %s, tokenizer: %s", model, tokenizer)


    def find_semantic_neighbors(self, query: str, k: int) -> List[Tuple[str, float]]:
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

        query_embedding = self._get_embedding(query)
        semantic_neighbors = []
        for text in self.document:
            corpus_embedding = self._get_embedding(text).to(self.device)
            similarity = F.cosine_similarity(query_embedding, corpus_embedding).item()
            semantic_neighbors.append((text, similarity))
            if len(semantic_neighbors) == k:
                break
        semantic_neighbors.sort(key=lambda x: x[1], reverse=True)

        logger.debug("Found semantic neighbors for query: %s", query)

        return semantic_neighbors

    def _get_embedding(self, text: str) -> torch.Tensor:
        """
        Get the embedding for the input text using the tokenizer and model.

        Parameters
        ----------
        text : str
            The input text for which the embedding needs to be generated.
            
        Returns
        -------
        torch.Tensor
            The embedding for the input text.
        """
        logger.debug("Generating embedding for text: %s", text)
        input_ids = self.tokenizer(
            [text],
            return_tensors="pt",
            padding="max_length",
            return_token_type_ids=False,
            truncation=True,
            max_length=self.max_seq_length,
        ).to(self.device)

        try:
            outputs = self.model(**input_ids)
        except Exception as e:
            logging.error("Failed to generate embedding. Out of memory")
            raise e

        logger.debug("Generated embedding for text: %s", text)

        return outputs["sentence_embedding"]



if __name__ == "__main__":
    model_name = "togethercomputer/m2-bert-80M-32k-retrieval"
    tokenizer_name = "bert-base-uncased"
    document = [
        "Every morning, I make a cup of coffee to start my day.",
        "I enjoy reading books to relax and unwind.",
        "I love going for hikes in the beautiful outdoors.",
        "Cooking is a fun way to experiment with different recipes.",
        "I enjoy playing video games in my free time.",
    ]

    semantic_search = SemanticSearch(
        model_name, tokenizer_name, document, input_text=False
    )

    # Test the find_semantic_neighbors method
    query = "I like caffeine"
    k = 4
    semantic_neighbors = semantic_search.find_semantic_neighbors(query, k)

    for neighbor, similarity in semantic_neighbors:
        print(f"Neighbor: {neighbor}")
        print(f"Similarity: {similarity}")
        print()
