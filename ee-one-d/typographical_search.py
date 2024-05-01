"""
Module: typographical_neighbors.py
Author: Simhadri Adhit,Akil krisna,Sri Guru Datta Pisupati,M Rishikesh

This module provides a class, TypographicalNeighbors, for finding typographical neighbors of a given input string using pre-trained language models and edit distance as the default metric.

Dependencies:
- nltk
- numpy
- torch
- transformers

Usage:
1. Initialize an instance of TypographicalNeighbors with an input string.
2. Call the find_typographical_neighbors method to retrieve and display typographical neighbors.
"""
import logging
from typing import List

import nltk
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class TypographicalNeighbors:
  """
    A class for finding typographical neighbors of a given input string.

    Attributes:
        input_string (str): The input string to be used for typographical search.
        metric (str): The metric used for calculating the distance between strings.
        model_name (str): The name of the pre-trained model to be used.
        device (torch.device): The device (CPU or CUDA) used for computations.
        tokenizer (AutoTokenizer): The tokenizer for tokenizing input strings.
        model (AutoModel): The pre-trained model for encoding input strings.
        typographical_neighbors (List[str]): A list to store typographical neighbors.
    """

    def __init__(self, input_string: str, metric: str = "edit_distance", model_name:str ="bert-base-uncased"):
        """
        Initializes an instance of the class with the given input string, metric, and model name.

        Parameters:
            input_string (str): The input string to be used for typographical search.
            metric (str, optional): The metric to be used for calculating the distance between strings.
                Defaults to "edit_distance".
            model_name (str, optional): The name of the pre-trained model to be used.
                Defaults to "bert-base-uncased".
        """
        logger.debug("Initializing TypographicalNeighbors class")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metric = nltk.edit_distance
        
        self.input_string = input_string
        self.typographical_neighbors = []
        
        self.model_name = model_name
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name).to(self.device)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)

    def tokenize_inputs(self):
        """
        Tokenizes the input string using the tokenizer, generates the input representation using the model,
        and returns the tokenized inputs and the input representation.
        """
        logger.debug("Tokenizing inputs")
        tokenized_inputs = self.tokenizer.encode_plus(self.input_string, return_tensors="pt")
        input_representation = self.model(**tokenized_inputs).to(self.device).last_hidden_state.mean(dim=1)
        return tokenized_inputs, input_representation

    def get_neighbors(self):
        """
        Retrieves the typographical neighbors of the input string by iterating through the words in the NLTK corpus
        and checking for words that have a metric distance of 1 from the input string.

        Returns:
            typographical_neighbors (List[str]): A list of typographical neighbors of the input string.
        """
        for neighbor in nltk.corpus.words.words():
            if self.metric(neighbor, self.input_string) == 1:
                self.typographical_neighbors.append(neighbor)
        return self.typographical_neighbors

    def encode_neighbors(self)-> List[np.ndarray]:
        """
        Encodes the typographical neighbors of the current instance using a tokenizer and a model.

        Returns:
            encoded_neighbors (List[np.ndarray]): A list of encoded neighbors, where each neighbor is represented as a numpy array.
        """
        logger.debug("Encoding neighbors")

        encoded_neighbors = []
        for neighbor in self.typographical_neighbors:
            tokenized_neighbor = self.tokenizer.encode_plus(neighbor, return_tensors="pt")
            with torch.no_grad():
                output = self.model(**tokenized_neighbor)
            encoded_neighbor = output.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
            encoded_neighbors.append(encoded_neighbor)
        
        logger.debug("Encoded neighbors")

        return encoded_neighbors

    def get_similar_neighbors(self)-> List[str]:
        """
        Calculates the cosine similarity between the input representation and the encoded neighbors.

        Returns:
            sorted_neighbors (List[str]): A list of neighbors sorted in descending order of similarity.
        """
        logger.debug("Calculating similar neighbours")

        tokenized_inputs, input_representation = self.tokenize_inputs()
        encoded_neighbors = self.encode_neighbors()
        similarities = []
        for neighbor_embedding in encoded_neighbors:
            similarity_score = np.dot(input_representation.detach().numpy(), neighbor_embedding.T)
            cosine_similarity = similarity_score / (np.linalg.norm(input_representation.detach().numpy()) * np.linalg.norm(neighbor_embedding))
            similarities.append(cosine_similarity)
        sorted_neighbors = [x for _, x in sorted(zip(similarities, self.typographical_neighbors), reverse=True)]

        logger.debug("Calculated similar neighbours")

        return sorted_neighbors

    def find_typographical_neighbors(self, n: int = 10) -> List[str]:
        """
        Display the typographical neighbors of the input string up to a specified number.

        Parameters:
            n (int): The number of typographical neighbors to display.

        Returns:
            typographical_neighbors_list (List[str]): A list of typographical neighbors.
        """
        sorted_neighbors = self.get_similar_neighbors()
        typographical_neighbors_list = sorted_neighbors[:n]

        return typographical_neighbors_list
    
    

if __name__=="__main__":
    # Example usage:
    input_string = "dogs"
    tn = TypographicalNeighbors(input_string)
    tn.get_neighbors()
    tn.find_typographical_neighbors()
