import logging
from typing import List
import nltk
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from attention import AttentionModel

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TypographicalNeighbors:
    """
    A class for finding typographical neighbors of a given input string.

    Attributes:
        input_string : str
            The input string to be used for typographical search.
        metric : str
            The metric used for calculating the distance between strings.
        model_name : str
            The name of the pre-trained model to be used.
        tokenizer : AutoTokenizer
            The tokenizer for tokenizing input strings.
        model : AutoModel
            The pre-trained model for encoding input strings.
        typographical_neighbors : List[str]
            A list to store typographical neighbors.
    """

    def __init__(self, input_string: str, model_name: str = "bert-base-uncased"):
        """
        Initializes an instance of the class with the given input string, metric, and model name.

        Parameters
        ----------
        input_string : str
            The input string to be used for typographical search.
        model_name : str, optional
            The name of the pre-trained model to be used.
            Defaults to "bert-base-uncased".
        """
        logger.debug("Initializing TypographicalNeighbors class")

        self.metric = nltk.edit_distance
        self.input_string = input_string
        self.typographical_neighbors = []
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.attention = AttentionModel(self.input_string, self.model_name)

        if len(self.input_string.split(" ")) > 1: 
            self.input_string = self.attention.get_n_words(3)



    def tokenize_inputs(self): # --> done
        """
        Tokenizes the input string using the tokenizer, generates the input representation using the model,
        and returns the tokenized inputs and the input representation.
        """
        logger.debug("Tokenizing inputs")

        tokenized_inputs_list = []
        tokenized_representations_list = []
        for input_string in self.input_string:
            tokenized_inputs = self.tokenizer.encode_plus(
                input_string, return_tensors="pt"
            )
            input_representation = self.model(**tokenized_inputs).last_hidden_state.mean(
                dim=1
            )
            tokenized_inputs_list.append(tokenized_inputs)
            tokenized_representations_list.append(input_representation)

        return tokenized_inputs_list, input_representation
    
    def get_neighbors(self): # --> done 
        """
        Retrieves the typographical neighbors of the input string by iterating through the words in the NLTK corpus
        and checking for words that have a metric distance of 1 from the input string.

        Returns
        -------
        typographical_neighbors : List[str]
            A list of typographical neighbors of the input string.
        """
        logger.debug("Started get_neighbors function")
        for input_string in self.input_string:    

            string_neighbors = []
            for neighbor in nltk.corpus.words.words():
                if self.metric(neighbor, input_string) == 1:
                    string_neighbors.append(neighbor)
            self.typographical_neighbors.append(string_neighbors)
            
        logging.debug("Ended get_neighbors function")
        
        return self.typographical_neighbors
        
    def encode_neighbors(self): # -> partially donw
        """
        Encodes the typographical neighbors of the current instance using a tokenizer and a model.

        Returns
        -------
        encoded_neighbors : List[np.ndarray]
            A list of encoded neighbors, where each neighbor is represented as a numpy array.
        """
        logger.debug("started Encoding neighbors")

        neighbors = self.get_neighbors()
        print(f"returned from self.get_neighbors(): {neighbors}")

        encoded_neighbors = []
        
        for typographical_neighbors in neighbors:
            encoded = []
            for neighbor in typographical_neighbors:
                tokenized_neighbor = self.tokenizer.encode_plus(
                    neighbor, return_tensors="pt"
                )
                with torch.no_grad():
                    output = self.model(**tokenized_neighbor)
                encoded_neighbor = (
                    output.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
                )
                encoded.append(encoded_neighbor)
            encoded_neighbors.append(encoded)

        logger.debug("ended encoded neighbors")

        return encoded_neighbors

    def get_similar_neighbors(self) -> List[str]:
        """
        Calculates the cosine similarity between the input representation and the encoded neighbors.

        Returns
        -------
        sorted_neighbors : List[str]
            A list of neighbors sorted in descending order of similarity.
        """
        logger.debug("Calculating similar neighbours")

        tokenized_inputs, input_representation = self.tokenize_inputs()
        encoded_neighbors = self.encode_neighbors()
        similarities = []
        for neighbor_embedding in encoded_neighbors:
            similarity_score = np.dot(
                input_representation.detach().numpy(), neighbor_embedding.T
            )
            cosine_similarity = similarity_score / (
                np.linalg.norm(input_representation.detach().numpy())
                * np.linalg.norm(neighbor_embedding)
            )
            similarities.append(cosine_similarity)

        sorted_neighbors = [
            x
            for _, x in sorted(
                zip(similarities, self.typographical_neighbors), reverse=True
            )
        ]

        logger.debug(sorted_neighbors)
        logger.debug("Calculated similar neighbours")

        return sorted_neighbors

    def find_typographical_neighbors(self, n: int = 10) -> List[str]:
        """
        Display the typographical neighbors of the input string up to a specified number.

        Parameters
        ----------
        n : int
            The number of typographical neighbors to display.

        Returns
        -------
        typographical_neighbors_list : List[str]
            A list of typographical neighbors.
        """
        logging.debug("Finding typographical neighbors")

        sorted_neighbors = np.unique(self.get_similar_neighbors())
        typographical_neighbors_list = sorted_neighbors[:n]

        return typographical_neighbors_list


if __name__ == "__main__":
    # Example usage:
    input_string = "A cat sat on the mat"
    tn = TypographicalNeighbors(input_string)
    print(tn.encode_neighbors())
    # print(tn.find_typographical_neighbors(4))
