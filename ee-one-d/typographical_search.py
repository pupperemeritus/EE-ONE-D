import numpy as np
import torch
import nltk
from typing import List
from transformers import AutoTokenizer, AutoModel

class TypographicalNeighbors:
    def __init__(self, input_string: str, metric: str = "edit_distance"):
        self.metric = nltk.edit_distance
        self.input_string = input_string
        self.typographical_neighbors = []

        self.model_name = 'bert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

    def tokenize_inputs(self):
        tokenized_inputs = self.tokenizer.encode_plus(self.input_string, return_tensors="pt")
        input_representation = self.model(**tokenized_inputs).last_hidden_state.mean(dim=1)
        return tokenized_inputs, input_representation

    def get_neighbors(self):
        for neighbor in nltk.corpus.words.words():
            if self.metric(neighbor, self.input_string) == 1:
                self.typographical_neighbors.append(neighbor)
        return self.typographical_neighbors

    def encode_neighbors(self):
        encoded_neighbors = []
        for neighbor in self.typographical_neighbors:
            tokenized_neighbor = self.tokenizer.encode_plus(neighbor, return_tensors="pt")
            with torch.no_grad():
                output = self.model(**tokenized_neighbor)
            encoded_neighbor = output.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
            encoded_neighbors.append(encoded_neighbor)
        return encoded_neighbors

    def get_similar_neighbors(self):
        tokenized_inputs, input_representation = self.tokenize_inputs()
        encoded_neighbors = self.encode_neighbors()
        similarities = []
        for neighbor_embedding in encoded_neighbors:
            similarity_score = np.dot(input_representation.detach().numpy(), neighbor_embedding.T)
            cosine_similarity = similarity_score / (np.linalg.norm(input_representation.detach().numpy()) * np.linalg.norm(neighbor_embedding))
            similarities.append(cosine_similarity)
        sorted_neighbors = [x for _, x in sorted(zip(similarities, self.typographical_neighbors), reverse=True)]
        return sorted_neighbors

    def display_typogrphical_neighbors(self, n: int = 10):
        sorted_neighbors = self.get_similar_neighbors()
        typographical_neighbors_list = sorted_neighbors[:n]

        print(f"Input string: {self.input_string}")
        print(f"Typographical neighbors: {typographical_neighbors_list}")

# Example usage:
input_string = "dogs"
tn = TypographicalNeighbors(input_string)
tn.get_neighbors()
tn.display_typogrphical_neighbors()
