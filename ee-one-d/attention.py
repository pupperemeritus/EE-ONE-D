"""
Module: attention.py
Author: Akil Krishna, M Rishikesh, Sri Guru Datta Pisupati,Simhadri Adhit

This module provides a class, AttentionModel, for finding the most important tokens in a given input string using pre-trained language models as the default metric.

Dependencies:
- nltk
- torch
- transformers

Usage:
1. Initialize an instance of AttentionModel with an input string.
2. Call the get_weights method to retrieve and display the importance scores.
"""

import logging
from typing import List

import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from transformers import BertModel, BertTokenizer

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AttentionModel(torch.nn.Module):
    def __init__(self, input_string: str, model_name: str = "bert-base-uncased"):
        """
        Initializes the AttentionModel with the input_string and an optional model_name.

        Parameters
        ----------
            input_string : str
                The input string for the model.
            model_name : str
                The name of the pre-trained BERT model to use (default is "bert-base-uncased").

        Returns
        -------
            None
        """
        super(AttentionModel, self).__init__()
        self.input_string = input_string.lower()
        self.model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.lemmatizer = WordNetLemmatizer()

        self.stop_words = set(stopwords.words("english"))

    def preprocess_input(self) -> torch.Tensor:
        """
        Preprocesses the input string by removing stop words, lemmatizing, and tokenizing it.
        """
        self.tokens = word_tokenize(self.input_string)
        filtered_words = [
            self.lemmatizer.lemmatize(word)
            for word in self.tokens
            if word not in self.stop_words
        ]
        filtered_sentence = " ".join(filtered_words)

        self.filtered_tokens = self.tokenizer.tokenize(filtered_sentence)
        token_ids = self.tokenizer.convert_tokens_to_ids(self.filtered_tokens)
        tokens_tensor = torch.tensor([token_ids])

        logging.debug("Original Sentence: %s", self.input_string)
        logging.debug("Filtered and Lemmatized Sentence: %s", filtered_sentence)
        logging.debug("Tokenized Sentence: %s", tokens_tensor)
        logging.debug(self.filtered_tokens)

        return tokens_tensor

    def forward(self) -> torch.Tensor:
        """
        Performs forward pass of the model on the input tensor.
        """
        tokens_tensor = self.preprocess_input()
        with torch.no_grad():
            outputs = self.model(tokens_tensor)

        last_hidden_state = outputs.last_hidden_state
        attention_weights = torch.matmul(
            last_hidden_state, last_hidden_state.transpose(1, 2)
        )
        average_attention_weights = attention_weights.mean(dim=0)
        importance_scores = average_attention_weights.sum(dim=0)

        return importance_scores

    def get_weights(self) -> torch.Tensor:
        """
        Calculates the importance scores for each token in the input string.
        """
        self.importance_scores = self.forward()
        return self.importance_scores

    def get_n_words(self, n: int) -> List[str]:
        """
        Returns the top n most important tokens in the input string.
        """
        self.get_weights()
        top_n_indices = self.importance_scores.argsort(descending=True)[:n]

        top_n_tokens = [
            self.filtered_tokens[i]
            for i in top_n_indices
            if i < len(self.filtered_tokens)
        ]

        return top_n_tokens


if __name__ == "__main__":
    model = AttentionModel(input_string="A cat sat on the mat")
    top_tokens = model.get_n_words(2)
    print("Top tokens:", top_tokens)