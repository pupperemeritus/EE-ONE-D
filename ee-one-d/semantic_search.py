from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import torch
from typing import List, Tuple, Optional
from nltk.tokenize import sent_tokenize
import logging

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


class SemanticSearch:
    def __init__(
        self, model: str, tokenizer: str, document: List[str],
         input_text: bool = True, max_seq_length: Optional[int]=1024
    ):
        """
        Initialize the class with the provided model name, tokenizer name, and document.
        
        Parameters:
            model (str): The name of the model to be loaded.
            tokenizer (str): The name of the tokenizer to be loaded.
            document (List[str]): A list of strings representing the document.
            input_text (bool, optional): A flag indicating whether the input is text or not. Defaults to True.
            max_seq_length (int, optional): The maximum sequence length. Defaults to 1024.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model, trust_remote_code=True
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer, trust_remote_code=True, model_max_length=self.max_seq_length
        )
        if input_text:
            self.document = sent_tokenize(document)
        else:
            self.document = document

    def find_semantic_neighbors(self, query: str, k: int) -> List[Tuple[str, float]]:
        """
        Find semantic neighbors for a given query in the document.
        
        Parameters:
            query (str): The query string for which to find semantic neighbors.
            k (int): The number of semantic neighbors to retrieve.
        
        Returns:
            List[Tuple[str, float]]: A list of tuples containing the text of the semantic neighbor and its similarity score.
        """
        query_embedding = self._get_embedding(query)
        semantic_neighbors = []

        for text in self.document:
            corpus_embedding = self._get_embedding(text).to(self.device)
            similarity = F.cosine_similarity(query_embedding, corpus_embedding).item()
            semantic_neighbors.append((text, similarity))
            if len(semantic_neighbors) == k:
                break

        semantic_neighbors.sort(key=lambda x: x[1], reverse=True)

        return semantic_neighbors

    def _get_embedding(self, text: str) -> torch.Tensor:
        """
        Get the embedding for the input text using the tokenizer and model.
        
        Args:
            text (str): The input text for which the embedding needs to be generated.
        
        Returns:
            torch.Tensor: The embedding for the input text.
        """
        input_ids = self.tokenizer(
            [text],
            return_tensors="pt",
            padding="max_length",
            return_token_type_ids=False,
            truncation=True,
            max_length=self.max_seq_length,
        ).to(self.device)
        outputs = self.model(**input_ids)
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
    query = "tea"
    k = 4
    semantic_neighbors = semantic_search.find_semantic_neighbors(query, k)

    for neighbor, similarity in semantic_neighbors:
        print(f"Neighbor: {neighbor}")
        print(f"Similarity: {similarity}")
        print()
