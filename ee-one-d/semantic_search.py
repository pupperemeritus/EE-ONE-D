from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import torch
from typing import List, Tuple
from nltk.tokenize import sent_tokenize


class SemanticSearch:
    def __init__(
        self, model: str, tokenizer: str, document: List[str], input_text: bool = True
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_seq_length = 512
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
        query_embedding = self._get_embedding(query)
        document_embeddings = [
            self._get_embedding(text).to(self.device) for text in self.document
        ]

        similarities = [
            F.cosine_similarity(query_embedding, corpus_embedding).item()
            for corpus_embedding in document_embeddings
        ]
        semantic_neighbors = sorted(
            zip(self.document, similarities), key=lambda x: x[1], reverse=True
        )[:k]

        return semantic_neighbors

    def _get_embedding(self, text: str) -> torch.Tensor:
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
    model_name = "togethercomputer/m2-bert-80M-2k-retrieval"
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
