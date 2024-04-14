import logging
from typing import List, Optional, Tuple
import nltk
from nltk.corpus import wordnet
import torch
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForSequenceClassification, AutoTokenizer
nltk.download('wordnet')
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class SemanticSearch:
    def __init__(
        self, model: str="togethercomputer/m2-bert-80M-32k-retrieval", tokenizer: str="bert-base-uncased", wordnet_synsets: List[nltk.corpus]=[],
         input_text: bool = True, max_seq_length: Optional[int]=1024
    ):
        """
        Initialize the class with the provided model name, tokenizer name, and WordNet synsets.

        Parameters
        ----------
        model : str
            The name of the model to be loaded.
        tokenizer : str
            The name of the tokenizer to be loaded.
        wordnet_synsets : List[wordnet.Synset]
            A list of WordNet synsets representing the corpus.
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
        self.wordnet_synsets = wordnet_synsets
        self.input_text = input_text

        logger.debug("SemanticSearch initialized with model: %s, tokenizer: %s", model, tokenizer)


    def find_semantic_neighbors(self, query: str, k: int) -> List[Tuple[str, float]]:
        """
        Find semantic neighbors for a given query in the WordNet synsets.

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

        query_synsets = self._get_synsets(query)

        query_embedding = self._get_embedding(query)
        semantic_neighbors = []
        for synset in self.wordnet_synsets:
            similarity = self._compute_similarity(query_synsets, synset)
            semantic_neighbors.append((synset.definition(), similarity))
            if len(semantic_neighbors) == k:
                break
        semantic_neighbors.sort(key=lambda x: x[1], reverse=True)

        logger.debug("Found semantic neighbors for query: %s", query)

        return semantic_neighbors

    def _get_synsets(self, word: str) -> List[nltk.corpus]:
        """
        Get the WordNet synsets for the input word.

        Parameters
        ----------
        word : str
            The input word.

        Returns
        -------
        List[wordnet.Synset]
            The list of WordNet synsets for the input word.
        """
        return wordnet.synsets(word)

    def _compute_similarity(self, synsets1: List[nltk.corpus], synset2: nltk.corpus) -> float:
        """
        Compute semantic similarity between two lists of synsets.

        Parameters
        ----------
        synsets1 : List[wordnet.Synset]
            The list of synsets for the first word.
        synset2 : wordnet.Synset
            The synset for the second word.

        Returns
        -------
        float
            The semantic similarity score.
        """
        max_similarity = 0
        for synset1 in synsets1:
            similarity = synset1.path_similarity(synset2)
            if similarity is not None and similarity > max_similarity:
                max_similarity = similarity
        return max_similarity

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
    query = "I like caffeine"
    k = 4
    # Get WordNet synsets for the query
    query_synsets = []
    for word in query.split():
        query_synsets.extend(wordnet.synsets(word))

    semantic_search = SemanticSearch(
        model_name, tokenizer_name, wordnet_synsets=query_synsets, input_text=False
    )

    # Test the find_semantic_neighbors method
    semantic_neighbors = semantic_search.find_semantic_neighbors(query, k)

    for neighbor, similarity in semantic_neighbors:
        print(f"Neighbor: {neighbor}")
        print(f"Similarity: {similarity}")
        print()
