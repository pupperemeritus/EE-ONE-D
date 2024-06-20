import logging
from collections import defaultdict
from typing import Dict, List

from attention import AttentionModel
from fuzzy_search import FuzzySearch
from pipeline import SearchPipeline
from semantic_search import SemanticSearch
from typographical_search import TypographicalNeighbors

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LFSS:

    def __init__(
        self,
        init_arg_dict: Dict,
        call_arg_dict: Dict,
        query: str,
        document: List[str],
        use_attention: bool = False,
    ):
        
        logger.debug(
            f"Initializing LFSS with input parameters. Query: {query}, Document: {document}, Use Attention: {use_attention}"
        )

        self.query = query
        self.document = document
        self.use_attention = use_attention
        self.class_list = [
            TypographicalNeighbors,
            SemanticSearch,
            AttentionModel,
            FuzzySearch,
        ]

        logger.debug(f"Original class list: {self.class_list}")

        if not use_attention:
            logger.debug(f"Removing AttentionModel from class list")
            self.class_list.pop(-2)

        logger.debug(f"Modified class list: {self.class_list}")

        self.pipeline = SearchPipeline(
            self.class_list, query, "query", init_arg_dict, call_arg_dict
        )

    def __call__(self):
        logger.debug("Running pipeline")
        return self.pipeline.run()


if __name__ == "__main__":
    input_string = "dog"

    document = [
        "The loyal dog waited patiently by the door.",
        "She enjoyed taking her furry companion for long walks.",
        "The puppy's playful antics brought joy to the family.",
        "He trained his canine friend to do impressive tricks.",
        "The barking of the neighbor's dog could be heard in the distance.",
        "The sun was setting over the horizon, painting the sky in hues of orange and pink.",
        "She sipped her coffee slowly, savoring the rich aroma.",
        "The students eagerly listened to the professor's lecture on quantum mechanics.",
        "The scent of fresh flowers wafted through the open window.",
        "The city bustled with activity as people hurried to their destinations.",
        "The old oak tree stood majestic against the backdrop of the clear blue sky.",
        "The soothing sound of rain pattering on the roof lulled her to sleep.",
        "The aroma of freshly baked bread filled the quaint bakery.",
        "The children laughed and played in the park, their voices ringing with joy.",
        "The bookshelf was filled with a collection of novels spanning various genres.",
    ]

    lfss = LFSS(
        init_arg_dict=[{}, {"k": 4}, {}],
        call_arg_dict=[{}, {}, {}],
        query=input_string,
        document=document,
        use_attention=False,
    )
    print(lfss())
