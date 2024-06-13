import logging
from collections import defaultdict

from attention import *
from fuzzy_search import *
from pipeline import *
from semantic_search import *
from typographical_search import *

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)


class LFSS:

    def __init__(
        self,
        init_arg_dict: Dict,
        call_arg_dict: Dict,
        query: str,
        document: List[str],
        use_attention: bool = False,
    ):
        self.query = query
        self.document = document
        self.use_attention = use_attention
        self.class_list = [
            SemanticSearch,
            TypographicalNeighbors,
            AttentionModel,
            FuzzySearch,
        ]
        if not use_attention:
            self.class_list.pop(-2)

        self.pipeline = SearchPipeline(
            self.class_list, query, init_arg_dict, call_arg_dict
        )

    def __call__(self):
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
        init_arg_dict={},
        call_arg_dict={},
        query=input_string,
        document=document,
        use_attention=False,
    )
    print(lfss())
