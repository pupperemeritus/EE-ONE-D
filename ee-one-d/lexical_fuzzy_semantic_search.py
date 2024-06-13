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
