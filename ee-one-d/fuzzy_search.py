"""
Module: fuzzy_search.py
Author: Akil Krishna, M Rishikesh, Sri Guru Datta Pisupati,Simhadri Adhit

This module provides a function, find_nearest_neighbors, for finding nearest neighbors of a query substring within a given string.

Usage:
1. Import the find_nearest_neighbors function from the fuzzy_search module.
2. Call the function with the query substring, string, distance threshold, length threshold, and number of nearest neighbors as arguments.
"""

import logging
import logging.config
import os
import re
from typing import List, Tuple

from base import SearchClass
from nltk import tokenize

try:
    logging.config.fileConfig(os.path.join(os.getcwd(), "ee-one-d", "logging.conf"))
except Exception as e:
    logging.error("Cwd must be root of project directory")
logger = logging.Logger(__name__)


def _sellers_dist(
    query_substring: str,
    input_string: str,
    cost_sub: float = 0.5,
    cost_ins: float = 1,
    cost_del: float = 1,
) -> float:
    """
    A function to calculate the Levenshtein distance between two strings.

    Parameters
    ----------
    str1 : str
        The first input string
    str2 : str
        The second input string
    cost_sub : float, optional
        The cost of substitution (default is 0.5)
    cost_ins : float, optional
        The cost of insertion (default is 1)
    cost_del : float, optional
        The cost of deletion (default is 1)

    Returns
    -------
    float
        The calculated Levenshtein distance
    """
    logger.debug(
        "Calculating Levenshtein distance between strings: %s and %s",
        query_substring,
        input_string,
    )
    logger.debug("Query Substring: %s", query_substring)
    logger.debug("Input String: %s", input_string)

    prev_row = list(range(len(input_string) + 1))

    min_distance = float("inf")

    for i in range(len(query_substring)):
        curr_row = [i + 1]
        for j in range(len(input_string)):
            cost = 0 if query_substring[i] == input_string[j] else cost_sub
            curr_row.append(
                min(
                    curr_row[j] + cost_ins,
                    prev_row[j + 1] + cost_del,
                    prev_row[j] + cost,
                )
            )

        if curr_row[-1] < min_distance:
            min_distance = curr_row[-1]

        prev_row = curr_row

    logger.debug("Levenshtein distance calculated: %f", min_distance)

    return float(min_distance)


def find_nearest_neighbors(
    query: str,
    string: str,
    distance_threshold: int = 2,
    length_threshold: int = 3,
    limit: int = 4,
) -> List[Tuple[str, float, int, int]]:
    nearest_neighbors = []

    # Split the string into words
    words = re.findall(r"\b\w+\b", string)

    for i, word in enumerate(words):
        if (
            len(word) < len(query) - length_threshold
            or len(word) > len(query) + length_threshold
        ):
            continue

        distance = _sellers_dist(query, word)

        if (distance <= distance_threshold) or (len(nearest_neighbors) < limit):
            # Calculate start and end indices for context
            start = max(0, i - 2)
            end = min(len(words), i + 3)
            context = " ".join(words[start:end])
            nearest_neighbors.append((word, distance, context))

    nearest_neighbors.sort(key=lambda x: x[1])

    logger.info("Nearest neighbors found: %s", nearest_neighbors[:limit])

    if limit is not None:
        return nearest_neighbors[:limit]
    else:
        return nearest_neighbors


class FuzzySearch(SearchClass):
    def __init__(self, query: str, document: List[str]) -> None:
        self.query = query
        self.document = document

    def __call__(
        self,
        distance_threshold: int = 2,
        limit: int = 4,
        length_threshold: int = 3,
    ) -> List:
        result = []
        for sentence_index, sentence in enumerate(self.document):

            neighbors = find_nearest_neighbors(
                self.query,
                sentence,
                distance_threshold,
                length_threshold,
                limit,
            )
            if neighbors:
                # Only take the best match for each sentence
                best_match = min(neighbors, key=lambda x: x[1])
                word, distance, context = best_match
                result.append(
                    {
                        "sentence_index": sentence_index,
                        "sentence": sentence,
                        "word": word,
                        "distance": distance,
                        "context": context,
                    }
                )

        # Sort results by distance and limit to max_results
        result.sort(key=lambda x: x["distance"])
        return result


if __name__ == "__main__":
    query = "apple"
    document = [
        "I like to eat apples and bananas.",
        "The apple does't fall far from the tree.",
        "An ape ate an apple.",
        "This sentence has no relevant words.",
    ]

    fuzzy_search = FuzzySearch(query, document)
    results = fuzzy_search(distance_threshold=2, limit=5)

    print(f"Fuzzy search results for '{query}':")
    for result in results:
        print(f"\nIn sentence: '{result['sentence']}'")
        print(f"\nFound: '{result['word']}'\n(Distance: {result['distance']})")
        print(f"\nContext: '{result['context']}'")
