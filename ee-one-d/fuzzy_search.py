"""
Module: fuzzy_search.py
Author: Akil Krishna, M Rishikesh, Sri Guru Datta Pisupati,Simhadri Adhit

This module provides a function, find_nearest_neighbors, for finding nearest neighbors of a query substring within a given string.

Usage:
1. Import the find_nearest_neighbors function from the fuzzy_search module.
2. Call the function with the query substring, string, distance threshold, length threshold, and number of nearest neighbors as arguments.
"""

import logging
from typing import List, Tuple

from base import SearchClass

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _sellers_dist(
    str1: str,
    str2: str,
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
        "Calculating Levenshtein distance between strings: %s and %s", str1, str2
    )
    logger.debug("Query Substring: %s", query_substring)
    logger.debug("Input String: %s", string)
    logger.debug("Distance Threshold: %s", distance_threshold)
    logger.debug("Length Threshold: %s", length_threshold)
    logger.debug("Number of Neighbors: %s", n)

    prev_row = list(range(len(str2) + 1))

    min_distance = float("inf")

    for i in range(len(str1)):
        curr_row = [i + 1]
        for j in range(len(str2)):
            cost = 0 if str1[i] == str2[j] else cost_sub
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
    n: int = 4,
    length_threshold: int = 0,
) -> List[Tuple[str, float]]:
    """
    Find the nearest neighbors of a query substring within a given string.

    Parameters
    ----------
    query_substring : str
        The query substring to find nearest neighbors for.
    string : str
        The string to search for nearest neighbors within.
    distance_threshold : int, optional
        The maximum allowable distance between query substring and its neighbors. Defaults to 2.
    n : int, optional
        The maximum number of nearest neighbors to return. Defaults to 4.
    length_threshold : int, optional
        The maximum allowable difference in length between query substring and its neighbors. Defaults to 0.

    Returns
    -------
    List[Tuple[str, float]]
        A list of tuples containing the nearest neighbors and their distances, sorted by distance.
    """
    nearest_neighbors = []

    for i in range(len(string)):
        for j in range(i + 1, len(string) + 1):
            substring = string[i:j]
            if (
                len(substring) < len(query) - length_threshold
                or len(substring) > len(query) + length_threshold
            ):
                continue

            distance = _sellers_dist(query, substring)

            if (distance <= distance_threshold) or (len(nearest_neighbors) < n):
                nearest_neighbors.append((substring, distance))

    nearest_neighbors.sort(key=lambda x: x[1])

    logger.debug("Nearest neighbors found: %s", nearest_neighbors)

    if n is not None:
        return nearest_neighbors[:n]
    else:
        return nearest_neighbors


class FuzzySearch(SearchClass):

    def __init__(self, input_str: str):
        self.input_str = input_str

    def __call__(
        self,
        string: str,
        distance_threshold: int = 2,
        n: int = 4,
        length_threshold: int = 0,
    ):
        return find_nearest_neighbors(
            self.input_str,
            string,
            distance_threshold,
            length_threshold,
            n,
        )


if __name__ == "__main__":

    query_substring = "apple"
    string = "ape apples banana ackle ssna pple"
    length_threshold = 3
    distance_threshold = 3
    n = 4
    nearest_neighbors_by_threshold = find_nearest_neighbors(
        query_substring,
        string,
        distance_threshold=distance_threshold,
        length_threshold=length_threshold,
    )
    nearest_neighbors_by_number = find_nearest_neighbors(
        query_substring, string, n=n, length_threshold=length_threshold
    )

    print(
        f"The nearest neighbors for '{query_substring}' within an edit distance of the {distance_threshold} and a length threshold of {length_threshold} in the string:"
    )
    for neighbor, distance in nearest_neighbors_by_threshold:
        print(f"Neighbor: '{neighbor}' (Edit Distance: {distance})")

    print(
        f"\n{n} nearest neighbors for '{query_substring}' with a length threshold of {length_threshold} in the string:"
    )
    for neighbor, distance in nearest_neighbors_by_number:
        print(f"Neighbor: '{neighbor}' (Edit Distance: {distance})")
