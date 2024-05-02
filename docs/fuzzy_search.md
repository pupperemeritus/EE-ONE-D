
## About this module
- The `fuzzy_search.py` module was built to find the fuzzy/approximate words to a given string input. The module will be tied together with `typpgraphical.py` and `semantic_search.py` to find the nearest neighbors.
- The words that are used for finding the neighbors will be taken from the given document. The metric used to determine the closeness between two strings is  **Levenshtein distance**.

- ## Details
	- **Inputs**: `a string/word` and `a document`.
	- **Outputs**: `A list of words that contain the nearest neighbors to the given word.`
	- **Requirements**: `None`
	
- The module has the following functions: 
	- `sellers_dist(str1: str, str2: str, cost_sub: float=0.5, cost_ins: float=1, cost_del=1) -> float`is a mathematical representation of the **edit distance (levenshtein distance)**
	- `find_nearest_neighbors(query_substring: str, string: str, distance_threshold: int = 2, n: int = 4, length_threshold: int = 0) -> List[Tuple[str, float]]:` Uses the `sellers_dist` function as a metric to find the `n` nearest neighbors to the given sequence.
	
## How to run this module
- To play around this module, run the following code:
```
query_substring = "apple"
string = "ape apples banana ackle ssna pple"
length_threshold = 3
distance_threshold = 3
n=4

nearest_neighbors_by_threshold = find_nearest_neighbors(query_substring, string,distance_threshold=distance_threshold, length_threshold=length_threshold)

nearest_neighbors_by_number = find_nearest_neighbors(query_substring, string,n=n, length_threshold=length_threshold)

print(f"The nearest neighbors for '{query_substring}' within an edit distance of the {distance_threshold} and a length threshold of {length_threshold} in the string:")

for neighbor, distance in nearest_neighbors_by_threshold:
     print(f"Neighbor: '{neighbor}' (Edit Distance: {distance})")

print(f"\n{n} nearest neighbors for '{query_substring}' with a length threshold of {length_threshold} in the string:")

for neighbor, distance in nearest_neighbors_by_number:
     print(f"Neighbor: '{neighbor}' (Edit Distance: {distance})"
```

## What are we adding to this module:
    - As of now, I don't know