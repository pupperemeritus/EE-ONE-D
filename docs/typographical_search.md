## A Laymans Description
This Typographical Python module helps find words similar to a given input word. It works by comparing the input word with a list of words from a large corpus. It uses advanced language models, in this case, *bert-base-uncased* for tokenisation and encoding of text and to identify words that have a resemblance to the input word. It then returns a list of similar words, sorted by how closely they match the input word.

## What are we doing?

### Packages Overview
- **logging** (used for logging messages to provide insights into the execution flow and debugging.)
- **typing** (provides support for defining type hints in Python code.)
- **nltk**
- **numpy**
- **torch**
- **transformers** (Hugging Face's transformers library)

### Input

input_string `(str)`: The input string for which typographical neighbors are to be found.\
metric `(str, optional)`: The metric used to calculate the distance between strings and it's set to *edit_distance* by default.\
model_name `(str, optional)`: The name of the pre-trained language model to be used and it's set to *bert-base-uncased* by default.

### Output

typographical_neighbors_list `(List[str])`: A list of typographical neighbors of the input string, sorted by similarity.

Initializing the model: `self.model = AutoModel.from_pretrained(self.model_name).to(self.device)`

## A brief overview of the functions used:

**__init__**`(self, input_string: str, metric: str = "edit_distance", model_name: str = "bert-base-uncased")`\
Initializes the class with the given input string, metric, and pre-trained model name.
Sets up the tokenizer and model for **NLP tasks**.\
\
The tasks in question are:

1. Tokenization: *Breaking down the input text into smaller units or subwords per se.*
2. Word Similarity Calculation: *Comparing words to find those that are similar to the input word.*
3. Encoding Text: *Converting words or sentences into numerical representations that can be understood by machine learning models.*
4. Cosine Similarity Calculation: *Measuring the similarity between two vectors representing text embeddings.*
5. Neighbor Retrieval: *Searching through a corpus of words to find those that are similar to the input word, often using metrics like edit distance.*

**tokenize_inputs**`(self)`\
Tokenizes the input string using the tokenizer.
Generates the input representation using the pre-trained model.
Returns tokenized inputs and the input representation.

**get_neighbors**`(self)`\
Retrieves typographical neighbors of the input string from the NLTK corpus.
Neighbors are words with a metric distance of 1 from the input string.
Returns a list of typographical neighbors.

**encode_neighbors**`(self) -> List[np.ndarray]`\
Encodes typographical neighbors using the tokenizer and the pre-trained model.
Returns a list of encoded neighbors, each represented as a numpy array.

**get_similar_neighbors**`(self) -> List[str]`\
Calculates the cosine similarity between the input representation and the encoded neighbors.
Sorts the neighbors in descending order of similarity.
Returns a list of similar neighbors.

**find_typographical_neighbors**`(self, n: int = 10) -> List[str]`\
Displays typographical neighbors of the input string up to a specified number.
Calls **get_similar_neighbors()** to retrieve similar neighbors.
Returns a list of typographical neighbors.


## Return Types of Functions
- tokenize_inputs(): *Returns tokenized inputs and the input representation.*
- get_neighbors(): *Returns a list of typographical neighbors.*
- encode_neighbors(): *Returns a list of encoded neighbors, each represented as a numpy array.*
- get_similar_neighbors(): *Returns a list of similar neighbors.*
- find_typographical_neighbors(): *Returns a list of typographical neighbors.*

## How It All Adds Together

### Initialization:
The class is initialized with an input string, metric, and model name. The tokenizer and pre-trained model are set up for NLP tasks.

### Typographical Neighbor Retrieval:
Typographical neighbors are retrieved using the NLTK corpus. The input string is compared with each word in the corpus to find neighbors with a distance of 1.

### Encoding and Similarity Calculation:
Typographical neighbors are encoded using the pre-trained model. Cosine similarity is calculated between the input representation and encoded neighbors.

### Displaying Neighbors:
The most similar typographical neighbors are displayed, sorted by similarity.
