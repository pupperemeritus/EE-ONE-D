## A Laymans Description

- **SemanticSearch Class:**
  - Imagine being given an extensive book, and you want to find sentences in it similar to your search query.
  - This class performs that. It reads the book (your document) and finds sentences that are similar to your query.
- **find_semantic_neighbors:**
  - When passed a query, this function looks through the sentences in the book to find the ones that resemble the query.
- **_get_embedding:**
  - This function turns a sentence into a computer understandable format.

## What are we doing?

### Packages Overview
- `logging` (provides logging functionality for debugging.)
- `torch`
- `nltk`
- `transformers`

### Input

- The `SemanticSearch` class takes several inputs:
  - `model`: The name of the model to be loaded.
  - `tokenizer`: The name of the tokenizer to be loaded.
  - `document`: A list of strings representing the document.
  - `input_text`: A flag indicating whether the input is text or not.
  - `max_seq_length`: The maximum sequence length.
 
**Note**: The model being used in the SemanticSearch class is *togethercomputer/m2-bert-80M-32k-retrieval*. This model is loaded from the Hugging Face model hub using the AutoModelForSequenceClassification class made clear through thefollowing code snippet:
`self.model = AutoModelForSequenceClassification.from_pretrained(
    model, trust_remote_code=True
).to(self.device)
`

### Output

- The `find_semantic_neighbors` method returns a list of tuples. Each tuple contains the text of a semantic neighbor and its similarity score.
- The `_get_embedding` method returns the embedding (a numerical representation) for the input text.


### A Brief Overview of the Functions Used

- **SemanticSearch initialization:**
  - Initializes the class with the provided model, tokenizer, and document.
  - Loads the model and tokenizer.
  - Splits the document into sentences if `input_text` is True.
- **find_semantic_neighbors:**
  - It computes the embedding for the input query using the _get_embedding method.
  - For each text in the document, it computes the embedding and calculates the cosine similarity between the query embedding and each document embedding.
  - The similarity score indicates how similar each document sentence is to the query.
  - It keeps track of the k most similar sentences.
- **_get_embedding:**
  - Tokenizes the input text.
  - Generates the embedding for the input text using the loaded model.

### Return Types of Functions

- **SemanticSearch initialization (`__init__` method):**
  - `None`

- **find_semantic_neighbors (`find_semantic_neighbors` method):**
  - Returns: `List[Tuple[str, float]]` - A list of tuples containing the text of the semantic neighbor and its similarity score.

- **_get_embedding (`_get_embedding` method):**
  - Returns: `torch.Tensor` - The embedding for the input text.
 
## How It Adds Together

### Initialization `(__init__ method)`:
Initializes the SemanticSearch object with the provided model, tokenizer, and document.

### Getting Semantic Neighbors `(find_semantic_neighbors method)`:
For a given query, it retrieves the semantic neighbors from the document by calculating the similarity between the query and each document sentence.
Returns a list of tuples containing the text of the semantic neighbor and its similarity score.

### Getting Embedding `(_get_embedding method)`:
Tokenizes the input text and generates the embedding using the loaded model.
Returns the embedding tensor for the input text.
