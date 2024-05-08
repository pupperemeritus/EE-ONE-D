# About the module:

- This module's has been purposefully built to return the attention weights, which are a numerical representation of the given input. The module is created for integration with `\typographical_search.py`.
- The goal was to utilize a transformer's encoder block to obtain a word's **attention weights**, a key concept used by them to gain a deeper contextual understanding and understand the role that each word is playing in a sentence.

- ## Details:

  ```
  - **Input type**: `a string of words/ a single word`
  ```

  - **Output type**: `A list of numbers (words)/ a single number, representing the attention weights.`
  - **Transformer used**: `bert-base-uncased`
  - **Requirements**: `nltk, torch, transformers(huggingface)`

- The module has the following functions:

  - `__init__(self, model_name = 'bert-base-uncased)` initialize the transformer's model and tokenizer, `WoroNetLemmatizer` and `word_tokenize()`

    - `preprocess_input(self)` Remove stop words and lemmatize the sentence.
    - `forward(self)` To send the tokens to the model obtain the attention weights. These weights are then averaged to reduce the dimensions. We have used **Bert** `BertModel.from_pretrained('bert-base-uncased')`given to it's ease of use and availability.
    - `get_weights(self)`: Depending on the number of words in the input sequence, the number of "neighbors" can be adjusted
    - `get_n_words(self)` Uses all the above functions and generates the output.

## How to use this module

- If you wish to play around with this module, then run the following code: `

  ```
  model = AttentionModel(input_string="A cat sat on the mat")
        top_tokens = model.get_n_words()
        print("Top tokens:", top_tokens)
  ```

  ## What are we planning to add?

- We aim to add more ways to eliminate edge cases that can appear in the inputs.
- Adding a parse tree generator.
- We are also planning bigger language models other than Bert.
