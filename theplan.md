# Typographical Module

Input : Words, Phrase or Sentence

Responsibility : Rishikesh, Akil

Options for chunking and complete tokenization.
Attention mechanisms to reduce compute time.
Attention threshold for controlling sensitivity of attention mechanism.
Use large corpus to determine typographical neighbors.

Output : Typographical Neighbors

# Semantic Module

Input : Words, Phrases or Sentences with typographical neighbors substituted

Responsibility : Guru, Adhit

Find semantic neighbors using vector search.
Chunk the sentence, then get the vector sum path as a metric.
Weighted path sum (attention).

Output: Semantic Neighbors

# Fuzzy Search Module

Input : Words, Phrases or Sentences

Responsibility : Guru, Rishikesh

Already implemented.

# Attention Module

Input:Sentances

Responsibility:Rishikesh

Output: Top N important words, weighted vector sum(Attentions of each word and Vector embeddings of each word)