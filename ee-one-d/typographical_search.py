import nltk
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
typographical_module = nltk.edit_distance

model_name = "bert-base-uncased"  # BERT base model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

input_string = "sentient"

typographical_neighbors = []
for neighbor in nltk.corpus.words.words():
    if typographical_module(neighbor, input_string) == 1:  # Check if edit distance is 1
        typographical_neighbors.append(neighbor)

encoded_neighbors = []
for neighbor in typographical_neighbors:
    tokenized_neighbor = tokenizer(neighbor, return_tensors="pt")
    with torch.no_grad():
        output = model(**tokenized_neighbor)
    encoded_neighbors.append(output.last_hidden_state.mean(dim=1))

input_tokenized = tokenizer(input_string, return_tensors="pt")
input_representation = model(**input_tokenized).last_hidden_state.mean(dim=1)

similarities = []
for neighbor_embedding in encoded_neighbors:
    similarity_score = np.dot(input_representation.detach().numpy(), neighbor_embedding.T.detach().numpy()).item()
    similarities.append(similarity_score)

sorted_neighbors = [x for _, x in sorted(zip(similarities, typographical_neighbors), reverse=True)]

typographical_neighbors_list = [neighbor for neighbor in sorted_neighbors[:10]]

print("Input string:", input_string)
print("Typographical Neighbors:")
for i, neighbor in enumerate(typographical_neighbors_list, 1):
    print(f"{i}. {neighbor}")
