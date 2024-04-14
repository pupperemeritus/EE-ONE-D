import nltk
import torch
from transformers import BertModel, BertTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class AttentionModel(torch.nn.Module):
    def __init__(self, input_string, model_name="bert-base-uncased"):
        """
        Initializes the AttentionModel with the input_string and an optional model_name.

        Parameters:
            input_string (str): The input string for the model.
            model_name (str): The name of the pre-trained BERT model to use (default is "bert-base-uncased").

        Returns:
            None
        """
        super(AttentionModel, self).__init__()
        self.input_string = input_string.lower()
        self.model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.lemmatizer = WordNetLemmatizer()
        
        self.stop_words = set(stopwords.words('english'))
        self.tokens = word_tokenize(self.input_string)

        self.importance_scores = self.get_weights()
    
    def preprocess_input(self):
        filtered_words = [self.lemmatizer.lemmatize(word) for word in self.tokens if word not in self.stop_words]
        filtered_sentence = ' '.join(filtered_words)
        
        self.filtered_tokens = self.tokenizer.tokenize(filtered_sentence)
        token_ids = self.tokenizer.convert_tokens_to_ids(self.filtered_tokens)
        tokens_tensor = torch.tensor([token_ids])
        
        print("Original Sentence:", self.input_string)
        print("Filtered and Lemmatized Sentence:", filtered_sentence)
        print("Tokenized Sentence:", tokens_tensor)
        print(self.filtered_tokens)

        return tokens_tensor
    
    def forward(self):
        tokens_tensor = self.preprocess_input()
        with torch.no_grad():
            outputs = self.model(tokens_tensor)
        
        last_hidden_state = outputs.last_hidden_state
        attention_weights = torch.matmul(last_hidden_state, last_hidden_state.transpose(1, 2))
        average_attention_weights = attention_weights.mean(dim=0)
        importance_scores = average_attention_weights.sum(dim=0)
        
        return importance_scores

    def get_weights(self):
        outputs = self.forward()
        return outputs

    def get_n_words(self):
        top_n_indices = self.importance_scores.argsort(descending=True)[:10]
        
        top_n_tokens = [self.filtered_tokens[i] for i in top_n_indices if i < len(self.filtered_tokens)]
        
        return top_n_tokens

# Example usage:
model = AttentionModel(input_string="A cat sat on the mat")
top_tokens = model.get_n_words()
print("Top tokens:", top_tokens)
