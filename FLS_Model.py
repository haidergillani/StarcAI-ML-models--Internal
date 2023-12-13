from transformers import AutoTokenizer, BertTokenizer, BertForSequenceClassification, pipeline
import torch
import os
import nltk
import collections
from torch.nn.functional import softmax

class FLSModel:
    def __init__(self, model_name = 'yiyanghkust/finbert-fls', num_labels=3):
        self.fls = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.nlp = pipeline("text-classification", model=self.fls, tokenizer=self.tokenizer)    
    
    LABELS = [
        ('Highly Specific Forward-Looking', lambda s, n, no: s > 0.90),
        ('Specific Forward-Looking',        lambda s, n, no: s > 0.70),
        ('Slightly Specific Forward-Looking', lambda s, n, no: s > 0.50),
        ('Highly Non-Specific Forward-Looking', lambda s, n, no: n > 0.90),
        ('Non-Specific Forward-Looking', lambda s, n, no: n > 0.70),
        ('Slightly Non-Specific Forward-Looking', lambda s, n, no: n > 0.50),
        ('Not Forward-Looking yet Slightly Specific', lambda s, n, no: no > s and no > n and s > n and s > 0.10),
        ('Not Forward-Looking yet Slightly Non-Specific', lambda s, n, no: no > s and no > n and n > s and n > 0.10),
        ('Certainly Not Forward-Looking', lambda s, n, no: no > 0.90),
        ('Not Forward-Looking', lambda s, n, no: no > 0.70),
        ('Slightly Not Forward-Looking', lambda s, n, no: no > 0.50 or no < 0.50 and s < 0.5 and n < 0.5),
    ]
         
    def tokenize(self, text):
        # Split text into sentences using the NLTK tokenizer
        text = nltk.sent_tokenize(text)
        return text

    def get_sentimentsFLS(self, text):
        
        # If text is empty or None, raise a ValueError
        if not text:
            raise ValueError("Input text cannot be empty or None.")
        
        try:
            # using tokenize function to convert text into sentences for finbert-tone model
            sentences = self.tokenize(text)
            
            # Preprocessing a batch of sentences for input into transformer-based FinBERT 
            # Use the tokenizer to encode all the sentences at once
            encoded_batch = self.tokenizer.batch_encode_plus(
                sentences,
                add_special_tokens=True,
                max_length=512,     # Specifies max length
                padding='max_length',   # Pads to max_length
                truncation=True,    # Enables truncation
                return_attention_mask=True,
                return_tensors='pt'
            )

            # Perform model inference
            
            # Get the model outputs with the forward method, without performing backpropagation
            # this disables gradient descent and saves memory since model is not being trained
            with torch.no_grad():
                outputs = self.fls(**encoded_batch)
                logits = outputs.logits
                probabilities = softmax(logits, dim=1).detach().numpy()

            results = []
            for i, sentence in enumerate(sentences):
                # Extract probabilities for each sentence
                prob = probabilities[i]
                
                # LABEL_0: Not FLS; LABEL_1: Non-specific FLS; LABEL_2: Specific FLS 
                result = {
                    'sentence': sentence,
                    'Not FLS': prob[0],
                    'Non-specific FLS': prob[1],
                    'Specific FLS': prob[2]
                }
                result['LabelFLS'] = self.generate_label(result)
                results.append(result)

            return results
        
        except Exception as e:
            raise ValueError("No valid text to process.")
        
    def generate_label(self, result):
        
        Specific = result['Specific FLS']
        nonSpecific = result['Non-specific FLS']
        notFLS = result['Not FLS']
        
        for label, condition in self.LABELS:
            if condition(Specific, nonSpecific, notFLS):
                return label
        return 'Undefined'  # Default label in case none of the conditions above are met
    
    def sentiment_countFLS(self, results):
        if results is None:
            print("No results to process.")
            return
        
        labels = [result['LabelFLS'] for result in results]
        counts = collections.Counter(labels)
        total_labels = len(labels)

        sentimentsFLS = [sentiment for sentiment, _ in self.LABELS]
        print("\n") # add line break at start to separate from previous output of Tone model
        for sentiment in sentimentsFLS:
            if counts.get(sentiment, 0) > 0:
                # Using list comprehension to print results
                print(f"{sentiment:<50}: {counts[sentiment]} {round(counts[sentiment] / total_labels * 100, 2)} %")

            
# Example usage
if __name__ == "__main__":
    sentiment_analyzer = FLSModel()
    while True:
        input_text = input("Enter your text: ")
        results = sentiment_analyzer.get_sentimentsFLS(input_text)
        print("Results: ", results)
        sentiment_analyzer.sentiment_countFLS(results)
        choice = input("Continue? (y/n): ")
        if choice.lower() != 'y':
            break
# We have successfully optimized our operations. We now expect the age of our fleet to enhance availability and reliability due to reduced downtime for repairs.
