# Batch Processing

from torch.nn.functional import softmax

from FinBERT_Model import SentimentAnalysis

import torch
import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
    FinBERT: Financial Sentiment Analysis with BERT Fine-Tuned
    
    This is the model for 6-scale sentiment analysis of financial news articles using probabilities.
"""

#Code Below for 6 point scale with probabilities
#Note: All functions in this docstring are specific to the 5 point scale
    
class BatchSentimentAnalysis6(SentimentAnalysis):

#6 point scale with probabilities        
    def get_sentiments(self, sentences, batch_size=16):
        if not sentences:
            return []
        try:

            results = []
            input_ids = []
            attention_masks = []

            # Encode each sentence in the batch
            for sentence in sentences:
                encoded_dict = self.tokenizer.encode_plus(
                    sentence,
                    add_special_tokens=True,
                    max_length=512,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                input_ids.append(encoded_dict['input_ids'])
                attention_masks.append(encoded_dict['attention_mask'])

            # Concatenate the lists into PyTorch tensors
            input_ids = torch.cat(input_ids, dim=0)
            attention_masks = torch.cat(attention_masks, dim=0)

            # Get the model outputs with the forward method, without performing backpropagation
            with torch.no_grad():
                outputs = self.finbert(input_ids, attention_mask=attention_masks)

            # Apply softmax function to get probabilities from model's raw outputs
            probs = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()

            for sentence, prob in zip(sentences, probs):
                result = {
                    'sentence': sentence,
                    'Positive': prob[1],
                    'Neutral': prob[0],
                    'Negative': prob[2]
                }
                result['label'] = self.generate_label(result)

                results.append(result)

            return results

        except Exception as e:
            print("An error occurred during model inference: ", str(e))
            return []

        # Output syntax is a list of dictionaries:
        #[{'sentence': 'Apple often gets its parts from multiple sources.', 'Positive': 8.310216e-08, 'Neutral': 0.99997985, 'Negative': 2.004896e-05, 'label': 'Strong Neutral'}]
   
    #6 point scale with probabilities        
    def generate_label(self, result):
        # Given the sentiment scores, generate a label based on the defined thresholds
        if result['Positive'] > 0.9:
            return 'Strong Positive'
        elif result['Positive'] > 0.5:
            return 'Positive'
        elif result['Negative'] > 0.9:
            return 'Strong Negative'
        elif result['Negative'] > 0.5:
            return 'Negative'
        elif result['Neutral'] > 0.9:
            return 'Strong Neutral'
        elif result['Neutral'] > 0.5:
            return 'Neutral'
        else:
            return 'Undefined'  # Default label in case none of the conditions above are met
    
    def batched_inference(self, text, batch_size=16):
        # Tokenize the input text into sentences
        sentences = self.tokenize(text)

        results = []
        # Iterate over the sentences in batches
        for i in range(0, len(sentences), batch_size):
            # Get the current batch of sentences
            batch_sentences = sentences[i: i + batch_size]
            # Get sentiments for the current batch
            batch_results = self.get_sentiments(batch_sentences)
            # Add the results to our list
            results.extend(batch_results)

        return results     
        
    #5 point scale with probabilities        
    def sentiment_count(self, results):
        # finbert-tone uses LABEL_1 for positive, LABEL_0 for neutral, and LABEL_2 for negative
        labels = [result['label'] for result in results]
        counts = collections.Counter(labels)
        print("Strong Positives: ", counts['Strong Positive'], round( counts['Strong Positive']/len(labels)*100, 2), "%") 
        print("Positives: ", counts['Positive'], round( counts['Positive']/len(labels)*100,2), "%")
        
        print("Strong Negatives: ", counts['Strong Negative'], round( counts['Strong Negative']/len(labels)*100,2), "%") 
        print("Negatives: ", counts['Negative'], round(counts['Negative']/len(labels)*100,2), "%")
        
        print("Strong Neutrals: ", counts['Strong Neutral'], round(counts['Strong Neutral']/len(labels)*100,2), "%")
        print("Neutrals: ", counts['Neutral'], round(counts['Neutral']/len(labels)*100,2), "%")
        
        print("Undefined: ", counts['Undefined'], round(counts['Undefined']/len(labels)*100,2), "%")
        #Thanks mate
        #I think we can use this to get the sentiment of the whole article
        #and then we can use the sentiment to predict the stock price
    
    
    def sentiment_df(self, results):
        # create a dataframe with columns: sentence, label, score
        data = {'sentence': [], 'label': [], 'positive': [], 'negative': [], 'neutral': []}
        for result in results:
            data['sentence'].append(result['sentence'])
            data['label'].append(result['label'])
            data['positive'].append(result['Positive'])
            data['negative'].append(result['Negative'])
            data['neutral'].append(result['Neutral'])

        df = pd.DataFrame(data)
        return df
    
