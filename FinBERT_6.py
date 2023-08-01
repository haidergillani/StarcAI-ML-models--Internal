
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
    
class SentimentAnalysis6(SentimentAnalysis):

    #6 point scale with probabilities        
    def get_sentiments(self, text):
        if not text:
            # If text is empty or None, raise a ValueError
            raise ValueError("Input text cannot be empty or None.")

        try:
            # Tokenize the text into sentences
            sentences = self.tokenize(text)

            # Use the tokenizer to encode all the sentences at once
            encoded_dict = self.tokenizer.batch_encode_plus(
                sentences,
                add_special_tokens=True,
                max_length=512,     # Specifies max length
                padding='max_length',   # Pads to max_length
                truncation=True,    # Enables truncation
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            # Get the encoded inputs and attention masks from the dictionary
            input_ids = encoded_dict['input_ids']
            attention_masks = encoded_dict['attention_mask']

            # Get the model outputs with the forward method, without performing backpropagation
            with torch.no_grad():
                outputs = self.finbert(input_ids, attention_mask=attention_masks)

            # Apply softmax function to get probabilities from model's raw outputs
            probs = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()

            # Prepare results
            results = []
            # Loop over each sentence and its corresponding probabilities
            for sentence, prob in zip(sentences, probs):
                result = {}
                # Save the sentence
                result['sentence'] = sentence
                # Save the probabilities for each label
            # NOTE: for the ProsusAI model, the labels are in the following order:
            # LABEL_0: positive; LABEL_1: negative; LABEL_2: neutral 
                result['Positive'] = prob[1]  # positive is LABEL_1
                result['Neutral'] = prob[0]  # neutral is LABEL_0
                result['Negative'] = prob[2]  # negative is LABEL_2
                # Append the result to our results list
                result['label'] = self.generate_label(result)  # Call the new function here

                results.append(result)

            # Return the results
            return results

        except Exception as e:
            print("An error occurred during model inference: ", str(e))
            return None

   
    #6 point scale with probabilities        
    def generate_label(self, result):
        # Given the sentiment scores, generate a label based on the defined thresholds
        if result['Positive'] > 0.99:
            return 'Strong Positive'
        elif result['Positive'] > 0.5:
            return 'Positive'
        elif result['Negative'] > 0.99:
            return 'Strong Negative'
        elif result['Negative'] > 0.5:
            return 'Negative'
        elif result['Neutral'] > 0.99:
            return 'Strong Neutral'
        elif result['Neutral'] > 0.5:
            return 'Neutral'
        else:
            return 'Undefined'  # Default label in case none of the conditions above are met
        
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
        
        
    def sentiment_prob_scores(self, results):
        positives = [result['Positive'] for result in results]
        negatives = [result['Negative'] for result in results]
        neutrals = [result['Neutral'] for result in results]
        scores = [positives, negatives, neutrals]
        print(scores)
    
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
    
    def sentiment_plots(self, results):
        # finbert-tone uses LABEL_1 for positive, LABEL_0 for neutral, and LABEL_2 for negative
        labels = [result['label'] for result in results]
        counts = collections.Counter(labels)
        
        #Plot sentiment
        plt.figure(figsize=(8, 6))
        plt.bar(counts.keys(), counts.values(), color=['#1F77B4', '#2CA02C', '#FF7F0E','#b02151','#735199', '#4f4912', '#454542'])
        plt.xlabel("Sentiment Label")
        plt.ylabel("Count")
        plt.title("Sentiment of Text")
        plt.show()
        
        #Plot sentiment in percentage
        plt.figure(figsize=(8, 6))
        plt.bar(counts.keys(), (np.array(list(counts.values()))/len(labels)) * 100, color=['#1F77B4', '#2CA02C', '#FF7F0E','#b02151','#735199', '#4f4912', '#454542'])
        plt.xlabel("Sentiment Label")
        plt.ylabel("Percentage (%)")
        plt.title("Sentiment of Text")
        plt.show()