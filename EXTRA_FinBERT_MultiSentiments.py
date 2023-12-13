
from torch.nn.functional import softmax

from FinBERT_Tone_base import SentimentAnalysis

import torch
import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# investor Paranoia
# investor Confidence
# Neutral
# Specifc FLS
# Non-specific FLS


"""
    FinBERT: Financial Sentiment Analysis with BERT Fine-Tuned
    
    This is the model for 6-scale sentiment analysis of financial news articles using probabilities.
"""

#Code Below for 6 point scale with probabilities
#Note: All functions in this docstring are specific to the 5 point scale
    
class SentimentAnalysis6(SentimentAnalysis):
    LABELS = [
        ('Highly Confident and Optimistic', lambda confidence, optimism, pessimism, uncertainty: confidence > 0.90 and optimism > 0.90),
        ('Confident and Optimistic',        lambda confidence, optimism, pessimism, uncertainty: confidence > 0.70 and optimism > 0.70),
        ('Slightly Confident and Optimistic', lambda confidence, optimism, pessimism, uncertainty: confidence > 0.50 and optimism > 0.50),
        ('Highly Uncertain and Pessimistic', lambda confidence, optimism, pessimism, uncertainty: uncertainty > 0.90 and pessimism > 0.90),
        ('Uncertain and Pessimistic',       lambda confidence, optimism, pessimism, uncertainty: uncertainty > 0.70 and pessimism > 0.70),
        ('Slightly Uncertain and Pessimistic', lambda confidence, optimism, pessimism, uncertainty: uncertainty > 0.50 and pessimism > 0.50),
        ('Highly Confident',                lambda confidence, optimism, pessimism, uncertainty: confidence > 0.90),
        ('Confident',                       lambda confidence, optimism, pessimism, uncertainty: confidence > 0.70),
        ('Slightly Confident',              lambda confidence, optimism, pessimism, uncertainty: confidence > 0.50),
        ('Highly Uncertain',                lambda confidence, optimism, pessimism, uncertainty: uncertainty > 0.90),
        ('Uncertain',                       lambda confidence, optimism, pessimism, uncertainty: uncertainty > 0.70),
        ('Slightly Uncertain',              lambda confidence, optimism, pessimism, uncertainty: uncertainty > 0.50),
        ('Highly Optimistic',               lambda confidence, optimism, pessimism, uncertainty: optimism > 0.90),
        ('Optimistic',                      lambda confidence, optimism, pessimism, uncertainty: optimism > 0.70),
        ('Slightly Optimistic',             lambda confidence, optimism, pessimism, uncertainty: optimism > 0.50),
        ('Highly Pessimistic',              lambda confidence, optimism, pessimism, uncertainty: pessimism > 0.90),
        ('Pessimistic',                     lambda confidence, optimism, pessimism, uncertainty: pessimism > 0.70),
        ('Slightly Pessimistic',            lambda confidence, optimism, pessimism, uncertainty: pessimism > 0.50),
        ('Undefined',                       lambda confidence, optimism, pessimism, uncertainty: True) # Default label
    ]

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
        optimism = result['Positive'] - result['Negative'] + (0.15 * result['Neutral'])
        pessimism = result['Negative'] - result['Positive'] - (0.15 * result['Neutral'])
        confidence = result['Positive'] + 0.5 * (1 - result['Negative']) - (0.1 * result['Neutral'])
        uncertainty = 1 - confidence
        
        for label, condition in self.LABELS:
            if condition(confidence, optimism, pessimism, uncertainty):
                return label
            else:
                return 'Undefined'  # Default label in case none of the conditions above are met
        
    #5 point scale with probabilities        
    def sentiment_count(self, results):
        # finbert-tone uses LABEL_1 for positive, LABEL_0 for neutral, and LABEL_2 for negative
        labels = [result['label'] for result in results]
        counts = collections.Counter(labels)
        total_labels = len(labels)

        print("\nSentiment Analysis Results: ")
        print(f"\nTotal Sentences: {total_labels}\n")

        sentiments = [sentiment for sentiment, _ in self.LABELS]

        # Using list comprehension to print results
        [print(f"{sentiment:<18}: {counts.get(sentiment, 0)} {round(counts.get(sentiment, 0) / total_labels * 100, 2)} %") for sentiment in sentiments if counts.get(sentiment, 0) > 0]

            
        
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
        
"""        
#Functions Sample:
#For a dataframe of the results
df = AppleNET.sentiment_df(results)
print(df) 
#export dataframe as csv
df.to_csv("file_name.csv", index=False)

Extra:
AppleNET.sentiment_labels(results)
AppleNET.sentiment_conf_scores(results)     
AppleNET.sentiment_plots(results)
"""