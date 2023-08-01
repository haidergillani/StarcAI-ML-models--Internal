from FinBERT_Model import SentimentAnalysis

import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class SentimentAnalysis3(SentimentAnalysis):

    #Normal on a 3 scale without probabilities
    def get_sentiments(self, text):
        if not text:
            # If text is empty or None, raise a ValueError
            raise ValueError("Input text cannot be empty or None.")

        try:
            #LABEL_0: neutral; LABEL_1: positive; LABEL_2: negative
            # using tokenize function to convert text into sentences for finbert-tone model
            sentences = self.tokenize(text)
            results = self.nlp(sentences)
            # Combine each sentence with their sentiment results
            for sentence, result in zip(sentences, results):
                result['sentence'] = sentence
            return results
        # Output syntax is a list of dictionaries:
        # [{'label': 'Neutral', 'score': 0.9999798536300659, 'sentence': 'Apple often gets its parts from multiple sources.'}]
        except Exception as e:
            print("An error occurred during model inference: ", str(e))
            return None
    
    # for a 3 point label scale    
    def sentiment_count(self, results):
        # finbert-tone uses LABEL_1 for positive, LABEL_0 for neutral, and LABEL_2 for negative
        labels = [result['label'] for result in results]
        counts = collections.Counter(labels)
        print("Positives: ", counts['Positive'], round( counts['Positive']/len(labels),2)*100, "%")
        
        print("Negatives: ", counts['Negative'], round(counts['Negative']/len(labels),2)*100, "%")
        
        print("Neutrals: ", counts['Neutral'], round(counts['Neutral']/len(labels), 2)*100,"%")
        
        #Thanks mate
        #I think we can use this to get the sentiment of the whole article
        #and then we can use the sentiment to predict the stock price   
        
    # for a 3 point label
    def sentiment_conf_scores(self, results):
        scores = [result['score'] for result in results]
        print(scores)
        
    # for a 3 point label
    def sentiment_df(self, results):
        # create a dataframe with columns: sentence, label, score
        data = {'sentence': [], 'label': [], 'score': []}
        for result in results:
            data['sentence'].append(result['sentence'])
            data['label'].append(result['label'])
            data['score'].append(result['score'])

        df = pd.DataFrame(data)
        return df
    
    def sentiment_plots(self, results):
        # finbert-tone uses LABEL_1 for positive, LABEL_0 for neutral, and LABEL_2 for negative
        labels = [result['label'] for result in results]
        counts = collections.Counter(labels)
        
        #Plot sentiment
        plt.figure(figsize=(8, 6))
        plt.bar(counts.keys(), counts.values(), color=['#1F77B4', '#2CA02C', '#FF7F0E'])
        plt.xlabel("Sentiment Label")
        plt.ylabel("Count")
        plt.title("Sentiment of Text")
        plt.show()
        
        #Plot sentiment in percentage
        plt.figure(figsize=(8, 6))
        plt.bar(counts.keys(), (np.array(list(counts.values()))/len(labels)) * 100, color=['#1F77B4', '#2CA02C', '#FF7F0E'])
        plt.xlabel("Sentiment Label")
        plt.ylabel("Percentage (%)")
        plt.title("Sentiment of Text")
        plt.show()