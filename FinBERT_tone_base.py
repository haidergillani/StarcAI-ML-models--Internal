from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import collections
import pandas as pd
from utils import FastFinancialTokenizer

class SentimentAnalysis:
    def __init__(self, model_name='yiyanghkust/finbert-tone', num_labels=3):
        # 'yiyanghkust/finbert-fls' for forward-looking statements (finetuned on 3500 sentences)
        # 'ProsusAI/finbert' for original FinBERT model
        # 'yiyanghkust/finbert-tone' for finbert-tone (finetuned on 10,000 sentences)
        
        self.finbert = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.nlp = pipeline("sentiment-analysis", model=self.finbert, tokenizer=self.tokenizer)
    
    def tokenize(self, text):
        # text is a list of sentences
        # tokenize the text into sentences for finbert-tone model
        text = FastFinancialTokenizer().tokenize(text)
        return text
        
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
    
    def sentiment_labels(self, results):
        labels = [result['label'] for result in results]
        print(labels)
        
    # for a 3 point label scale    
    def sentiment_count(self, results):
        # finbert-tone uses LABEL_1 for positive, LABEL_0 for neutral, and LABEL_2 for negative
        labels = [result['label'] for result in results]
        counts = collections.Counter(labels)
        print("Positives: ", counts['Positive'], round( counts['Positive']/len(labels),2)*100, "%")
        
        print("Negatives: ", counts['Negative'], round(counts['Negative']/len(labels),2)*100, "%")
        
        print("Neutrals: ", counts['Neutral'], round(counts['Neutral']/len(labels), 2)*100,"%")
        
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