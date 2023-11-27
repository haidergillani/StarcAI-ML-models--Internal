from transformers import AutoTokenizer, BertTokenizer, BertForSequenceClassification, pipeline
#from keras.preprocessing.sequence import pad_sequences
#from torch.nn.functional import softmax

import nltk
#import torch
import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        text = nltk.tokenize.sent_tokenize(text)
        return text
    
    def sentiment_labels(self, results):
        labels = [result['label'] for result in results]
        print(labels)
