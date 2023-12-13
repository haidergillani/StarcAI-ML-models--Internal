import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, BertTokenizer, BertForSequenceClassification, pipeline
#from keras.preprocessing.sequence import pad_sequences
#from torch.nn.functional import softmax
import collections

import os
import nltk

# Setting the NLTK data path to the local 'nltk_data' directory
nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))

"""
    FinBERT: Financial Sentiment Analysis with BERT Fine-Tuned
    
    This is the model for 6-scale sentiment analysis of financial news articles using probabilities.
"""

#Code Below for 6 point scale with probabilities
#Note: All functions in this docstring are specific to the 5 point scale

class ToneModel():
    def __init__(self, model_name='yiyanghkust/finbert-tone', num_labels=3):
        # 'yiyanghkust/finbert-fls' for forward-looking statements (finetuned on 3500 sentences)
        # 'ProsusAI/finbert' for original FinBERT model
        # 'yiyanghkust/finbert-tone' for finbert-tone (finetuned on 10,000 sentences)
        
        self.finbert = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.nlp = pipeline("sentiment-analysis", model=self.finbert, tokenizer=self.tokenizer)
        
    # p = positive, n = negative, nu = neutral
    LABELS = [
        ('Strong Positive', lambda p, n, nu: p > 0.90),
        ('Positive',        lambda p, n, nu: p > 0.70),
        ('Slightly Positive', lambda p, n, nu: p > 0.50),
        ('Strong Negative', lambda p, n, nu: n > 0.90),
        ('Negative', lambda p, n, nu: n > 0.70),
        ('Slightly Negative', lambda p, n, nu: n > 0.50),
        ('Neutral yet Slightly Positive', lambda p, n, nu: nu > p and nu > n and p > n and p > 0.10),
        ('Neutral yet Slightly Negative', lambda p, n, nu: nu > p and nu > n and n > p and n > 0.10),
        ('Strong Neutral', lambda p, n, nu: nu > 0.90),
        ('Neutral', lambda p, n, nu: nu > 0.70),
        ('Slightly Neutral', lambda p, n, nu: nu > 0.50 or nu < 0.50 and p < 0.5 and n < 0.5),
    ]

    def tokenize(self, text):
        # text is a list of sentences
        # tokenize the text into sentences for finbert-tone model
        text = nltk.tokenize.sent_tokenize(text)
        return text
    
    def get_sentiments(self, text):
        sentences = self.tokenize(text)
        if not sentences:
            # If text is empty or None, raise a ValueError
            raise ValueError("Sentences is not working text cannot be empty or None.")

        try:
            # Tokenize the text into sentences
            #sentences = self.tokenize(text)

            # Preprocessing a batch of sentences for input into transformer-based FinBERT 
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

            # Get the model outputs with the forward method, without performing backpropagation
            # this disables gradient descent and saves memory since model is not being trained
            with torch.no_grad():
                outputs = self.finbert(**encoded_dict)

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
            # Note: for the ProsusAI model, the labels are in the following order:
            # LABEL_0: positive; LABEL_1: negative; LABEL_2: neutral 
            
                result['Positive'] = prob[1]  # positive is LABEL_1
                result['Neutral'] = prob[0]  # neutral is LABEL_0
                result['Negative'] = prob[2]  # negative is LABEL_2
                # Append the result to our results list
                result['label'] = self.generate_label(result)  # Call the new function here
                result['LabelTone'] = result.pop('label')  # Replace 'label' with 'labelFLS'

                results.append(result)

            # Return the results
            return results

        except Exception as e:
            raise ValueError("Error processing input text. Please ensure it is a valid string.") from e

   
    # label generation with probabilities        
    def generate_label(self, result):
        
        positive = result['Positive']
        negative = result['Negative']
        neutral = result['Neutral']
        
        for label, condition in self.LABELS:
            if condition(positive, negative, neutral):
                return label
        return 'Undefined'  # Default label in case none of the conditions above are met
        
    #5 point scale with probabilities        
    def sentiment_count(self, results):
        if results is None:
            print("No results to process.")
            return
        
        # finbert-tone uses LABEL_1 for positive, LABEL_0 for neutral, and LABEL_2 for negative
        labels = [result['LabelTone'] for result in results]
        counts = collections.Counter(labels)
        total_labels = len(labels)

        sentiments = [sentiment for sentiment, _ in self.LABELS]
        for sentiment in sentiments:
            if counts.get(sentiment, 0) > 0:
                # Using list comprehension to print results
                print(f"{sentiment:<50}: {counts[sentiment]} {round(counts[sentiment] / total_labels * 100, 2)} %")
            
    def sentiment_prob_scores(self, results):
        if results is None:
            print("No results to process.")
            return

        # Extracting individual probabilities for each sentiment
        positives = [result['Positive'] for result in results]
        negatives = [result['Negative'] for result in results]
        neutrals = [result['Neutral'] for result in results]

        # Calculating the sum of probabilities for each sentiment
        sum_positives = sum(positives)
        sum_negatives = sum(negatives)
        sum_neutrals = sum(neutrals)

        # Return these sums for probabiity scores
        return sum_positives, sum_negatives, sum_neutrals
'''
    def sentiment_df(self, results):
        # create a dataframe with columns: sentence, label, score
        data = {'sentence': [], 'LabelTone': [], 'positive': [], 'negative': [], 'neutral': []}
        for result in results:
            data['sentence'].append(result['sentence'])
            data['LabelTone'].append(result['LabelTone'])
            data['positive'].append(result['Positive'])
            data['negative'].append(result['Negative'])
            data['neutral'].append(result['Neutral'])

        df = pd.DataFrame(data)
        return df
    
    def sentiment_plots(self, results):
        # finbert-tone uses LABEL_1 for positive, LABEL_0 for neutral, and LABEL_2 for negative
        labels = [result['LabelTone'] for result in results]
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
   
'''
 
# Example usage
if __name__ == "__main__":
    sentiment_analyzer = ToneModel()
    while True:
        input_text = input("Enter your text: ")
        results = sentiment_analyzer.get_sentiments(input_text)
        print("Results: ", results)
        sentiment_analyzer.sentiment_count(results)
        choice = input("Continue? (y/n): ")
        if choice.lower() != 'y':
            break    