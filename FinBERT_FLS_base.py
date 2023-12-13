from transformers import AutoTokenizer, BertTokenizer, BertForSequenceClassification, pipeline
import os
import nltk
import collections

class FLSModel:
    def __init__(self, model_name = 'yiyanghkust/finbert-fls', num_labels=3):
        self.fls = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.nlp = pipeline("text-classification", model=self.fls, tokenizer=self.tokenizer)    
         
    def tokenize(self, text):
        # Split text into sentences using the NLTK tokenizer
        text = nltk.sent_tokenize(text)
        return text

    def get_sentimentsFLS(self, text):
        if not text:
            # If text is empty or None, raise a ValueError
            raise ValueError("Input text cannot be empty or None.")

        label_mapping = {
            'Not FLS': 'Not Forward Looking',
            'Specific FLS': 'Specific Forward Looking',
            'Non-specific FLS': 'Non-specific Forward Looking'
        }
        try:
            # using tokenize function to convert text into sentences for finbert-tone model
                sentences = self.tokenize(text)
                results = self.nlp(sentences)
                
                # Combine each sentence with their sentiment results
                for sentence, result in zip(sentences, results):
                    result['sentence'] = sentence
                    
                    # Replace 'label' with 'labelFLS' and map to new label
                    # Check if 'label' exists in the result
                    if 'label' in result:
                        original_label = result.pop('label')
                        result['LabelFLS'] = label_mapping.get(original_label, original_label)

                    # Replace 'score' with 'ScoreFLS'  
                    result['ScoreFLS'] = result.pop('score') 
                
                return results
        
        # Output syntax is a list of dictionaries:
        # [{'label': 'Specific FLS', 'score': 0.9999798536300659, 'sentence': 'We expect the age of our fleet to enhance availability and reliability due to reduced downtime for repairs.'}]
        except Exception as e:
            print("An error occurred during model inference: ", str(e))
            return None
    
    def sentiment_countFLS(self, results):
        if results is None:
            print("No results to process.")
            return
        labels = [result['LabelFLS'] for result in results]
        counts = collections.Counter(labels)
        total = len(labels)
        for label, count in counts.items():
            if count > 0:
                print(f"{label:<30}: {count} {round((count / total) * 100, 2)} %")
    
        
# Example usage
if __name__ == "__main__":
    sentiment_analyzer = FLSModel()
    while True:
        input_text = input("Enter your text: ")
        results = sentiment_analyzer.get_sentimentsFLS(input_text)
        print("Results:", results)
        sentiment_analyzer.sentiment_countFLS(results)
        choice = input("Continue? (y/n): ")
        if choice.lower() != 'y':
            break
# We have successfully optimized our operations. We now expect the age of our fleet to enhance availability and reliability due to reduced downtime for repairs.
