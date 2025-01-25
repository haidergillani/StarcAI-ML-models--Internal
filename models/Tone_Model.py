import torch
from transformers import BertTokenizer, BertForSequenceClassification
import collections
from utils import FastFinancialTokenizer

# Initialize model and device once at module level
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_tokenizer = None
_finbert = None

def _initialize_model(model_name='yiyanghkust/finbert-tone', num_labels=3):
    global _tokenizer, _finbert
    if _tokenizer is None:
        _tokenizer = BertTokenizer.from_pretrained(model_name)
        _finbert = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(_device)
        _finbert.eval()  # Set to evaluation mode

class ToneModel:
    # Pre-define label conditions as class constants
    LABELS = [
        ('Strong Positive', lambda p, n, nu: p > 0.90),
        ('Positive', lambda p, n, nu: p > 0.70),
        ('Slightly Positive', lambda p, n, nu: p > 0.50),
        ('Strong Negative', lambda p, n, nu: n > 0.90),
        ('Negative', lambda p, n, nu: n > 0.70),
        ('Slightly Negative', lambda p, n, nu: n > 0.50),
        ('Neutral yet Slightly Positive', lambda p, n, nu: nu > p and nu > n and p > n and p > 0.10),
        ('Neutral yet Slightly Negative', lambda p, n, nu: nu > p and nu > n and n > p and n > 0.10),
        ('Strong Neutral', lambda p, n, nu: nu > 0.90),
        ('Neutral', lambda p, n, nu: nu > 0.70),
        ('Slightly Neutral', lambda p, n, nu: nu > 0.50 or (nu < 0.50 and p < 0.5 and n < 0.5)),
    ]

    def __init__(self, model_name='yiyanghkust/finbert-tone', num_labels=3):
        _initialize_model(model_name, num_labels)
        self.tokenizer = _tokenizer
        self.finbert = _finbert
    
    @staticmethod
    def tokenize(text):
        return FastFinancialTokenizer().tokenize(text)
    
    def get_sentiments(self, text):
        sentences = self.tokenize(text)
        if not sentences:
            raise ValueError("Sentences is not working text cannot be empty or None.")

        try:
            # Process sentences in batches for better performance
            batch_size = 32
            results = []
            
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                
                # Batch encode all sentences at once
                encoded_dict = self.tokenizer.batch_encode_plus(
                    batch,
                    add_special_tokens=True,
                    max_length=512,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                
                # Move tensors to device
                encoded_dict = {k: v.to(_device) for k, v in encoded_dict.items()}

                # Get model outputs
                with torch.no_grad():
                    outputs = self.finbert(**encoded_dict)
                
                # Get probabilities
                probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()

                # Process results
                batch_results = [
                    {
                        'sentence': sentence,
                        'Positive': float(prob[1]),
                        'Neutral': float(prob[0]),
                        'Negative': float(prob[2]),
                        'LabelTone': self.generate_label({
                            'Positive': float(prob[1]),
                            'Neutral': float(prob[0]),
                            'Negative': float(prob[2])
                        })
                    }
                    for sentence, prob in zip(batch, probs)
                ]
                results.extend(batch_results)

            return results

        except Exception as e:
            raise ValueError("Error processing input text. Please ensure it is a valid string.") from e

    @staticmethod
    def generate_label(result):
        positive = result['Positive']
        negative = result['Negative']
        neutral = result['Neutral']
        
        for label, condition in ToneModel.LABELS:
            if condition(positive, negative, neutral):
                return label
        return 'Undefined'

    def sentiment_count(self, results):
        if results is None:
            print("No results to process.")
            return
        
        labels = [result['LabelTone'] for result in results]
        counts = collections.Counter(labels)
        total_labels = len(labels)

        sentiments = [sentiment for sentiment, _ in self.LABELS]
        for sentiment in sentiments:
            if counts.get(sentiment, 0) > 0:
                print(f"{sentiment:<50}: {counts[sentiment]} {round(counts[sentiment] / total_labels * 100, 2)} %")
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