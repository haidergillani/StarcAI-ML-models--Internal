from transformers import BertTokenizer, BertForSequenceClassification
import torch
import collections
from utils import FastFinancialTokenizer

# Initialize model and device once at module level
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_tokenizer = None
_fls = None

def _initialize_model(model_name='yiyanghkust/finbert-fls', num_labels=3):
    global _tokenizer, _fls
    if _tokenizer is None:
        _tokenizer = BertTokenizer.from_pretrained(model_name)
        _fls = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(_device)
        _fls.eval()  # Set to evaluation mode

class FLSModel:
    def __init__(self, model_name = 'yiyanghkust/finbert-fls', num_labels=3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fls = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
    
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
        text = FastFinancialTokenizer().tokenize(text)
        return text

    def get_sentimentsFLS(self, text):
        if not text:
            raise ValueError("Input text cannot be empty or None.")
        
        try:
            sentences = self.tokenize(text)
            batch_size = 32
            results = []
            
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                encoded_batch = self.tokenizer.batch_encode_plus(
                    batch,
                    add_special_tokens=True,
                    max_length=512,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                
                encoded_batch = {k: v.to(_device) for k, v in encoded_batch.items()}
                
                with torch.no_grad():
                    outputs = self.fls(**encoded_batch)
                
                probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()
                
                batch_results = [
                    {
                        'sentence': sentence,
                        'Not FLS': float(prob[2]),
                        'Specific FLS': float(prob[0]),
                        'Non-specific FLS': float(prob[1]),
                        'LabelFLS': self.generate_label({
                            'Specific FLS': float(prob[0]),
                            'Non-specific FLS': float(prob[1]),
                            'Not FLS': float(prob[2])
                        })
                    }
                    for sentence, prob in zip(batch, probs)
                ]
                results.extend(batch_results)
                
            # Print debug only for first call
            if not hasattr(self, '_debug_printed'):
                print(f"Debug - Sample FLS result: {results[0]}")
                self._debug_printed = True
                
            return results
            
        except Exception as e:
            print(f"Error in FLS processing: {str(e)}")
            raise ValueError("Error processing input text.") from e
    
    def generate_label(self, result):
        s, n, no = result['Specific FLS'], result['Non-specific FLS'], result['Not FLS']
        
        if no > 0.90: return 'Certainly Not Forward-Looking'
        if no > 0.70: return 'Not Forward-Looking'
        if s > 0.90: return 'Highly Specific Forward-Looking'
        if s > 0.70: return 'Specific Forward-Looking'
        # ... etc
        return 'Undefined'
    
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
