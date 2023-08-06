#Testing

from FinBERT_6 import SentimentAnalysis6

#from FinBERT_3 import SentimentAnalysis3
#from Batch_Processing_6 import BatchSentimentAnalysis6




# Define class for a web sentiment analysis interface
class Test:

    # Initialize the sentiment analysis model on class instantiation
    def __init__(self):
        self.sentiment_model = SentimentAnalysis6()

    # Method to get sentences from the user until 'x' is entered
    # Method to get sentences from the user until 'x' is entered
    def get_user_sentences(self):
        sentences = []
        print("\nPlease enter your sentences for scoring, ('end' to exit):\n")
        while True:
            sentence = input()
            if sentence.lower() == 'end':
                break
            elif sentence.isdigit() or len(sentence.strip()) <= 1:
                print("Please enter a valid sentence or 'end' to exit.")
                continue
            sentences.append(sentence)
        return sentences

        
    # Method to get sentiments from the FinBERT6 model for cleaned url texts
    def get_urls_sentiments(self, cleaned_texts):
        
        if not cleaned_texts:
            raise Exception("Unable to access the provided URL. Please provide a valid URL.")
        results = self.sentiment_model.get_sentiments(cleaned_texts)
        return results
    
    # Method to print sentiment analysis score for a given sentence
    def overall_sentiment_results(self, results):
        return self.sentiment_model.sentiment_count(results)

    # Method to print sentiment analysis score for a given sentence
    def print_sentence_score(self, results):
        
        for result in results:
            
            print(f"\n{self.counter}. Sentence: {result['sentence']}")
            print(f"Label: '{result['label']}' ")
            
            print("\nBreakdown:")
            # Loop through the sentiment types and print the score for each
            for sentiment in ['Positive', 'Neutral', 'Negative']:
                print(f"{sentiment} sentiment probability:  {round(float(result[sentiment]), 4)*100} %")
            
            # Increment to the counter for the next sentence.
            self.counter += 1

    # Main method to run the interface
    def run(self):
        cleaned_texts = self.get_user_sentences()
        print(cleaned_texts)
        
            
web_sentiment_interface = Test()
web_sentiment_interface.run()