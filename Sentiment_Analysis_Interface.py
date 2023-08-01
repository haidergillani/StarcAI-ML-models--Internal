# Import the 6 scale model from FinBERT_6.py
from FinBERT_6 import SentimentAnalysis6

# Define class for a sentiment analysis interface
class SentimentAnalysisInterface:

    # Initialize the sentiment analysis model on class instantiation
    def __init__(self):
        self.sentiment_model = SentimentAnalysis6()

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

    # Method to print sentiment analysis score for a given sentence
    def print_sentence_score(self, sentence):
        results = self.sentiment_model.get_sentiments(sentence)
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
        while True:
            sentences = self.get_user_sentences()
            print("\nYour sentiment scores:")
            self.counter = 1
            for sentence in sentences:
                self.print_sentence_score(sentence)

            # Ask the user to continue or not
            while True:
                check = input("\nDo you want to run another check? (y/n): ").lower()
                if check in ['y', 'n']:
                    break
                print("Please enter a valid response ('y' or 'n').")
            # Break the loop if the user chooses not to continue
            if check == 'n':
                print("\nAlways happy to help. Goodbye!")
                break
                
# Main guard to run the interface only when the script is run directly
if __name__ == "__main__":
    sentiment_interface = SentimentAnalysisInterface()
    sentiment_interface.run()