from FinBERT_6 import SentimentAnalysis6

# Define class for a sentiment analysis interface
class SentimentalTest:

    # Initialize the sentiment analysis model on class instantiation
    def __init__(self):
        self.sentiment_model = SentimentAnalysis6()

    # Method to get sentences from the user until 'x' is entered
    def get_user_sentences(self):
        sentences = []
        print("\nPlease enter your sentences for scoring. Press Enter twice to finish input:\n")
        while True:
            try:
                # Read lines until a blank line is entered
                sentence = input()
                if sentence.strip() == '':
                    break
                sentences.append(sentence)
                        # Convert list of sentences to single text string with each sentence on a new line

                text = "\n".join(sentences)
            except EOFError:
                # Break the loop if End-of-File (Ctrl-D) is received
                break
            
        return text


    # Method to print sentiment analysis score for given sentences
    def overall_sentiment_results(self, text):
        results = self.sentiment_model.get_sentiments(text)
        # to print the main results of the FinBERT
        print("This is the overall results to help you understand the datat we're working with:")
        print(results)
        return self.sentiment_model.sentiment_count(results)

    # Method to print sentiment analysis score for a given sentence
    def print_sentence_score(self, text):
        results = self.sentiment_model.get_sentiments(text)
        self.counter = 1
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
            self.overall_sentiment_results(sentences)

            print("\nHere's a rather sentimental analysis:")
            self.print_sentence_score(sentences)

            # Ask the user to continue or not
            while True:
                check = input("\nDo you want to run another paragraph check? (y/n): ").lower()
                if check in ['y', 'n']:
                    break
                print("Please enter a valid response ('y' or 'n').")
            # Break the loop if the user chooses not to continue
            if check == 'n':
                print("\nAlways happy to help!")
                break
            
sentiment_interface = SentimentalTest()
sentiment_interface.run()