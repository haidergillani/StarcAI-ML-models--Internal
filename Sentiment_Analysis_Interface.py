# Import the tone scale model from ToneModel.py

from Tone_Model import ToneModel
from FLS_Model import FLSModel
from Sentiment_FLS_join import DataMerger
        
# Define class for a sentiment analysis interface
class SentimentAnalysisInterface(DataMerger):

    # Initialize the sentiment analysis model on class instantiation
    def __init__(self):
        self.tone_model = None
        self.FLS_model = None
        
    # Define weights for each label
    weights = {
        # Sentiment weights
        'Positive': 1.0,    # Most desirable sentiment
        'Neutral': 0.7,     # Neutral sentiment
        'Negative': 0.0,    # Least desirable sentiment

        # FLS weights
        'Specific FLS': 1.0,          # Most desirable FLS
        'Non-specific FLS': 0.0,      # Least desirable FLS
        'Not FLS': 1.0                # Not a FLS
    }

    # Lazy Load the models
    def load_models(self):
        if not self.tone_model:
            self.tone_model = ToneModel()
        if not self.FLS_model:
            self.FLS_model = FLSModel()
            
    # Method to get sentences from the user until 'x' is entered
    def get_user_sentences(self):
        print("\nPlease enter your sentences for scoring. Press Enter twice to finish input:\n")
        text = ""
        while True:
            try:
                sentence = input()
                if sentence.strip() == '':
                    print('-- Text stored succesfully --\n')
                    break
                text += sentence + "\n"
            except EOFError:
                break
        return text.strip()

    # Method to get sentiments from the Tone and FLS models for user text
    def get_sentence_sentiments(self, text):
        SentimentData = self.tone_model.get_sentiments(text)
        FLSData = self.FLS_model.get_sentimentsFLS(text)
        merged_result = self.merge_data(SentimentData, FLSData)
        
        return merged_result
    
    # Method to print overall sentiment analysis score for entire text
    def overall_sentiment_results(self, merged_result):
        overall_score = self.overall_text_sentiment_scores(merged_result)
        print("\nOverall Sentiment Analysis Results:")
        print(f"\nOverall Text Score: {round(overall_score,2)}%")
        print(f"\nTotal Sentences: {len(merged_result)}\n")
        
        return self.tone_model.sentiment_count(merged_result), self.FLS_model.sentiment_countFLS(merged_result)

    # Method to print sentiment analysis score for a given sentence
    def print_sentence_scores(self, merged_result):

        self.counter = 1
        for result in merged_result:
            
            print(f"\n{self.counter}. Sentence: {result['sentence']}")
            '''
        REVIEW:
        # The maximum possible score for every sentence
        max_sentence_score = 2.0
        for sentence in merged_result:
            # Calculate the normalized score for each sentence
            normalized_score = self.calculate_overall_sentence_score(sentence) / max_sentence_score * 100
            
            print(f"Normalized Sentence Score: {normalized_score}")  # Need this with every sentence
            '''
            # Print overall sentence sentiment
            print(f"\nOverall: '{result['LabelTone']}' and '{result['LabelFLS']}'")
            
            print("\nBreakdown:")
            # Loop through the sentiment types and print the score for each, upto 3 decimal points
            for sentiment in ['Positive', 'Negative', 'Neutral', 'Specific FLS', 'Non-specific FLS', 'Not FLS']:
                score = int(float(result[sentiment]) * 100 * 1000) / 1000.0
                if score > 0.0005:  # Only print if the score is greater than 0.0005%
                    print(f"{sentiment:<20}: {score} %")
             
            # Increment to the counter for the next sentence.
            self.counter += 1

    # Function to calculate overall sentence score
    def calculate_overall_sentence_score(self, sentence_data):
        sentence_score = sum(self.weights.get(label, 0) * probability for label, probability in sentence_data.items() if label in self.weights)
        return sentence_score
    
    def overall_text_sentiment_scores(self, merged_results):      
        # Calculate scores for each sentence and normalize
        max_sentence_score = 2  # The maximum score is set as the sum of all weights
        
        # Calculate the scores for each sentence
        sentence_scores = [self.calculate_overall_sentence_score(sentence) / max_sentence_score * 100 for sentence in merged_results]
        
        # Calculate overall score for the text
        return sum(sentence_scores) / len(sentence_scores)


    # Method to prompt the user whether to re-run or not
    def user_prompt(self, prompt, valid_responses):
        while True:
            response = input(prompt).lower()
            if response in valid_responses:
                return response
            print("Please enter a valid response: " + ', '.join(valid_responses))
            
    # Main method to run the interface
    def run(self):
        self.load_models()  # Lazy load models
        while True:
            sentences = self.get_user_sentences()
            merged_result  = self.get_sentence_sentiments(sentences)
            
            self.overall_sentiment_results(merged_result)
            self.overall_text_sentiment_scores(merged_result)
            # Print detailed analysis if the user wants
            detailed = self.user_prompt("\nDo you want a detailed sentence by sentence analysis? (y/n): ", ['y', 'n'])
            if detailed == 'y':
                self.print_sentence_scores(merged_result)

            # Ask the user whether to continue for another analysis or not
            check = self.user_prompt("\nDo you want to run another paragraph check? (y/n): ", ['y', 'n'])
            if check == 'n':
                break

# Main guard
if __name__ == "__main__":
    sentiment_interface = SentimentAnalysisInterface()
    sentiment_interface.run()
    
# Example Output:
# We have successfully optimized our operations. We now expect the age of our fleet to enhance availability and reliability due to reduced downtime for repairs.

# [{'sentence': 'We have successfully optimized our operations.', 
# 'Positive': 0.9999893, 'Neutral': 7.481316e-06, 'Negative': 3.219998e-06, 'LabelTone': 'Strong Positive', 
# 'Not FLS': 0.988741, 'Non-specific FLS': 0.0058845473, 'Specific FLS': 0.005374497, 'LabelFLS': 'Certainly Not Forward-Looking'}]