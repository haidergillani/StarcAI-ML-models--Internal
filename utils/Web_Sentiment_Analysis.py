from Tone_Model import ToneModel
from FLS_Model import FLSModel
from Sentiment_FLS_join import DataMerger

from Web_Scrapping_Text_Cleaning import WebPageScraper

# Define class for a web sentiment analysis interface
class WebSentimentAnalysis(DataMerger):

    # Initialize the sentiment analysis model on class instantiation
    def __init__(self):
        self.tone_model = None
        self.FLS_model = None

    # Lazy Load the models
    def load_models(self):
        if not self.tone_model:
            self.tone_model = ToneModel()
        if not self.FLS_model:
            self.FLS_model = FLSModel()

    # Method to get sentences from the user until 'x' is entered
    def get_user_urls_text(self):
        urls = []
        print("Please enter web urls for scoring, on separate lines. Press 'Enter' twice to exit:\n")
        while True:
            try:
                url = input()
                if url.strip() == '':
                    print('urls received succesfully.\n')
                    break
                elif url.isdigit() or len(url.strip()) <= 1:
                    print("Please enter a valid url.")
                    continue
                urls.append(url)

            except EOFError:
                break

        scraper = WebPageScraper(urls)
        cleaned_texts = scraper.scrape_all()
        if not cleaned_texts:
            raise Exception("Unable to access the provided URL. Please provide a valid URL.")
        return cleaned_texts
        
    # Method to get sentiments from the Tone and FLS models for cleaned url texts
    def get_urls_sentiments(self, cleaned_texts):
        SentimentData = self.tone_model.get_sentiments(cleaned_texts)
        FLSData = self.FLS_model.get_sentimentsFLS(cleaned_texts)
        merged_result = self.merge_data(SentimentData, FLSData)
        return merged_result
    
    # Method to print overall sentiment analysis score for entire text
    def overall_sentiment_results(self, merged_result):
        return self.tone_model.sentiment_count(merged_result), self.FLS_model.sentiment_countFLS(merged_result)

    # Method to print sentiment analysis score for a given sentence
    def print_sentence_score(self, merged_result):

        self.counter = 1
        for result in merged_result:
            
            print(f"\n{self.counter}. Sentence: {result['sentence']}")

            # Print overall sentence sentiment
            print(f"\nOverall: {result['LabelTone']}")

            # Print FLS sentiment and score to 3 decimal points
            print(f"{result['LabelFLS']}: {int(float(result['ScoreFLS']) * 100 * 1000) / 1000.0} %")            
            
            print("\nBreakdown:")
            
            # Loop through the sentiment types and print the score for each, upto 3 decimal points
            for sentiment in ['Positive', 'Negative', 'Neutral']:
                print(f"{sentiment}: {int(float(result[sentiment]) * 100 * 1000) / 1000.0} %")
             
            # Increment to the counter for the next sentence.
            self.counter += 1

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
            cleaned_texts = self.get_user_urls_text()

            if not cleaned_texts:
                print("\nNo URL provided. Please provide a valid URL.")
                continue
            
            self.urlcounter = 1
            for text in cleaned_texts:
                print(f"\nURL {self.urlcounter} Analysis:")  
                merged_result = self.get_urls_sentiments(text)
                self.overall_sentiment_results(merged_result)
                 
                # Print detailed analysis if the user wants
                detailed = self.user_prompt("\nDo you want a detailed sentence by sentence analysis? (y/n): ", ['y', 'n'])
                if detailed == 'y':
                    self.print_sentence_score(merged_result)
                self.urlcounter += 1
                    
            # Ask the user whether to continue for another analysis or not
            check = self.user_prompt("\nDo you want to run another web check? (y/n): ", ['y', 'n'])
            if check == 'n':
                break

# We have successfully optimized our operations. We now expect the age of our fleet to enhance availability and reliability due to reduced downtime for repairs.

# Main guard
if __name__ == "__main__":
    sentiment_interface = WebSentimentAnalysis()
    sentiment_interface.run()