from FinBERT_6 import SentimentAnalysis6

#from FinBERT_3 import SentimentAnalysis3
#from Batch_Processing_6 import BatchSentimentAnalysis6

from Web_Scrapping_Text_Cleaning import WebPageScraper

"""
# for a data frame of sentence results:
df = self.sentiment_model.sentiment_df(results)
print(df)
"""


# Define class for a web sentiment analysis interface
class WebSentimentAnalysis:

    # Initialize the sentiment analysis model on class instantiation
    def __init__(self):
        self.sentiment_model = SentimentAnalysis6()

    # Method to get sentences from the user until 'x' is entered
    def get_user_urls_text(self):
        urls = []
        while True:
            url = input("\nPlease enter the web url for scoring:\n")
            if url.isdigit() or len(url.strip()) <= 1:
                print("Please enter a valid url.")
                continue
            else:
                break
        urls.append(url)

        scraper = WebPageScraper(urls)
        cleaned_texts = scraper.scrape_all()
        return cleaned_texts
        
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
        while True:
            cleaned_texts = self.get_user_urls_text()

            if not cleaned_texts:
                print("\nNo URL provided. Please provdide a valid URL.")
                continue
            
            self.counter = 1
            for text in cleaned_texts:
                results = self.get_urls_sentiments(text)                
                self.overall_sentiment_results(results)
                 
            # Ask the user if the want a sentence by sentence analysis
            while True:
                detailed = input("\nDo you want a detailed sentence by sentence analysis? (y/n): ").lower()
                if detailed in ['y', 'n']:
                    break
                print("Please enter a valid response ('y' or 'n').")
                
            # Print detailed analysis if the user chooses yes
            if detailed == 'y':
                print("\nHere's a detailed analysis:\n")
                self.print_sentence_score(results)

            # Ask the user to continue for another check or not
            while True:
                check = input("\nDo you want to run another web check? (y/n): ").lower()
                if check in ['y', 'n']:
                    break
                print("Please enter a valid response ('y' or 'n').")
            # Break the loop if the user chooses not to continue
            if check == 'n':
                break
