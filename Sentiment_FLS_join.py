from Tone_Model import ToneModel
from FLS_Model import FLSModel

class DataMerger:
    @staticmethod
    def merge_data(list1, list2):
        # Merge corresponding dictionaries from both lists
        return [{**d1, **d2} for d1, d2 in zip(list1, list2)]

class StarcSentimentModel(ToneModel, FLSModel):
    def __init__(self):
        self.tone_model = ToneModel()
        self.FLS_model = FLSModel()
        self.merger = DataMerger()
    
    def starc_sentiments(self, text):
        SentimentData = self.tone_model.get_sentiments(text)
        FLSData = self.FLS_model.get_sentimentsFLS(text)
        merged_result = self.merger.merge_data(SentimentData, FLSData)
        
        return self.starc_sentiment_count(merged_result)

    def starc_sentiment_count(self, merged_result):
        SentimentData = self.tone_model.sentiment_count(merged_result)
        FLSData = self.FLS_model.sentiment_countFLS(merged_result)
        return SentimentData, FLSData
    
        # Method to print sentiment analysis score for a given sentence
    def print_sentence_score(self, text):
        SentimentData = self.tone_model.get_sentiments(text)
        FLSData = self.FLS_model.get_sentimentsFLS(text)
        merged_result = self.merger.merge_data(SentimentData, FLSData)
        self.counter = 1
        for result in merged_result:
            
            print(f"\n{self.counter}. Sentence: {result['sentence']}")
            print(f"Label: '{result['LabelTone']}' ")
            
            print("\nBreakdown:")
            # Loop through the sentiment types and print the score for each
            for sentiment in ['Positive', 'Neutral', 'Negative', 'Specific FLS', 'Non-specific FLS', 'not FLS']:
                print(f"{sentiment} sentiment probability:  {round(float(result[sentiment]), 4)*100} %")
            
            # Increment to the counter for the next sentence.
            self.counter += 1

        

