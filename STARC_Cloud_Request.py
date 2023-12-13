# STARC Model that will receive requests and give overall scoring

from Tone_Model import ToneModel
from FLS_Model import FLSModel
from Sentiment_FLS_join import DataMerger
        
# Define class for a sentiment analysis interface
class CloudSentimentAnalysis(DataMerger):

    # Initialize the sentiment analysis model on class instantiation
    def __init__(self):
        self.tone_model = ToneModel()
        self.FLS_model = FLSModel()
        
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

    # Method to get sentiments from the Tone and FLS models for user text
    def get_sentence_sentiments(self, text):
        SentimentData = self.tone_model.get_sentiments(text)
        FLSData = self.FLS_model.get_sentimentsFLS(text)
        merged_result = self.merge_data(SentimentData, FLSData)
        return merged_result
    
    def sentiment_probability_scores(self, merged_result):
        if merged_result is None:
            print("No results to process.")
            return

        # Extracting individual probabilities for each sentiment
        positives = [result['Positive'] for result in merged_result]
        neutrals = [result['Neutral'] for result in merged_result]
        negatives = [result['Negative'] for result in merged_result]
        specific_fls = [result['Specific FLS'] for result in merged_result]
        non_specific_fls = [result['Non-specific FLS'] for result in merged_result]
        not_fls = [result['Not FLS'] for result in merged_result]

        # Calculating the sum of probabilities for each sentiment
        sum_positives = sum(positives)
        sum_neutrals = sum(neutrals)
        sum_negatives = sum(negatives)
        sum_specific_fls = sum(specific_fls)
        sum_non_specific_fls = sum(non_specific_fls)
        sum_not_fls = sum(not_fls)

        # Return these sums for probabiity scores
        return sum_positives, sum_neutrals, sum_negatives, sum_specific_fls, sum_non_specific_fls, sum_not_fls
    
    # Method to print overall sentiment analysis score for entire text
    def overall_sentiment_results(self, merged_result):
        overall_score = self.overall_text_sentiment_scores(merged_result)
        
        sum_positives, sum_neutrals, sum_negatives, sum_specific_fls, sum_non_specific_fls, sum_not_fls = self.sentiment_probability_scores(merged_result)    
        
        # Total of sentiment scores
        # Equations used: 
        # Optimism = (Positive + Neutral) / (Positive + Neutral + Negative)
        # Confidence = Positive / (Positive + Negative) 
        # Specific FLS = Specific FLS / (Specific FLS + Non-specific FLS)
        
        # only for tone
        sum_for_optimism = sum_positives + sum_neutrals
        sum_for_confidence = sum_positives
        total_score_for_optimism = sum_positives + sum_neutrals + sum_negatives
        total_score_for_confidence = sum_positives + sum_negatives

        # only for forward looking statements
        total_scores_fls = sum_specific_fls + sum_non_specific_fls
        
        # Storing as probability values
        optimism_percentage = (sum_for_optimism / total_score_for_optimism) * 100
        confidence_percentage = (sum_for_confidence / total_score_for_confidence) * 100
        specific_fls_percentage = (sum_specific_fls / total_scores_fls) * 100

        # Add these to a list of sentiments
        sentiments = [round(overall_score,2), round(optimism_percentage,2), round(confidence_percentage,2), round(specific_fls_percentage,2)]

        # Return the list of sentiment scores
        return sentiments    

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

    # Main method to run the interface
    def cloud_run(self, user_input):
        merged_result  = self.get_sentence_sentiments(user_input)
        return self.overall_sentiment_results(merged_result)


# Main guard
if __name__ == "__main__":
    cloud_sentiment_analysis = CloudSentimentAnalysis()
    user_text = ' We have successfully optimized our operations. We now expect the age of our fleet to enhance availability and reliability due to reduced downtime for repairs.'
    user_text_pessimistic = 'We expected economic weakness in some emerging markets. This turned out to have a significantly greater impact than we had projected. Based on these estimates, our revenue will be lower than our original guidance for the quarter, with other items remaining broadly in line with our guidance. As we exit a challenging quarter, we may still find a way to retain the strength of our business. We use periods of adversity to re-examine our approach and use our flexibility, adaptability, and creativity to emerge better afterward.'
    user_text_optimistic = 'We had anticipated a slightly shaky economic growth in select emerging markets. This had a greater impact than we were previously expecting. However, while we anticipate a slight dip in quarterly revenue, other items remain broadly aligned with our forecast, which is promising. As we exit a challenging quarter, we are as confident as ever in the fundamental strength of our business. We have always used periods of adversity to re-examine our approach, to take advantage of our culture of flexibility, adaptability, and creativity, and to emerge better as a result.'

    print(cloud_sentiment_analysis.cloud_run(user_text_pessimistic))
    #cloud_sentiment_analysis.cloud_run(user_text_optimistic)
    
# Example Output -- merged_result:
# We have successfully optimized our operations. We now expect the age of our fleet to enhance availability and reliability due to reduced downtime for repairs.

# [{'sentence': 'We have successfully optimized our operations.', 
# 'Positive': 0.9999893, 'Neutral': 7.481316e-06, 'Negative': 3.219998e-06, 'LabelTone': 'Strong Positive', 
# 'Not FLS': 0.988741, 'Non-specific FLS': 0.0058845473, 'Specific FLS': 0.005374497, 'LabelFLS': 'Certainly Not Forward-Looking'}]