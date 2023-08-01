from FinBERT_6 import SentimentAnalysis6

#from FinBERT_3 import SentimentAnalysis3
#from Batch_Processing_6 import BatchSentimentAnalysis6

from Dataset import LetterJ02, LetterJ29, Guard13, nyt13, cleaned_texts

'''
#TEST:

AppleNET = SentimentAnalysis6()
results = AppleNET.get_sentiments(LetterJ02[0])
print(results)
AppleNET.sentiment_count(results)
df = AppleNET.sentiment_df(results)
print(df)
'''

"""
Sample:
AppleNET = SentimentAnalysis3()
results = AppleNET.get_sentiments(list[0])
AppleNET.sentiment_count(results)

Data Frame:
#For a dataframe of the results
df = AppleNET.sentiment_df(results)
print(df) 
#export dataframe as csv
df.to_csv("file_name.csv", index=False)

Extra:
AppleNET.sentiment_labels(results)
AppleNET.sentiment_conf_scores(results)     
AppleNET.sentiment_plots(results)


"""

"""
print("VR")
AppleNET = SentimentAnalysis6()
results = AppleNET.get_sentiments(cleaned_texts[0])
AppleNET.sentiment_count(results)
df = AppleNET.sentiment_df(results)
print(df) 


print("iPad")
AppleNET2 = SentimentAnalysis3()
results = AppleNET2.get_sentiments(cleaned_texts[1])
AppleNET2.sentiment_count(results)

print("Letter Jan 2")
AppleL1 = SentimentAnalysis3()
results = AppleL1.get_sentiments(LetterJ02[0])
AppleL1.sentiment_count(results)

print("Letter Jan 29")
AppleL2 = SentimentAnalysis3()
results = AppleL2.get_sentiments(LetterJ29[0])
AppleL2.sentiment_count(results)

print("NYT")
Apple2013NYT = SentimentAnalysis3()
results = Apple2013NYT.get_sentiments(nyt13[0])
Apple2013NYT.sentiment_count(results)

print("Guardian")
Apple2013G = SentimentAnalysis3()
results = Apple2013G.get_sentiments(Guard13[0])
Apple2013G.sentiment_count(results)




"""


# CHECK:
#Apple2013G = SentimentAnalysis3()
#results = Apple2013G.get_sentiments(Guard13[0])
#for result in results:
    #print(f"Sentence: {result['sentence']}")
    #print(f"Positive sentiment probability: {result['Positive']}")
    #print(f"Neutral sentiment probability: {result['Neutral']}")
    #print(f"Negative sentiment probability: {result['Negative']}")
    #print() #Apple2013G.sentiment_count(results)


