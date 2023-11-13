# STARC App

from Sentiment_Analysis_Interface import SentimentAnalysisInterface
from Web_Sentiment_Analysis import WebSentimentAnalysis


sentiment_interface = SentimentAnalysisInterface()
web_sentiment_interface = WebSentimentAnalysis()

while True:
    while True:
        print("\nDo you want to run a sentiment analysis on a URL or paste in a paragraph?")
        print("1. URL")
        print("2. Paragraph")
        choice = input()
        
        if choice in ['1', '2']:
            break
        print("Please enter a valid response ('1' or '2').")
    # Break the loop if the user chooses not to continue    

    if choice == '1':
        web_sentiment_interface.run()
        
    elif choice == '2':
        sentiment_interface.run()


            # Ask the user to continue or not
    while True:
        check = input("\nDo you want to run another sentiment analysis? (y/n): ").lower()
        if check in ['y', 'n']:
            break
        print("Please enter a valid response ('y' or 'n').")
        # Break the loop if the user chooses not to continue
    if check == 'n':
        print("\nAlways happy to help. Goodbye!")
        break
            
