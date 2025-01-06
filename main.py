from GPT_rewrite import OpenAIGPTRewriter
from STARC_Cloud_Request import CloudSentimentAnalysis
from dotenv import load_dotenv
import os
# Load environment variables
load_dotenv()

def entry_pointSA(request):
    # API Key Check
    expected_api_key = os.getenv('GOOGLE_CLOUD_API_KEY')
    if not expected_api_key:
        raise ValueError("Google Cloud API key not found in environment variables")
    
    request_api_key = request.args.get('apikey')
    if not request_api_key or request_api_key != expected_api_key:
        return ('Invalid API key', 403)
    
    if request.method == 'POST':
        # Extracting user input from POST request body
        request_json = request.get_json(silent=True)
        user_input = request_json.get('text') if request_json else 'Default Value'
    elif request.method == 'GET':
        user_input = request.args.get('input', 'Default Value')
    else:
        raise ValueError("Invalid request method. Only GET and POST requests are supported.")
    
    #process user input
    sa_interface = CloudSentimentAnalysis()
    # Return model's response
    scores = sa_interface.cloud_run(user_input)
    
    return scores

def entry_pointGPT(request):
    # API Key Check
    expected_api_key = os.getenv('GOOGLE_CLOUD_API_KEY')
    if not expected_api_key:
        raise ValueError("Google Cloud API key not found in environment variables")
    
    request_api_key = request.args.get('apikey')
    if not request_api_key or request_api_key != expected_api_key:
        return ('Invalid API key', 403)
    
    if request.method == 'POST':
        # Extracting user input from POST request body
        request_json = request.get_json(silent=True)
        user_input = request_json.get('text') if request_json else 'Default Value'
    elif request.method == 'GET':
        user_input = request.args.get('input', 'Default Value')
    else:
        raise ValueError("Invalid request method. Only GET and POST requests are supported.")

    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OpenAI API key not found in environment variables")
    
    rewriter = OpenAIGPTRewriter(openai_api_key)
    rewritten_text = rewriter.rewrite_text(user_input)
    
    # Return the GPT output
    return rewritten_text

if __name__ == "__main__":
    # Load environment variables for local testing
    load_dotenv()
    
    user_data_sample = 'We expected economic weakness in some emerging markets. This turned out to have a significantly greater impact than we had projected.'     
    test_user_data = "Overall, the market came in below expectations, particularly in China. Total print revenue was down 3% on a reported basis and 2% in constant currency. And while hardware units declined 2% year-over-year, total print market share increased both year-over-year and sequentially. And momentum in industrial graphics continued with supplies and services driving the fourth straight quarter of year-over-year revenue growth."
    user_input = input("Enter text something: ")
    
    # sentiments scoring
    sa_interface = CloudSentimentAnalysis()
    scores = sa_interface.cloud_run(user_input)    
    print("Order of scores = [overall, optimism, confidence, Strategic Forecasts]")
    print(scores)
    
    # Text rewriting
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OpenAI API key not found in environment variables")
    
    rewriter = OpenAIGPTRewriter(openai_api_key)
    rewritten_text = rewriter.rewrite_text(user_input)
    print(rewritten_text)
    
    scores = sa_interface.cloud_run(rewritten_text)    
    print("New Text Order of scores = [overall, optimism, confidence, Strategic Forecasts]")
    print(scores)
    