from GPT_rewrite import OpenAIGPTRewriter
from STARC_Cloud_Request import CloudSentimentAnalysis

from google.cloud import storage
import os
import nltk


def setup_nltk():
    # Add the nltk_data directory to NLTK's search path
    nltk_dir = os.path.join(os.getcwd(), 'nltk_data')
    nltk.data.path.append(nltk_dir)
    
    
def entry_pointSA(request):
    # API Key Check
    expected_api_key = 'AIzaSyB__dj4U5k5rvYDORBswgthA-Ov8g8hris'
    request_api_key = request.args.get('apikey')
    if not request_api_key or request_api_key != expected_api_key:
        return ('Invalid API key', 403)
    
    setup_nltk()

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
    expected_api_key = 'AIzaSyB__dj4U5k5rvYDORBswgthA-Ov8g8hris'
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

    openai_api_key = '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'

    rewriter = OpenAIGPTRewriter(openai_api_key)

    rewritten_text = rewriter.rewrite_text(user_input)
    
    # Return the GPT output
    return rewritten_text

# The following line is for local testing only
# When deployed to Google Cloud Functions, the entry_point functions will be called with the HTTP request object
if __name__ == "__main__":
    
    user_data = 'We expected economic weakness in some emerging markets. This turned out to have a significantly greater impact than we had projected.'     

    # sentiments scoring
    sa_interface = CloudSentimentAnalysis()
    scores = sa_interface.cloud_run(user_data)    
    print(scores)
    
    # Text rewriting
    openai_api_key = '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'
    
    rewriter = OpenAIGPTRewriter(openai_api_key)
    rewritten_text = rewriter.rewrite_text(user_data)
    print(rewritten_text)
    