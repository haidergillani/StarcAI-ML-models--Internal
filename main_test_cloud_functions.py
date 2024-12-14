import os
from dotenv import load_dotenv
import requests
import json

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
api_key = os.getenv('GOOGLE_CLOUD_API_KEY')
if not api_key:
    raise ValueError("Google Cloud API key not found in environment variables")

# The user text to be processed
user_data = 'We expected economic weakness in some emerging markets. This turned out to have a significantly greater impact than we had projected.'

# Base URL for the API endpoints
base_url = 'https://us-central1-starcai.cloudfunctions.net'

# Sentiment Analysis endpoint
url_SA = f'{base_url}/entry_pointSA?apikey={api_key}'
response_SA = requests.post(url_SA, json={'text': user_data})
print(response_SA.text)

# GPT endpoint
url_GPT = f'{base_url}/entry_pointGPT?apikey={api_key}'
response_GPT = requests.post(url_GPT, json={'text': user_data})
print(response_GPT.text)
