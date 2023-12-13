import requests
import json

api_key = 'AIzaSyB__dj4U5k5rvYDORBswgthA-Ov8g8hris'

# The user text to be processed
user_data = 'We expected economic weakness in some emerging markets. This turned out to have a significantly greater impact than we had projected.'


# Constructing the URL with the API key
url_SA = f'https://us-central1-starcai.cloudfunctions.net/entry_pointSA?apikey={api_key}'
# Making the POST request with JSON data
response_SA = requests.post(url_SA, json={'text': user_data})
# Printing the response from the function
print(response_SA.text)

# Constructing the URL with the API key
url_GPT = f'https://us-central1-starcai.cloudfunctions.net/entry_pointGPT?apikey={api_key}'
# Making the POST request with JSON data
response_GPT = requests.post(url_GPT, json={'text': user_data})

# Printing the response from the function
print(response_GPT.text)


