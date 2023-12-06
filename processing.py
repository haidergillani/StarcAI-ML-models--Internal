from flask import Flask, render_template, request
import openai
import os
from conversion import code_convert

key = os.environ['API']
openai.api_key = key

if not key:
  raise Exception("API key not found in environment variables")


import openai


def code_convert(material, chat_log=None):
  if chat_log is None:
    chat_log = []
  chat_log.append({
    'role':
    'system',
    'content':
    'You are assistant to convert code into different languages'
  })
  chat_log.append({'role': 'user', 'content': material})

  response = openai.ChatCompletion.create(model='gpt-3.5-turbo',
                                          messages=chat_log)
  chat_log.append({
    'role': 'assistant',
    'content': response['choices'][0]['message']['content']
  })
  return response['choices'][0]['message']['content'], chat_log



app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
  converted_code = ''
  if request.method == 'POST':
    org_lang = request.form['org_lang']
    new_lang = request.form['new_lang']
    code = request.form['code']
    prompt = f"Convert the following code from {org_lang} into {new_lang}:\n{code}"
    converted_code, _ = code_convert(prompt)

  return render_template('index.html', converted_code=converted_code)


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)
