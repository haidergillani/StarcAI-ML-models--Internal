import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

class OpenAIGPTRewriter:
    def __init__(self):
        # Get API key from environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("API key not found in environment variables")
        self.client = OpenAI(api_key=api_key)

    def rewrite_text(self, original_text):
        # Define the context for the rewriting task
        system_message = 'You are assistant to financial communication text into more optimistic and confidence inducing in the investors and readers. Make sure to not change key information and be aware of context. Ver important: keep the number of sentences same and the order of sentences consistent. Ignore words like rewrite etc and focus on the actual sentence content'
        # Create the chat messages
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": original_text}
        ]
        # Generate the completion using the OpenAI API
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model="gpt-3.5-turbo"
        )
        # Extract the rewritten text
        rewritten_text = chat_completion.choices[0].message.content
        return rewritten_text

if __name__ == '__main__':
    rewriter = OpenAIGPTRewriter()
    user_input = 'We expected economic weakness in some emerging markets. This turned out to have a significantly greater impact than we had projected.'
    rewritten_text = rewriter.rewrite_text(user_input)
    print(rewritten_text)
    
