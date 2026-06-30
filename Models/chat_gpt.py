import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override = True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

class ChatGPT:
    """
    A wrapper class for interacting with the OpenAI GPT API.

    Attributes:
        MODEL_NAME (str): The name of the GPT model being used (e.g., "gpt-3.5-turbo").
        client (OpenAI): The client instance for making API calls to OpenAI.
    """
    # MODEL_NAME = "gpt-4.1"
    # MODEL_NAME = "gpt-4.1-mini"
    # MODEL_NAME = "o4-mini"
    # MODEL_NAME = "o1"
    MODEL_NAME = ''

    def __init__(self, modelname):
        self.MODEL_NAME = modelname
        self.client = OpenAI()
        self.messages = []
        
    def prompt(self, prompt):
        """
        Sends a single-turn prompt to the model and returns the response.
        """
        response = self.client.chat.completions.create(
            model=self.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()



    def chat_with_model(self, message):
        """
        Sends a message in an ongoing conversation (with history).
        """
        self.messages.append({"role": "user", "content": message})

        response = self.client.chat.completions.create(
            model=self.MODEL_NAME,
            messages=self.messages,
            # verbosity="low",
            # reasoning_effort="low",
        )

        reply = response.choices[0].message.content.strip()
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def reset_chat(self):
        """
        Clears the chat history.
        """
        self.messages = []
        
    def get_chat_history(self):
        """
        Returns the current chat history.
        """
        return self.messages

    def inject_history(self, user_prompt, model_response):
        """
        Injects a simulated conversational turn into the conversation history.
        This ensures that the model knows the cached context when generating follow-up responses.
        """
        self.messages.append({"role": "user", "content": user_prompt})
        self.messages.append({"role": "assistant", "content": model_response})