import os
from google import genai
from google.genai import types
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override = True)
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

class GEMINI:
    """
    A wrapper class for interacting with the OpenAI GPT API.

    Attributes:
        MODEL_NAME (str): The name of the GPT model being used (e.g., "gpt-3.5-turbo").
        client (OpenAI): The client instance for making API calls to OpenAI.
    """
    # MODEL_NAME = "gemini-2.0-flash-thinking-exp-01-21"
    # MODEL_NAME = "gemini-2.0-flash"
    # MODEL_NAME = "gemini-2.0-flash-lite"
    # MODEL_NAME = 'gemini-1.5-flash-8b'
    # MODEL_NAME = "gemini-2.5-flash-lite-preview-06-17"
    MODEL_NAME = ''
    def __init__(self, modelname):
        """
        Initializes the ChatGPT class by setting up the OpenAI API client.

        The API key is fetched from the environment variables via `dotenv`. The client
        is an instance of the `OpenAI` class, which handles API interactions.
        """
        self.MODEL_NAME = modelname
        self.client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])
        self.chat = self.client.chats.create(model=modelname)

        self.current_input_tokens = 0
        self.current_output_tokens = 0
    def prompt(self, prompt):
        """
        Sends a prompt to the GPT model and retrieves the model's response.

        Constructs an API request configuration with the specified prompt, sends it
        to the GPT model, and parses the response to extract the model's reply.

        Args:
            prompt (str): The input prompt to send to the GPT model.

        Returns:
            str: The content of the model's response extracted from the API result.
        """

        response = self.client.models.generate_content(
            model=self.MODEL_NAME,
            contents=prompt,
        )
        # usage_data = response.usage_metadata

        # # Ensure log directory exists
        # log_dir = './logs'
        # if not os.path.exists(log_dir):
        #     os.makedirs(log_dir)
        
        # # Append usage data with a timestamp to the log file
        # log_file_path = os.path.join(log_dir, 'gemini.logs')
        # with open(log_file_path, 'a') as log_file:
        #     log_file.write(str(usage_data) + '\n')
        return response.text
    def chat_with_model(self, message):
        """
        Engage in a chat with the model using a series of messages.

        Args:
            messages (list): A list of messages to send to the model.

        Returns:
            str: The model's response to the chat.
        """
        response = self.chat.send_message(message)
        if response.usage_metadata:
            self.current_input_tokens += response.usage_metadata.prompt_token_count
            self.current_output_tokens += response.usage_metadata.candidates_token_count

        # usage_data = response.usage_metadata
        # # Ensure log directory exists
        # log_dir = './logs'
        # if not os.path.exists(log_dir):
        #     os.makedirs(log_dir)
        
        # # Append usage data with a timestamp to the log file
        # log_file_path = os.path.join(log_dir, 'gemini.logs')
        # with open(log_file_path, 'a') as log_file:
        #     log_file.write(str(usage_data) + '\n')
        return response.text
    def reset_chat(self):
        """
        Resets the chat session by creating a new chat instance.
        
        This clears all previous conversation history and starts fresh.
        """
        self.chat = self.client.chats.create(model=self.MODEL_NAME)
        self.current_output_tokens = 0
        self.current_input_tokens = 0

    def get_and_reset_turn_tokens(self):
        """Gibt Input und Output zurück und resettet."""
        in_tok = self.current_input_tokens
        out_tok = self.current_output_tokens
        self.current_input_tokens = 0
        self.current_output_tokens = 0
        return in_tok, out_tok

    def get_chat_history(self):
        """
        Retrieves the chat history of the current session.

        Returns:
            list: A list of messages exchanged in the chat session.
        """
        return self.chat.get_history()

    def inject_history(self, user_prompt, model_response):
        """
        Injects a simulated conversational turn into the conversation history.
        This ensures that the model knows the cached context when generating follow-up responses.
        """
        # Wir bauen die korrekten SDK-Objekte für User- und Modell-Nachrichten
        user_content = types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_prompt)]
        )
        model_content = types.Content(
            role="model",
            parts=[types.Part.from_text(text=model_response)]
        )

        # Anhängen an die bestehende Historie des Chat-Objekts
        history = self.get_chat_history()
        history.extend([user_content, model_content])