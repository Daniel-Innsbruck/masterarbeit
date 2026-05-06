
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from . import templatesValidation as templates
import Models.gemini as gemini
import Models.chat_gpt as chat_gpt
from .utils import parser_validation as parser
import time

model_name = 'gemini-2.5-flash'  # Change to 'gemini-2.0-flash' if needed
# model_name= 'gpt-5-nano'
model = gemini.GEMINI(model_name)
# model = chat_gpt.ChatGPT(model_name)
parser = parser.LLMResponseParser()

def send_request_to_LLM_validation(prompt):
    success = False
    while not success:
        try:
            llm_response = model.prompt(prompt)
            response = parser.parse_and_validate_validation(llm_response)
            if response != "":
                success = True
        except Exception as e:
            if '429' in str(e):
                print(f"Rate limit exceeded. Waiting for 60 seconds before evaluating next batch")
                time.sleep(60)
                # try again
            elif '503' in str(e):
                print(f"Service Unavailable. Waiting for 60 seconds before evaluating next batch")
                time.sleep(60)
                # try again
            else:
                success = True
                print(f"Error generating prompt data: {e}")
                return None
    return response



# Validate and rephrase prompts in one step
def validate_init_prompt_all_in_one(question,document):
    question_dict = {
        'rag_input': question.get('rag_input', question.get('question')),
        'question': question.get('question', ''),
        'answer': question.get('answer', ''),
        'thematic_link': question.get('thematic_link', ''),
        'logic_type': question.get('logic_type', '')
    }

    validate_prompt = templates.VALIDATION_PROMPTS['validate_init_prompt'].format(
        question=str(question_dict),
        document=str(document)
    )
    answer = send_request_to_LLM_validation(validate_prompt)

    return answer

# Validate follow-up questions in one step
def validate_follow_up_question_all_in_one(question, history):
    validate_prompt = templates.VALIDATION_PROMPTS['validate_follow_up_prompt'].format(
        question=str({'rag_input':question['rag_input'], 'type':question['type'], 'question':question['question'], 'answer':question['answer']}),
        conversation_history=str(history)
    )
    # print(f"Validate Prompt: {validate_prompt}")
    answer = send_request_to_LLM_validation(validate_prompt)
    return answer
        