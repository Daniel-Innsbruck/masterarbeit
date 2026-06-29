
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from . import templatesValidation as templates
import Models.gemini as gemini
import Models.chat_gpt as chat_gpt
from .utils import parser_validation as parser
import time

#model_name = 'gemini-2.5-flash'  # Change to 'gemini-2.0-flash' if needed
model_name= 'gpt-5-mini'
# model = gemini.GEMINI(model_name)
model = chat_gpt.ChatGPT(model_name)
parser = parser.LLMResponseParser()

def send_request_to_LLM_validation(prompt):
    success = False
    response = None
    while not success:
        try:
            # We use .prompt() for single-turn stateless validation
            llm_response = model.prompt(prompt)
            response = parser.parse_and_validate_validation(llm_response)
            if response != "":
                success = True
        except Exception as e:
            if '429' in str(e) or '503' in str(e):
                print("Rate limit or Server Error. Waiting for 60 seconds before evaluating next batch")
                time.sleep(60)
            else:
                success = True
                print(f"Error generating validation data: {e}")
                return None
    return response


def validate_init_prompt_all_in_one(question, document):
    # Capture all new keys for Multi-Hop validation
    if question is None:
        return {"correct": False,
                "reason": "LLM output format was invalid or missing required keys (Parser returned None)."}
    question_dict = {
        'rag_input': question.get('rag_input', question.get('question')),
        'question': question.get('question', ''),
        'answer': question.get('answer', ''),
        'type': question.get('type', 'Initial'),
        'logic_type': question.get('logic_type', ''),
        'multi_hop_flag': question.get('multi_hop_flag', 1),
        'bridging_topic': question.get('bridging_topic', '')
    }

    validate_prompt = templates.VALIDATION_PROMPTS['validate_init_prompt'].format(
        question=str(question_dict),
        document=str(document)
    )

    answer = send_request_to_LLM_validation(validate_prompt)
    return answer


# Validate follow-up questions in one step
def validate_follow_up_question_all_in_one(question, history, current_active_context):
    # Capture all keys including logic_type and multi_hop_flag
    question_dict = {
        'rag_input': question.get('rag_input', ''),
        'type': question.get('type', ''),
        'question': question.get('question', ''),
        'answer': question.get('answer', ''),
        'logic_type': question.get('logic_type', ''),
        'multi_hop_flag': question.get('multi_hop_flag', 0)
    }

    # Check flag purely in Python logic
    is_multihop = int(question_dict['multi_hop_flag']) == 1

    if is_multihop:
        template_name = 'validate_follow_up_multihop_prompt'
    else:
        template_name = 'validate_follow_up_singlehop_prompt'

    validate_prompt = templates.VALIDATION_PROMPTS[template_name].format(
        question=str(question_dict),
        conversation_history=str(history),
        active_context=current_active_context
    )

    answer = send_request_to_LLM_validation(validate_prompt)
    return answer