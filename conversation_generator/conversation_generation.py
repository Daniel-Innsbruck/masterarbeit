import requests
import sys
import os
import random
import gc
import time
import json

# Füge das übergeordnete Verzeichnis zum Python-Pfad hinzu
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Models.gemini as gemini
import Models.chat_gpt as chat_gpt
import templates as templates
from utils.chroma_connector import ChromaConnector
import utils.parser as parser
import conversation_validator.conversation_validator as conversation_validator

# API Endpoints
API_URL = "http://localhost:8000/rag"
API_URL_THREAD_ID = "http://localhost:8000/getThreadID"

# Model-Setup
model_name = 'gemini-2.5-flash'
model = gemini.GEMINI(model_name)
parser = parser.LLMResponseParser()

# Methodology-Params (Master-Thesis)

ALPHA_HOP_PROBABILITY = 0.3
QUERY_TYPES = ["inference", "comparison", "temporal"]

max = 3 # number conversations'
n=4 # number turns

# Logging & Output
output_file = "./data/conversation_data_" + model_name + "_turns_" + str(n) + "_conversation_" +str(max)+ ".jsonl"
log_file = "./data/conversation_data_" + model_name + "_turns_" + str(n) + "_conversation_" +str(max)+ ".log"
Role = "You are a highly attentive conversationalist who asks context-aware questions. Your questions should build naturally on previous exchanges, using referring expressions like 'this', 'that', or 'it' to maintain coherence and continuity."
# Role = "Your questions are very short and precise"
# Role = "You are a very confused and forgetful person who always misunderstands what has been said. You repeatedly ask the same questions as if you never heard the answer, often mixing up details and getting things wrong. Your questions are unclear or off-topic, and you struggle to follow the flow of conversation, causing you to constantly reask and seek clarification."

# =========================================================
# LLM Request Wrapper
# =========================================================

def send_request_to_LLM_conversation(prompt):
    success = False
    while not success:
        try:
            llm_response = model.chat_with_model(prompt)
            response = parser.parse_and_validate(llm_response)
            if response != "":
                success = True
        except Exception as e:
            if '429' in str(e):
                print(f"Rate limit exceeded. Waiting for 60 seconds...")
                time.sleep(60)
            elif '503' in str(e):
                print(f"Service Unavailable. Waiting for 60 seconds...")
                time.sleep(60)
            else:
                success = True
                print(f"Error generating prompt data: {e}")
                return None
    return response

# =========================================================
# Turn 0: Multi-Hop Generation (Cross-Document)
# =========================================================

def get_initial_multihop_prompt_data(chunk_a, chunk_b):
    """
    Generates the initial multi-hop query based on two chunks
    Uses 'init_multihop_prompt' from Appendix A.1.
    """
    query_type = random.choice(QUERY_TYPES)

    prompt = templates.CONVERSATION_PROMPTS['init_multihop_prompt'].format(
        query_type=query_type,
        chunk_a=chunk_a['text_snippet'],
        chunk_b=chunk_b['text_snippet']
    )

    answer = send_request_to_LLM_conversation(prompt)
    if answer is None:
        return None

    if not all(k in answer for k in ['rag_input', 'question', 'answer']):
        return None

    # combine cods for validator
    combined_docs = f"Snippet A: {chunk_a['text_snippet']}\n\nSnippet B: {chunk_b['text_snippet']}"

    validation = conversation_validator.validate_init_prompt_all_in_one(answer, combined_docs)
    if validation and not validation['correct']:
        print(f"Initial prompt validation failed: {validation['reason']}")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"Validation failed for initial multi-hop.\nAnswer: {answer}\nReason: {validation['reason']}\n\n")

        answer = send_request_to_LLM_conversation(templates.CONVERSATION_PROMPTS['rephrase_init_prompt'].format(
            reason=validation['reason']
        ))
        if answer is None or not all(k in answer for k in ['rag_input', 'question', 'answer']):
            return None

    return answer

# =========================================================
# Turn 1-n: Follow-up Generation
# =========================================================

def get_follow_up_question(answer, active_chunks):
    """
    Generates Follow-Up Questions. active_chunks is used for logging,
    since the llm has the context in their history.
    """
    follow_up_prompt = templates.CONVERSATION_PROMPTS['follow_up_prompt'].format(
        RAG_answer=answer
    )
    history = model.get_chat_history()
    response = send_request_to_LLM_conversation(follow_up_prompt)

    if response is None or not all(k in response for k in ['rag_input', 'type', 'question', 'answer']):
        return None

    validation = conversation_validator.validate_follow_up_question_all_in_one(response, history)

    if validation and not validation['correct']:
        print(f"Follow-up validation failed: {validation['reason']}")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"Follow-up Validation failed.\nAnswer: {response}\nReason: {validation['reason']}\n\n")

        response = send_request_to_LLM_conversation(templates.CONVERSATION_PROMPTS['rephrase_follow_up_prompt'].format(
            reason=validation['reason']
        ))
        if response is None or not all(k in response for k in ['rag_input', 'type', 'question', 'answer']):
            return None

    return response


# =========================================================
# Main Evaluation Loop
# =========================================================
def generate_conversation():
    db_connector = ChromaConnector()
    counter = 0
    failed_counter = 0

    while counter < max:
        conv = []
        try:
            res = requests.get(API_URL_THREAD_ID)
            thread_id = res.json().get("thread_id", "")
        except Exception as e:
            print(f"Error getting thread ID: {e}")
            return

        print(f"\n--- Conversation {counter + 1}/{max} ---")
        print("Searching for context bridge (Cross-Document)...")

        # 1. Stochastic Multi-Hop Context Discovery (find d_A and d_B)
        chunk_a = db_connector.get_random_chunk()
        if not chunk_a:
            print("DB is empty!")
            break

        chunk_b = db_connector.get_bridging_chunk(chunk_a['text_snippet'], chunk_a['article_id'])

        if not chunk_b:
            print("Could not find a matching partner chunk from a different article. Retrying...")
            continue

        active_chunks = {"A": chunk_a, "B": chunk_b}
        print(
            f"Initial Chunks found:\n A -> Article: {chunk_a['article_id']} [Index {chunk_a['chunk_index']}]\n B -> Article: {chunk_b['article_id']} [Index {chunk_b['chunk_index']}]")

        # 2. Generate initial Multi-Hop question
        question = get_initial_multihop_prompt_data(active_chunks["A"], active_chunks["B"])
        failed = question is None

        if not failed:
            for i in range(n):
                if failed:
                    print(f"Aborting conversation at turn {i + 1}.")
                    break

                print(
                    f"Turn {i + 1} | Active Chunks: A[{active_chunks['A']['chunk_index']}], B[{active_chunks['B']['chunk_index']}]")

                # Send question to your local RAG system
                try:
                    res = requests.post(API_URL, json={"question": question['rag_input'], "thread_id": thread_id})
                    answer = res.json().get("answer", "")
                    context = res.json().get("context", "")

                    # Log the turn
                    conv.append({
                        "rag_input": question['rag_input'],
                        "question": question['question'],
                        "answer": question['answer'],
                        "type": question.get('logic_type') if i == 0 else question.get('type', "Initial"),
                        "rag_answer": answer,
                        "context": context,
                        "turn_index": i,
                        "ground_truth_chunks": [active_chunks["A"]['id'], active_chunks["B"]['id']]
                    })
                except Exception as e:
                    print(f"Error during RAG request: {e}")
                    failed = True
                    break

                # 3. Dynamic Context Progression (Intra-Document Hop) for the NEXT turn
                if i < n - 1:
                    # Probability check (Alpha)
                    if random.random() < ALPHA_HOP_PROBABILITY:
                        target_key = random.choice(["A", "B"])
                        current_chunk = active_chunks[target_key]

                        # Fetch new chunk from the SAME article
                        new_chunk = db_connector.get_intra_document_chunk(
                            current_chunk['article_id'],
                            exclude_chunk_indices=[current_chunk['chunk_index']]
                        )

                        if new_chunk:
                            print(
                                f" -> [Intra-Document Hop!] Swapping Document {target_key} from Index {current_chunk['chunk_index']} to {new_chunk['chunk_index']}")
                            active_chunks[target_key] = new_chunk

                    # Generate follow-up question based on the RAG answer and chat history
                    question = get_follow_up_question(answer, active_chunks)
                    failed = question is None

                    # Reset LLM (Gemini) chat history for the next fresh conversation
        model.reset_chat()

        if not failed:
            # Success! Save conversation in the new JSONL format.
            data_item = {
                "parent_doc_A": chunk_a['article_id'],
                "parent_doc_B": chunk_b['article_id'],
                "role": Role,
                "conversation": conv
            }
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data_item, ensure_ascii=False) + "\n")
            print("\n" + "=" * 60)
            print(f"SUCCESS! CONVERSATION GENERATED:")
            print(f"Doc A: {chunk_a['article_id']}")
            print(f"Doc B: {chunk_b['article_id']}")
            print("-" * 60)

            for turn in conv:
                print(f"TURN {turn['turn_index'] + 1} [Type: {turn['type']}]")
                print(f"Simulated User : {turn['rag_input']}")
                print(f"Target RAG     : {turn['rag_answer']}")
                print("-" * 60)
            print("=" * 60 + "\n")
            counter += 1
            failed_counter = 0
        else:
            failed_counter += 1
            print(f"Failed conversations in a row: {failed_counter}")

        if failed_counter >= 3:
            print(f"Stopping evaluation after {failed_counter} consecutive failures.")
            break


if __name__ == "__main__":
    generate_conversation()