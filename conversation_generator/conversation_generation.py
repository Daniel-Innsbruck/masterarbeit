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
from context_discoverer.context_discoverer import ContextDiscoverer
from utils.chroma_connector import ChromaConnector
import utils.parser as parser
import conversation_validator.conversation_validator as conversation_validator

# API Endpoints
API_URL = "http://localhost:8000/rag"
API_URL_THREAD_ID = "http://localhost:8000/getThreadID"

# Model-Setup
model_name = 'gemini-3.1-flash-lite'
model = gemini.GEMINI(model_name)
parser = parser.LLMResponseParser()

# Methodology-Params (Master-Thesis)

ALPHA = 2
BETA = 3
EXPANSION_DIR = "below"

max_conversations = 10 # number conversations'
n=5 # number turns

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
    response = None
    while not success:
        try:
            llm_response = model.chat_with_model(prompt)
            response = parser.parse_and_validate(llm_response)
            if response != "":
                success = True
        except Exception as e:
            if '429' in str(e) or '503' in str(e):
                print("Rate limit or service unavailable. Waiting 60s...")
                time.sleep(60)
            else:
                success = True
                print(f"Error generating prompt data: {e}")
                return None
    return response


# =========================================================
# Hilfsfunktionen
# =========================================================

def _build_context_string(chunk_list):
    return "\n\n".join([c['text_snippet'] for c in chunk_list])

# =========================================================
# Turn 0: Multi-Hop Generation (Cross-Document)
# =========================================================

def get_initial_multihop_prompt_data(chunk_a, chunk_b, t_bridge):
    prompt = templates.CONVERSATION_PROMPTS['init_multihop_prompt'].format(
        Role=Role,
        t_bridge=t_bridge,
        chunk_a=chunk_a['text_snippet'],
        chunk_b=chunk_b['text_snippet']
    )

    answer = send_request_to_LLM_conversation(prompt)

    combined_docs = f"Snippet A: {chunk_a['text_snippet']}\n\nSnippet B: {chunk_b['text_snippet']}"

    validation = conversation_validator.validate_init_prompt_all_in_one(answer, combined_docs)
    if validation and not validation['correct']:
        print(f"Initial prompt validation failed: {validation['reason']}")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"Validation failed for initial multi-hop.\nAnswer: {answer}\nReason: {validation['reason']}\n\n")

        answer = send_request_to_LLM_conversation(templates.CONVERSATION_PROMPTS['rephrase_init_prompt'].format(
            Role=Role,
            reason=validation['reason']
        ))

    return answer


# =========================================================
# Turn 1-n: Follow-up Generation
# =========================================================

def get_follow_up_question(answer, active_chunks):
    follow_up_prompt = templates.CONVERSATION_PROMPTS['follow_up_prompt'].format(
        Role=Role,
        context_a=_build_context_string(active_chunks['A']),
        context_b=_build_context_string(active_chunks['B']),
        RAG_answer=answer
    )

    history = model.get_chat_history()
    response = send_request_to_LLM_conversation(follow_up_prompt)

    validation = conversation_validator.validate_follow_up_question_all_in_one(response, history)

    if validation and not validation['correct']:
        print(f"Follow-up validation failed: {validation['reason']}")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"Follow-up Validation failed.\nAnswer: {response}\nReason: {validation['reason']}\n\n")

        response = send_request_to_LLM_conversation(templates.CONVERSATION_PROMPTS['rephrase_follow_up_prompt'].format(
            Role=Role,
            reason=validation['reason']
        ))

    return response


# =========================================================
# Main Evaluation Loop
# =========================================================

def generate_conversation():
    db_connector = ChromaConnector('../data/v_eval/')
    cd = ContextDiscoverer(db_connector=db_connector, llm_model=model, k=4)
    counter = 0
    failed_counter = 0

    while counter < max_conversations:
        conv = []
        try:
            res = requests.get(API_URL_THREAD_ID)
            thread_id = res.json().get("thread_id", "")
        except Exception as e:
            print(f"Error getting thread ID: {e}")
            return

        print(f"\n--- Conversation {counter + 1}/{max_conversations} ---")
        print("Searching for context bridge (Cross-Document) via Context Discoverer...")

        discovered_context = cd.discover_valid_context()

        if not discovered_context:
            print("DB empty or no semantic bridges could be found. Aborting.")
            break

        chunk_a = discovered_context['chunk_a']
        chunk_b = discovered_context['chunk_b']
        t_bridge = discovered_context['t_bridge']

        active_chunks = {"A": [chunk_a], "B": [chunk_b]}
        print(
            f"Validated Context Bridge found! Topics: {t_bridge}\n"
            f" A -> Article: {chunk_a['article_id']} [Index {chunk_a['chunk_index']}/{chunk_a.get('total_chunks', '?')}]\n"
            f" B -> Article: {chunk_b['article_id']} [Index {chunk_b['chunk_index']}/{chunk_b.get('total_chunks', '?')}]"
        )

        question = get_initial_multihop_prompt_data(chunk_a, chunk_b, t_bridge)
        failed = question is None

        if not failed:
            for turn_idx in range(n):
                if failed:
                    print(f"Aborting conversation at turn {turn_idx + 1}.")
                    break

                current_indices_a = [c['chunk_index'] for c in active_chunks["A"]]
                current_indices_b = [c['chunk_index'] for c in active_chunks["B"]]
                print(f"Turn {turn_idx + 1} | Active Chunks: A{current_indices_a}, B{current_indices_b}")

                try:
                    res = requests.post(API_URL, json={"question": question['rag_input'], "thread_id": thread_id})
                    answer = res.json().get("answer", "")
                    context = res.json().get("context", "")

                    conv.append({
                        "rag_input": question['rag_input'],
                        "question": question['question'],
                        "answer": question['answer'],
                        "type": question['type'],
                        "logic_type": question.get('logic_type', 'none'),
                        "multi_hop_flag": question.get('multi_hop_flag', 0),
                        "bridging_topic": question.get('bridging_topic', None),
                        "rag_answer": answer,
                        "context": context,
                        "turn_index": turn_idx,
                        "ground_truth_chunks": [c['id'] for c in active_chunks["A"]] + [c['id'] for c in
                                                                                        active_chunks["B"]]
                    })
                except Exception as e:
                    print(f"Error during RAG request: {e}")
                    failed = True
                    break

                if turn_idx < n - 1:
                    next_turn = turn_idx + 2

                    if next_turn == ALPHA:
                        adj_chunk = db_connector.get_adjacent_chunk(
                            article_id=active_chunks["A"][0]['article_id'],
                            current_indices=current_indices_a,
                            total_chunks=active_chunks["A"][0].get('total_chunks', 1),
                            direction=EXPANSION_DIR
                        )
                        if adj_chunk:
                            active_chunks["A"].append(adj_chunk)
                            print(f" -> [Context Expansion] Expanded A with index {adj_chunk['chunk_index']}")

                    if next_turn == BETA:
                        adj_chunk = db_connector.get_adjacent_chunk(
                            article_id=active_chunks["B"][0]['article_id'],
                            current_indices=current_indices_b,
                            total_chunks=active_chunks["B"][0].get('total_chunks', 1),
                            direction=EXPANSION_DIR
                        )
                        if adj_chunk:
                            active_chunks["B"].append(adj_chunk)
                            print(f" -> [Context Expansion] Expanded B with index {adj_chunk['chunk_index']}")

                    question = get_follow_up_question(answer, active_chunks)
                    failed = question is None

            model.reset_chat()

        if not failed:
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
                print(
                    f"TURN {turn['turn_index'] + 1} | Type: {turn['type']} | Logic: {turn['logic_type']} | Multi-Hop: {turn['multi_hop_flag']}")
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