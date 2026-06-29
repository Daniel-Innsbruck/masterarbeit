import requests
import sys
import os

# Füge das übergeordnete Verzeichnis zum Python-Pfad hinzu
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import gc
import time
import json

from dialogue_cache.dialogue_cache import DialogueCache


import Models.gemini as gemini
import Models.chat_gpt as chat_gpt
import templates as templates
from context_discoverer.context_discoverer import ContextDiscoverer
from utils.chroma_connector import ChromaConnector
import utils.parser as parser
import conversation_validator.conversation_validator as conversation_validator

# API Endpoints
API_URL = "http://localhost:8080/rag"
API_URL_THREAD_ID = "http://localhost:8080/getThreadID"

# Model-Setup
model_name = 'gemini-3.1-flash-lite'
model = gemini.GEMINI(model_name)
parser = parser.LLMResponseParser()

# Methodology-Params (Master-Thesis)

ALPHA = 2
BETA = 3
EXPANSION_DIR = "below"
CACHE_TAU = 0.95
CACHE_SAFEGUARD = True
CACHE_PATH = "./data/dialogue_cache.json"

MAX_RETRIES = 3  # Max retries per turn (independent of n)
MAX_CONSECUTIVE_FAILS = 5 # max number of complete conversation restarts before qa-gernation ends early (avoid infinite loop)

# dialog configs
n = 5             # Target number of turns per conversation
max_conversations = 1 # number conversations'

# Logging & Output
output_file = "./data/2906_conversation_data_" + model_name + "_turns_" + str(n) + "_conversations_" + str(max_conversations)+ ".jsonl"
log_file = "./data/2906_conversation_data_" + model_name + "_turns_" + str(n) + "_conversations_" + str(max_conversations)+ ".log"

# Role = "You are a highly attentive conversationalist who asks context-aware questions. Your questions should build naturally on previous exchanges, using referring expressions like 'this', 'that', or 'it' to maintain coherence and continuity."
Role = "Your questions are very short and precise"
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


def _build_active_context_string_for_validator(active_chunks):
    """Baut einen klar strukturierten String aus allen aktuell aktiven Chunks."""
    context_a_str = _build_context_string(active_chunks["A"])
    context_b_str = _build_context_string(active_chunks["B"])

    return (
        f"--- CURRENT CONTEXT A ---\n{context_a_str}\n\n"
        f"--- CURRENT CONTEXT B ---\n{context_b_str}"
    )


# =========================================================
# Turn 0: Multi-Hop Generation (Cross-Document)
# =========================================================

def get_initial_multihop_prompt_data(chunk_a, chunk_b, t_bridge, max_retries=MAX_RETRIES):
    combined_docs_for_validator = (
        f"--- CONTEXT A ---\n{_build_context_string([chunk_a])}\n\n"
        f"--- CONTEXT B ---\n{_build_context_string([chunk_b])}"
    )
    answer = send_request_to_LLM_conversation(templates.CONVERSATION_PROMPTS['init_multihop_prompt'].format(
        Role=Role, t_bridge=t_bridge,
        chunk_a=_build_context_string([chunk_a]),
        chunk_b=_build_context_string([chunk_b])
    ))

    for attempt in range(max_retries + 1):
        validation = conversation_validator.validate_init_prompt_all_in_one(answer, combined_docs_for_validator)
        if validation and validation['correct']:
            return answer

        reason = validation['reason'] if validation else "Unknown validation error"
        print(f"  Initial prompt validation failed (attempt {attempt + 1}/{max_retries}): {reason}")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"Validation failed for initial multi-hop (attempt {attempt + 1}).\nAnswer: {answer}\nReason: {reason}\n\n")

        answer = send_request_to_LLM_conversation(templates.CONVERSATION_PROMPTS['rephrase_init_prompt'].format(
            Role=Role, reason=reason
        ))

    print(f"  Initial prompt failed after {max_retries} retries. Giving up.")
    return None


# =========================================================
# Turn 1-n: Follow-up Generation
# =========================================================

def get_follow_up_question(answer, active_chunks, expanding_context = "", max_retries=MAX_RETRIES):
    follow_up_prompt = templates.CONVERSATION_PROMPTS['follow_up_prompt'].format(
        expanding_context = expanding_context,
        RAG_answer=answer
    )

    history = model.get_chat_history()
    response = send_request_to_LLM_conversation(follow_up_prompt)

    current_active_context = _build_active_context_string_for_validator(active_chunks)

    for attempt in range(max_retries + 1):
        validation = conversation_validator.validate_follow_up_question_all_in_one(
            response, history, current_active_context
        )
        if validation and validation['correct']:
            return response

        reason = validation['reason'] if validation else "Unknown validation error"
        print(f"  Follow-up validation failed (attempt {attempt + 1}/{max_retries}): {reason}")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"Follow-up validation failed (attempt {attempt + 1}).\nAnswer: {response}\nReason: {reason}\n\n")

        response = send_request_to_LLM_conversation(templates.CONVERSATION_PROMPTS['rephrase_follow_up_prompt'].format(
            reason=reason
        ))

    print(f"Follow-up failed after {max_retries} retries. Giving up on this turn.")
    return None


# =========================================================
# Main Evaluation Loop
# =========================================================

def generate_conversation():
    db_connector = ChromaConnector('./data/v_eval_filtered/')
    cd = ContextDiscoverer(db_connector=db_connector, llm_model=model, k=4)

    counter = 0
    consecutive_fails = 0

    while counter < max_conversations:
        if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
            print(
                f"\n!!!{MAX_CONSECUTIVE_FAILS} consecutive conversation fails at converation {counter + 1}. Loop is aborted and intermediary results are saved.")
            break

        conv = []
        try:
            res = requests.get(API_URL_THREAD_ID)
            thread_id = res.json().get("thread_id", "")
        except Exception as e:
            print(f"Error getting thread ID: {e}")
            return

        print(f"\n--- Conversation {counter + 1}/{max_conversations} (Consecutive Fails: {consecutive_fails}) ---")

        # =========================================================
        # Context Discovery (Max 3 Tries via MAX_RETRIES)
        # =========================================================
        print("Searching for context bridge via Context Discoverer...")
        discovered_context = None
        for _ in range(MAX_RETRIES):
            discovered_context = cd.discover_valid_context()
            if discovered_context:
                break

        if not discovered_context:
            print(f"No semantic bridges found after {MAX_RETRIES} attempts. Restarting whole conversation.")
            consecutive_fails += 1
            continue

        chunk_a = discovered_context['chunk_a']
        chunk_b = discovered_context['chunk_b']
        t_bridge = discovered_context['t_bridge']

        # =========================================================
        # Turn 0: Initial Question (Max 3 Tries via MAX_RETRIES)
        # =========================================================
        question = get_initial_multihop_prompt_data(chunk_a, chunk_b, t_bridge, max_retries=MAX_RETRIES)

        if not question:
            print(f"Initial multi-hop generation failed after {MAX_RETRIES} retries. Restarting whole conversation.")
            consecutive_fails += 1
            model.reset_chat()
            continue

        active_chunks = {"A": [chunk_a], "B": [chunk_b]}
        turn_idx = 0
        conversation_failed = False

        # =========================================================
        # Turn Generation Loop
        # =========================================================
        while turn_idx < n:
            current_indices_a = [c['chunk_index'] for c in active_chunks["A"]]
            current_indices_b = [c['chunk_index'] for c in active_chunks["B"]]
            print(f"Turn {turn_idx + 1}/{n} | Active Chunks: A{current_indices_a}, B{current_indices_b}")

            # RAG API Call
            try:
                res = requests.post(API_URL, json={"question": question.get('rag_input'), "thread_id": thread_id})
                answer = res.json().get("answer", "")
                context = res.json().get("context", "")

                conv.append({
                    "rag_input": question.get('rag_input'),
                    "question": question.get('question'),
                    "answer": question.get('answer'),
                    "type": question.get('type'),
                    "logic_type": question.get('logic_type', 'none'),
                    "multi_hop_flag": question.get('multi_hop_flag', 0),
                    "bridging_topic": question.get('bridging_topic', None),
                    "rag_answer": answer,
                    "context": context,
                    "turn_index": turn_idx,
                    "ground_truth_chunks": [c['id'] for c in active_chunks["A"]] + [c['id'] for c in active_chunks["B"]]
                })
            except Exception as e:
                print(f"Error during RAG request: {e}")
                conversation_failed = True
                break

            turn_idx += 1

            if turn_idx >= n:
                break

            # =============================================
            # Context Expansion
            # =============================================
            next_turn = turn_idx + 1
            expanding_context_parts = []
            expanding_context = ""
            if next_turn == ALPHA:
                adj_chunk = db_connector.get_adjacent_chunk(
                    article_id=active_chunks["A"][0]['article_id'],
                    current_indices=current_indices_a,
                    total_chunks=active_chunks["A"][0].get('total_chunks', 1),
                    direction=EXPANSION_DIR
                )
                if adj_chunk:
                    active_chunks["A"].append(adj_chunk)
                    expanding_context_parts.append(f"--- Expanded Context A ---\n{adj_chunk['text_snippet']}")
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
                    expanding_context_parts.append(
                        f"--- Expanded Context B ---\n{adj_chunk['text_snippet']}"
                    )
                    print(f" -> [Context Expansion] Expanded B with index {adj_chunk['chunk_index']}")

                if expanding_context_parts:
                    expanding_context = (
                            "To keep the conversation natural and engaging, here is an extended excerpt "
                            "from the source documents. Use this context (if relevant) as additional "
                            "background for the dialogue:\n\n"
                            + "\n\n".join(expanding_context_parts)
                    )

            # =========================================================
            # Follow-Up Generation (Max 3 Tries via MAX_RETRIES)
            # =========================================================
            question = get_follow_up_question(answer, active_chunks, expanding_context=expanding_context,
                                              max_retries=MAX_RETRIES)
            if not question:
                print(
                    f"  ⚠ Follow-Up Generation failed after {MAX_RETRIES} retries at turn {turn_idx + 1}. Restarting whole conversation.")
                conversation_failed = True
                break

        # Wurde die Turn-Loop wegen eines Fehlers (API oder LLM) abgebrochen?
        if conversation_failed:
            consecutive_fails += 1
            model.reset_chat()
            continue

        # =========================================================
        # Save Success (Reset Fail Counter!)
        # =========================================================
        print(f"Conversation {counter + 1} successfully generated!")
        model.reset_chat()

        consecutive_fails = 0

        data_item = {
            "parent_doc_A": chunk_a['article_id'],
            "parent_doc_B": chunk_b['article_id'],
            "role": Role,
            "conversation": conv
        }
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data_item, ensure_ascii=False) + "\n")

        counter += 1

if __name__ == "__main__":
    generate_conversation()