import requests
import sys
import os

# Füge das übergeordnete Verzeichnis zum Python-Pfad hinzu
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import gc
import time
import json

from context_aware_dialogue_caching.dialogue_cache import DialogueCache


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

## Cache-Steuerung

BUILD_CACHE = False          # Soll Cache befüllt werden?
USE_CACHE = True            # Soll der Cache gelesen werden?
CACHE_MODE = "all"
CACHE_TAU = 0.95
CACHE_SAFEGUARD = True

# Fixe Liste von UIDs für das kontrollierte Experiment / Phase 1 Validation
FIXED_ROOT_IDS = [
    "392c9cc7-f4c1-4529-94fc-2a1f59ced3cd"
]

## Fail-Fast parameters
MAX_RETRIES = 3  # Max retries per turn (independent of n)
MAX_CONSECUTIVE_FAILS = 5 # max number of complete conversation restarts before qa-gernation ends early (avoid infinite loop)

# dialog configs
n = 5             # Target number of turns per conversation
max_conversations = 10 # number conversations'

# Logging & Output
output_file = "./data/cachetree_test_conversation_data_" + model_name + "_turns_" + str(n) + "_conversations_" + str(max_conversations)+ ".jsonl"
log_file = "./data/cachetree_test_conversation_data_" + model_name + "_turns_" + str(n) + "_conversations_" + str(max_conversations)+ ".log"
metrics_file = "./data/cache_metrics_" + model_name + "_turns_" + str(n) + ".jsonl"

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

    dialogue_cache = DialogueCache() if USE_CACHE or BUILD_CACHE else None

    counter = 0
    consecutive_fails = 0

    iterations = max_conversations # len(FIXED_ROOT_IDS) if (USE_CACHE and FIXED_ROOT_IDS) else max_conversations

    while counter < iterations:
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

        if BUILD_CACHE or USE_CACHE:
            dialogue_cache.clear_temp_stage()

        root_data = None
        is_new_root = False
        root_id = None
        # =========================================================
        # Turn 0: Context Discovery & Initial Question
        # =========================================================
        if USE_CACHE and FIXED_ROOT_IDS:
            root_id = FIXED_ROOT_IDS[0] #FIXED_ROOT_IDS[counter]
            # Versuche das spezifische Root-File von der Festplatte zu laden
            root_path = os.path.join(dialogue_cache.roots_dir, f"{root_id}.json")
            if os.path.exists(root_path):
                with open(root_path, 'r', encoding='utf-8') as f:
                    root_data = json.load(f)
                print(f"\n--- Run {counter + 1} | Using Fixed Root: {root_id} ---")
            else:
                print(f"\n[WARN] Fixed Root {root_id} not found on disk. Falling back to dynamic generation.")
        if root_data:
            chunk_a = root_data['chunk_a']
            chunk_b = root_data['chunk_b']
            t_bridge = root_data['t_bridge']
            question = root_data['initial_question']
        else:
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
            question = get_initial_multihop_prompt_data(chunk_a, chunk_b, t_bridge, max_retries=MAX_RETRIES)

            # =========================================================
            # Turn 0: Initial Question (Max 3 Tries via MAX_RETRIES)
            # =========================================================

            if not question:
                print(f"Initial multi-hop generation failed after {MAX_RETRIES} retries. Restarting whole conversation.")
                consecutive_fails += 1
                model.reset_chat()
                continue
            if BUILD_CACHE:
                root_id, root_data = dialogue_cache.prepare_new_root(
                    chunk_a, chunk_b, t_bridge, question, ALPHA, BETA, EXPANSION_DIR
                )
                is_new_root = True

        if dialogue_cache:
            dialogue_cache.start_new_path(root_id)

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
            # CACHE Schritt 1: Lesen
            # =========================================================
            next_question = None

            #ToDo entfernen(nur für experiment)
            sim_score = 0.0
            num_candidates = 0
            cache_accepted = False
            validator_passed = False # trackt urteil des validators

            if USE_CACHE and CACHE_MODE == "all" and not is_new_root:
                cached_bundle, sim_score, num_candidates = dialogue_cache.find_cache_hit(root_id, turn_idx - 1, answer)

                if cached_bundle:
                    proposed_question = cached_bundle['next_question_bundle']
                    if CACHE_SAFEGUARD:
                        print("Executing Safeguard Validation...")
                        current_active_context = _build_active_context_string_for_validator(active_chunks)
                        history = model.get_chat_history()
                        validation = conversation_validator.validate_follow_up_question_all_in_one(
                            proposed_question, history, current_active_context
                        )
                        if validation and validation['correct']:
                            print("Safeguard passed. Reusing cached follow-up.")
                            validator_passed = True

                            next_question = proposed_question
                            cache_accepted = True
                        else:
                            print(f"Safeguard failed. Regenerating.")
                            validator_passed = False
                    else:
                        print("Safeguard disabled. Reusing cached follow-up.")
                        next_question = proposed_question
                        cache_accepted = True
                        validator_passed = False

                    # HISTORY INJECTION BEI CACHE HIT
                    if cache_accepted:
                        dialogue_cache.active_parent_id = cached_bundle['node_id']
                        print("Injecting cached turn into model memory to prevent amnesia...")
                        simulated_prompt = templates.CONVERSATION_PROMPTS['follow_up_prompt'].format(
                            expanding_context=expanding_context, RAG_answer=answer
                        )
                        simulated_response = json.dumps(next_question, ensure_ascii=False)
                        model.inject_history(simulated_prompt, simulated_response)

            # =========================================================
            # STRUKTURIERTES EXPERIMENT-LOGGING (Für deine Plots)
            # =========================================================
            # Wir loggen jeden Turn, um zu sehen, ob überhaupt Cache da war (candidates > 0)
            metrics_entry = {
                "conversation_id": counter + 1,  # Die wievielte Konversation
                "turn_index": turn_idx,  # Die Konversationsebene (Turn)
                "cache_available": num_candidates > 0,  # Cache verfügbar ja/nein
                "candidates_count": num_candidates,  # Wie viele standen zur Auswahl
                "highest_sim": round(sim_score, 4) if num_candidates > 0 else 0.0,
                "validator_passed": validator_passed if num_candidates > 0 else None,  # Urteil des CV
                "cache_accepted_and_used": cache_accepted  # Wurde es im Lauf wirklich genutzt?
            }

            with open(metrics_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(metrics_entry) + "\n")

            # Alt-Logging für die Konsole (Kompakt gehalten)
            print(
                f"    [CACHE LOG] Turn {turn_idx} | Candidates: {num_candidates} | Sim: {sim_score:.4f} | Accepted: {cache_accepted}")
            # =========================================================
            # CACHE SCHRITT 2: Generieren (Falls Cache leer/abgelehnt) + Follow-Up generierung (max. 3 fails)
            # =========================================================
            if not cache_accepted:
                next_question = get_follow_up_question(answer, active_chunks, expanding_context=expanding_context,
                                                       max_retries=MAX_RETRIES)

                if not next_question:
                    print(f"Follow-Up Generation failed after {MAX_RETRIES} retries at turn {turn_idx + 1}. Restarting whole conversation.")
                    conversation_failed = True
                    break

                # =========================================================
                # CACHE SCHRITT 3: Schreiben (Nur wenn FRISCH generiert UND BUILD_CACHE)
                # =========================================================
                if BUILD_CACHE and CACHE_MODE == "all":
                    dialogue_cache.stage_tree_node(turn_idx - 1, answer, next_question)

            question = next_question

        # =========================================================
        # Abschluss & Speichern
        # =========================================================
        if conversation_failed:
            consecutive_fails += 1
            model.reset_chat()
            if dialogue_cache:
                dialogue_cache.clear_temp_stage()
            continue

        print(f"Conversation {counter + 1} successfully generated!")
        model.reset_chat()

        consecutive_fails = 0

        if BUILD_CACHE:
            if is_new_root:
                dialogue_cache.save_root(root_data)
            if CACHE_MODE == "all":
                dialogue_cache.commit_tree(root_id)
            print(f"Caching successful for Root: {root_id}")

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