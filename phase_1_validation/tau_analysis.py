import json
import time
import requests
import numpy as np
from collections import defaultdict

from dialogue_cache.dialogue_cache import DialogueCache
from utils.chroma_connector import ChromaConnector
import Models.gemini as gemini
import Models.chat_gpt as chat_gpt
import templates as templates
import utils.parser as parser
import conversation_validator.conversation_validator as conversation_validator

# =========================================================
# Config
# =========================================================
API_URL = "http://localhost:8000/rag"
API_URL_THREAD_ID = "http://localhost:8000/getThreadID"

model_name = 'gemini-3.1-flash-lite'
model = gemini.GEMINI(model_name)
parser = parser.LLMResponseParser()

JSONL_PATH = "../data/conversifation_data_gemini-3.1-flash-lite_turns_5_conversation_<built-in function max>.jsonl"
RESULTS_PATH = "../data/tau_analysis_results.json"

TAU_VALUES = [0.80, 0.85, 0.90, 0.95, 0.975, 1.0]
N_REPEATS = 20
N_ROOTS = 3
n = 5
MAX_RETRIES = 3
ALPHA = 2
BETA = 3
EXPANSION_DIR = "below"

Role = "You are a highly attentive conversationalist who asks context-aware questions. Your questions should build naturally on previous exchanges, using referring expressions like 'this', 'that', or 'it' to maintain coherence and continuity."

def get_embedding_with_retry(db_connector, text, max_retries=5):
    for attempt in range(max_retries):
        try:
            return db_connector.get_gemini_embedding(text)
        except Exception as e:
            if '429' in str(e) or '503' in str(e):
                wait = 2 ** attempt * 10
                print(f"    Embedding API error, retrying in {wait}s... ({e})")
                time.sleep(wait)
            else:
                raise
    raise Exception("Max retries exceeded for embedding")


# =========================================================
# Load roots from JSONL
# =========================================================
def load_roots_from_jsonl(jsonl_path, db_connector, cache):
    with open(jsonl_path, 'r') as f:
        conversations = [json.loads(line) for line in f]

    # Only load first N_ROOTS conversations
    conversations = conversations[:N_ROOTS]

    for conv in conversations:
        parent_doc_a = conv['parent_doc_A']
        parent_doc_b = conv['parent_doc_B']
        turns = conv['conversation']

        gt_chunks = turns[0].get('ground_truth_chunks', [])
        chunk_a_ref = gt_chunks[0] if len(gt_chunks) > 0 else None
        chunk_b_ref = gt_chunks[1] if len(gt_chunks) > 1 else None

        def parse_chunk_ref(chunk_ref, article_id):
            if chunk_ref and '#chunk_' in chunk_ref:
                idx = int(chunk_ref.split('#chunk_')[1])
                chunk = db_connector.get_chunk_by_id(article_id, idx)
                if chunk:
                    return chunk
            return db_connector.get_chunk_by_id(article_id, 0)

        chunk_a = parse_chunk_ref(chunk_a_ref, parent_doc_a)
        chunk_b = parse_chunk_ref(chunk_b_ref, parent_doc_b)

        if not chunk_a or not chunk_b:
            print(f"WARNING: Could not resolve chunks for {parent_doc_a} / {parent_doc_b}")
            continue

        initial_bundle = {
            'rag_input': turns[0]['rag_input'],
            'question': turns[0]['question'],
            'logic_type': turns[0].get('logic_type'),
            'multi_hop_flag': turns[0].get('multi_hop_flag'),
            'bridging_topic': turns[0].get('bridging_topic')
        }

        root = cache.register_root(
            parent_doc_a, parent_doc_b,
            {'chunk_a': chunk_a, 'chunk_b': chunk_b},
            initial_bundle
        )

        current_children = root.children

        for i in range(len(turns) - 1):
            answer_text = turns[i]['rag_answer']
            answer_embedding = get_embedding_with_retry(db_connector, answer_text)
            time.sleep(1)

            next_bundle = {
                'rag_input': turns[i + 1]['rag_input'],
                'question': turns[i + 1]['question'],
                'logic_type': turns[i + 1].get('logic_type'),
                'multi_hop_flag': turns[i + 1].get('multi_hop_flag'),
                'bridging_topic': turns[i + 1].get('bridging_topic')
            }

            node = cache.insert_child(
                children=current_children,
                response_text=answer_text,
                response_embedding=answer_embedding,
                next_bundle=next_bundle
            )
            current_children = node.children

        print(f"Loaded dialog: {parent_doc_a} || {parent_doc_b} ({len(turns)} turns)")

    print(f"Total roots: {len(cache.roots)}")


# =========================================================
# Helpers
# =========================================================

def _build_context_string(chunk_list):
    return "\n\n".join([c['text_snippet'] for c in chunk_list])


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
                time.sleep(60)
            else:
                print(f"Error: {e}")
                return None
    return response


def get_follow_up_question(answer, active_chunks):
    follow_up_prompt = templates.CONVERSATION_PROMPTS['follow_up_prompt'].format(
        Role=Role,
        context_a=_build_context_string(active_chunks['A']),
        context_b=_build_context_string(active_chunks['B']),
        RAG_answer=answer
    )
    history = model.get_chat_history()
    response = send_request_to_LLM_conversation(follow_up_prompt)

    for attempt in range(MAX_RETRIES):
        validation = conversation_validator.validate_follow_up_question_all_in_one(response, history)
        if validation and validation['correct']:
            return response
        reason = validation['reason'] if validation else "Unknown"
        response = send_request_to_LLM_conversation(
            templates.CONVERSATION_PROMPTS['rephrase_follow_up_prompt'].format(Role=Role, reason=reason)
        )

    validation = conversation_validator.validate_follow_up_question_all_in_one(response, history)
    if validation and validation['correct']:
        return response
    return None


# =========================================================
# Single conversation run
# =========================================================

def run_single_conversation(root_data, cache, db_connector, tau):
    chunk_a = root_data['chunk_a']
    chunk_b = root_data['chunk_b']
    parent_doc_a = root_data['parent_doc_a']
    parent_doc_b = root_data['parent_doc_b']
    root_id = f"{parent_doc_a}__{parent_doc_b}"

    cache_root = cache.find_root(parent_doc_a, parent_doc_b)
    if not cache_root:
        cache_root = cache.register_root(
            parent_doc_a, parent_doc_b,
            {'chunk_a': chunk_a, 'chunk_b': chunk_b},
            root_data['initial_bundle']
        )

    try:
        res = requests.get(API_URL_THREAD_ID)
        thread_id = res.json().get("thread_id", "")
    except Exception as e:
        print(f"  Error getting thread ID: {e}")
        return []

    model.reset_chat()
    active_chunks = {"A": [chunk_a], "B": [chunk_b]}
    question = root_data['initial_bundle']
    current_children = cache_root.children
    turn_records = []

    for turn_idx in range(n):
        try:
            res = requests.post(API_URL, json={
                "question": question['rag_input'],
                "thread_id": thread_id
            })
            answer = res.json().get("answer", "")
        except Exception as e:
            print(f"    RAG error at turn {turn_idx}: {e}")
            break

        if turn_idx >= n - 1:
            break

        next_turn = turn_idx + 2
        if next_turn == ALPHA:
            adj = db_connector.get_adjacent_chunk(
                chunk_a['article_id'],
                [c['chunk_index'] for c in active_chunks["A"]],
                chunk_a.get('total_chunks', 1), EXPANSION_DIR
            )
            if adj:
                active_chunks["A"].append(adj)

        if next_turn == BETA:
            adj = db_connector.get_adjacent_chunk(
                chunk_b['article_id'],
                [c['chunk_index'] for c in active_chunks["B"]],
                chunk_b.get('total_chunks', 1), EXPANSION_DIR
            )
            if adj:
                active_chunks["B"].append(adj)

        response_embedding = get_embedding_with_retry(db_connector, answer)
        cache_hit = False
        cache_accepted_by_cv = None
        similarity = None

        if current_children:
            best_sim = -1
            best_child = None
            for child in current_children:
                sim = np.dot(response_embedding, child.response_embedding) / (
                    np.linalg.norm(response_embedding) * np.linalg.norm(child.response_embedding)
                )
                if sim > best_sim:
                    best_sim = sim
                    best_child = child

            similarity = float(best_sim)

            if best_sim >= tau:
                cv_result = conversation_validator.validate_follow_up_question_all_in_one(
                    best_child.next_bundle, model.get_chat_history()
                )
                cache_accepted_by_cv = cv_result and cv_result.get('correct', False)

                if cache_accepted_by_cv:
                    cache_hit = True
                    question = best_child.next_bundle
                    current_children = best_child.children
                else:
                    cache_hit = False
            else:
                cv_result = conversation_validator.validate_follow_up_question_all_in_one(
                    best_child.next_bundle, model.get_chat_history()
                )
                cache_accepted_by_cv = cv_result and cv_result.get('correct', False)

        turn_records.append({
            'root_id': root_id,
            'turn': turn_idx + 1,
            'tau': tau,
            'similarity': similarity,
            'cv_accepted': cache_accepted_by_cv,
            'cache_hit': cache_hit
        })

        status = "CACHE" if cache_hit else "FRESH"
        sim_str = f"{similarity:.4f}" if similarity is not None else "N/A"
        cv_str = "✓" if cache_accepted_by_cv else ("✗" if cache_accepted_by_cv is not None else "-")
        print(f"    Turn {turn_idx + 1}: sim={sim_str}, cv={cv_str}, → {status}")

        if not cache_hit:
            question = get_follow_up_question(answer, active_chunks)
            if question is None:
                break

            new_node = cache.insert_child(
                children=current_children if current_children else [],
                response_text=answer,
                response_embedding=response_embedding,
                next_bundle=question
            )
            current_children = new_node.children

    model.reset_chat()
    return turn_records


# =========================================================
# Main
# =========================================================
def run_tau_analysis():
    db_connector = ChromaConnector('../data/v_eval_filtered/')
    all_records = []

    print("Pre-computing dialog embeddings...")
    preload_cache = DialogueCache(
        embedding_fn=db_connector.get_gemini_embedding,
        tau=0.0, safeguard=False
    )
    load_roots_from_jsonl(JSONL_PATH, db_connector, preload_cache)
    preload_cache.save("../data/preloaded_cache.json")

    for tau in TAU_VALUES:
        print(f"\n{'=' * 60}")
        print(f"  TAU = {tau}")
        print(f"{'=' * 60}")

        cache = DialogueCache(
            embedding_fn=db_connector.get_gemini_embedding,
            tau=tau,
            safeguard=True,
            persist_path="../data/preloaded_cache.json"
        )

        for key, root in cache.roots.items():
            print(f"\n  Root: {key}")
            root_data = {
                'parent_doc_a': root.parent_doc_a,
                'parent_doc_b': root.parent_doc_b,
                'chunk_a': root.source_set['chunk_a'],
                'chunk_b': root.source_set['chunk_b'],
                'initial_bundle': root.initial_bundle
            }

            for repeat in range(N_REPEATS):
                print(f"    Repeat {repeat + 1}/{N_REPEATS}")
                records = run_single_conversation(root_data, cache, db_connector, tau)
                all_records.extend(records)

                # Incremental save after each repeat
                with open(RESULTS_PATH, 'w') as f:
                    json.dump(all_records, f, indent=2)

        tau_cache_path = f"../data/cache_tau_{tau}.json"
        cache.save(tau_cache_path)

    print(f"\nTotal records: {len(all_records)}")
    analyze_results(all_records)


# =========================================================
# Analysis
# =========================================================

def analyze_results(records=None):
    if records is None:
        with open(RESULTS_PATH, 'r') as f:
            records = json.load(f)

    print(f"\n{'tau':>6} | {'TP':>5} | {'FP':>5} | {'FN':>5} | {'TN':>5} | {'Prec':>6} | {'HitRate':>7} | {'FP%':>5}")
    print("-" * 65)

    results = []
    for tau in TAU_VALUES:
        tau_recs = [r for r in records if r['tau'] == tau and r['similarity'] is not None]
        tp = sum(1 for r in tau_recs if r['similarity'] >= tau and r['cv_accepted'])
        fp = sum(1 for r in tau_recs if r['similarity'] >= tau and not r['cv_accepted'])
        fn = sum(1 for r in tau_recs if r['similarity'] < tau and r['cv_accepted'])
        tn = sum(1 for r in tau_recs if r['similarity'] < tau and not r['cv_accepted'])

        total = tp + fp + fn + tn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        hit_rate = (tp + fp) / total if total > 0 else 0.0
        fp_rate = fp / total if total > 0 else 0.0

        results.append({
            'tau': round(float(tau), 2),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': round(precision, 4),
            'hit_rate': round(hit_rate, 4),
            'fp_rate': round(fp_rate, 4)
        })

        print(f"{tau:>6.3f} | {tp:>5} | {fp:>5} | {fn:>5} | {tn:>5} | {precision:>6.3f} | {hit_rate:>7.3f} | {fp_rate:>5.3f}")

    print("\n\n=== Per-Turn Cache Hit Likelihood ===")
    for tau in TAU_VALUES:
        tau_recs = [r for r in records if r['tau'] == tau and r['similarity'] is not None]
        print(f"\ntau = {tau}")
        turns = sorted(set(r['turn'] for r in tau_recs))
        for t in turns:
            turn_recs = [r for r in tau_recs if r['turn'] == t]
            safe_hits = sum(1 for r in turn_recs if r['cache_hit'])
            total = len(turn_recs)
            rate = safe_hits / total if total > 0 else 0
            print(f"  Turn {t}: {safe_hits}/{total} cache hits ({rate:.1%})")

    with open(RESULTS_PATH.replace('.json', '_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_tau_analysis()