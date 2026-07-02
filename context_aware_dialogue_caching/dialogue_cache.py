import os
import json
import uuid
import numpy as np
from google import genai
from google.genai import types


class DialogueCache:
    def __init__(self, base_dir="./data/dialogue_cache", tau=0.2, safeguard=True):
        self.base_dir = base_dir
        self.roots_dir = os.path.join(base_dir, "roots")
        self.trees_dir = os.path.join(base_dir, "trees")
        self.tau = tau
        self.safeguard = safeguard

        # cache-dirs erstellen, falls noch nicht vorhanden
        os.makedirs(self.roots_dir, exist_ok=True)
        os.makedirs(self.trees_dir, exist_ok=True)

        # Gemini Client für die Embeddings der RAG-Antworten (für Sim)
        self.gemini_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

        # Temporärer Speicher für die aktuelle Konversation (wird nur bei Erfolg geschrieben)
        self.current_temp_tree_nodes = []

        # Trackt den aktuellen Pfad im Baum
        self.active_parent_id = None

    def _get_embedding(self, text):
        """Erzeugt ein Embedding für die Ähnlichkeitsberechnung der RAG-Antworten."""
        response = self.gemini_client.models.embed_content(
            model='gemini-embedding-001',
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        return response.embeddings[0].values

    def _cosine_similarity(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    # ==========================================
    # ROOT BOOTSTRAPPING (Initiale Fragen)
    # ==========================================
    def get_random_root(self, current_alpha=None, current_beta=None, current_dir=None):
        """
        Lädt eine zufällige initiale Konfiguration (Root).

        Bietet zwei Modi:
        1. Tuning-Modus (Parameter sind None): Ignoriert die Expansions-Konfiguration,
           und sucht aus allen generierten Initialfragen zufällig eine aus.
        2. CI/CD-Modus (Parameter gesetzt): Filtert strikt nach den Hyperparametern,
           damit die gecachten dialoge zur spezifierten Context Expansion passen
        """
        root_files = [f for f in os.listdir(self.roots_dir) if f.endswith('.json')]
        if not root_files:
            return None

        import random
        random.shuffle(root_files)

        for file in root_files:
            with open(os.path.join(self.roots_dir, file), 'r', encoding='utf-8') as f:
                root_data = json.load(f)

            # MODUS 1: jede root ok (für model-tuning)
            if current_alpha is None and current_beta is None and current_dir is None:
                return root_data

            # MODUS 2: Striktes Filtern für die exakte Context-Expansion Strategie (für regresion tests)
            config = root_data.get("expansion_config", {})
            if (config.get("alpha") == current_alpha and
                    config.get("beta") == current_beta and
                    config.get("direction") == current_dir):
                return root_data

        return None

    def prepare_new_root(self, chunk_a, chunk_b, t_bridge, initial_question, alpha, beta, direction):
        """Bereitet einen neuen Root vor. Wird erst nach erfolgreicher Konversation gespeichert."""
        root_id = str(uuid.uuid4())
        root_data = {
            "root_id": root_id,
            "chunk_a": chunk_a,
            "chunk_b": chunk_b,
            "t_bridge": t_bridge,
            "initial_question": initial_question,
            "expansion_config": {
                "alpha": alpha,
                "beta": beta,
                "direction": direction
            }
        }
        return root_id, root_data

    def save_root(self, root_data):
        """Saves root on disc"""
        filepath = os.path.join(self.roots_dir, f"{root_data['root_id']}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(root_data, f, ensure_ascii=False, indent=2)

    def start_new_path(self, root_id):
        """Wird am Start jeder neuen Konversation aufgerufen."""
        self.active_parent_id = root_id
        self.clear_temp_stage()

    # ==========================================
    # TREE CACHING (CI/CD Follow-Ups)
    # ==========================================
    def find_cache_hit(self, root_id, turn_index, current_rag_answer):
        """searches for semantically similar cache rag response in current branch."""

        tree_path = os.path.join(self.trees_dir, f"{root_id}.json")
        if not os.path.exists(tree_path):
            return None, 0.0, 0

        with open(tree_path, 'r', encoding='utf-8') as f:
            tree_data = json.load(f)

        turn_key = str(turn_index)
        if turn_key not in tree_data:
            return None, 0.0, 0

        valid_nodes = [n for n in tree_data[turn_key] if n.get('parent_id') == self.active_parent_id]

        num_candidates = len(valid_nodes)
        if not valid_nodes:
            return None, 0.0, 0

        current_embedding = self._get_embedding(current_rag_answer)

        best_match = None
        highest_sim = 0.0

        # search for semantically most similar rag response
        for cached_node in valid_nodes:
            sim = self._cosine_similarity(current_embedding, cached_node['rag_answer_embedding'])
            if sim > highest_sim:
                highest_sim = sim
                best_match = cached_node

        if highest_sim >= self.tau:
            print(f"    --> [CACHE HIT] Similarity: {highest_sim:.4f} >= {self.tau}")
            return best_match, highest_sim, num_candidates
        else:
            print(f"    --> [CACHE MISS] Best similarity: {highest_sim:.4f} < {self.tau}")
            return None, highest_sim, num_candidates

    def stage_tree_node(self, turn_index, rag_answer, next_question_bundle):
        """in-memory cache of node in tree that is currently build. Only saved upon success."""
        new_node_id = str(uuid.uuid4())

        self.current_temp_tree_nodes.append({
            "node_id": new_node_id,
            "parent_id": self.active_parent_id,
            "turn_index": str(turn_index),
            "rag_answer": rag_answer,
            "rag_answer_embedding": self._get_embedding(rag_answer),
            "next_question_bundle": next_question_bundle
        })
        self.active_parent_id = new_node_id

    def commit_tree(self, root_id):
        """saves all cached nodes on disc"""
        if not self.current_temp_tree_nodes:
            return

        tree_path = os.path.join(self.trees_dir, f"{root_id}.json")
        tree_data = {}
        if os.path.exists(tree_path):
            with open(tree_path, 'r', encoding='utf-8') as f:
                tree_data = json.load(f)

        for node in self.current_temp_tree_nodes:
            turn_key = node['turn_index']
            if turn_key not in tree_data:
                tree_data[turn_key] = []

            # omit duplicate entries in case the exact same rag response already exists in current child nodes.
            exists = any(n['rag_answer'] == node['rag_answer'] for n in tree_data[turn_key])
            if not exists:
                tree_data[turn_key].append(node)

        with open(tree_path, 'w', encoding='utf-8') as f:
            json.dump(tree_data, f, ensure_ascii=False, indent=2)

        # empty cache
        self.clear_temp_stage()

    def clear_temp_stage(self):
        """empties node cache (i.e. after errors)"""
        self.current_temp_tree_nodes = []