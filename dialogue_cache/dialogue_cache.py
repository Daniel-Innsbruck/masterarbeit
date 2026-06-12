import os
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


class CacheNode:
    """A node in the dialogue cache tree. Stores a response key and the next output bundle."""

    def __init__(self, response_text: str, response_embedding: list, next_bundle: dict):
        self.response_text = response_text  # y_t (target system response)
        self.response_embedding = response_embedding  # embedded y_t
        self.next_bundle = next_bundle  # G_{t+1}
        self.children: list['CacheNode'] = []  # branches for divergent responses at t+1


class DialogueCacheRoot:
    """Root node keyed by a pair of parent document IDs. Stores S_1 and G_1."""

    def __init__(self, parent_doc_a: str, parent_doc_b: str, source_set: dict, initial_bundle: dict):
        self.parent_doc_a = parent_doc_a
        self.parent_doc_b = parent_doc_b
        self.source_set = source_set  # S_1: {chunk_a, chunk_b, t_bridge}
        self.initial_bundle = initial_bundle  # G_1
        self.children: list[CacheNode] = []  # branches after turn 0 response


class DialogueCache:
    """
    Response-Aware Dialogue Caching as specified in the thesis.

    Tree structure:
        Root (keyed by sorted parent doc pair) -> CacheNode per turn
        Each CacheNode stores y_t (response embedding) and G_{t+1}
    """

    def __init__(self, embedding_fn, tau: float = 0.95, safeguard: bool = True,
                 persist_path: Optional[str] = None):
        """
        Args:
            embedding_fn: Callable that takes a string and returns an embedding vector.
            tau: Similarity threshold for cache hits.
            safeguard: If True, cached bundles are re-validated by the CV on hit.
            persist_path: Optional path to persist/load cache as JSON.
        """
        self.embedding_fn = embedding_fn
        self.tau = tau
        self.safeguard = safeguard
        self.persist_path = persist_path
        self.roots: dict[str, DialogueCacheRoot] = {}  # key: "docA||docB" (sorted)

        if persist_path and os.path.exists(persist_path):
            self._load(persist_path)

    # =========================================================
    # Key Generation
    # =========================================================

    @staticmethod
    def _make_root_key(parent_doc_a: str, parent_doc_b: str) -> str:
        """Canonical key from sorted parent doc IDs."""
        return "||".join(sorted([parent_doc_a, parent_doc_b]))

    # =========================================================
    # Cosine Similarity
    # =========================================================

    @staticmethod
    def _cosine_similarity(vec_a, vec_b) -> float:
        a = np.array(vec_a)
        b = np.array(vec_b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    # =========================================================
    # Root Lookup / Registration
    # =========================================================

    def find_root(self, parent_doc_a: str, parent_doc_b: str) -> Optional[DialogueCacheRoot]:
        key = self._make_root_key(parent_doc_a, parent_doc_b)
        return self.roots.get(key)

    def register_root(self, parent_doc_a: str, parent_doc_b: str,
                      source_set: dict, initial_bundle: dict) -> DialogueCacheRoot:
        key = self._make_root_key(parent_doc_a, parent_doc_b)
        root = DialogueCacheRoot(parent_doc_a, parent_doc_b, source_set, initial_bundle)
        self.roots[key] = root
        return root

    # =========================================================
    # Cache Lookup at a Given Node
    # =========================================================

    def lookup(self, children: list[CacheNode], response_text: str,
               response_embedding: list = None) -> tuple[Optional[CacheNode], float, bool]:
        """
        Search for a cache hit among children nodes.

        Returns:
            (best_node, best_similarity, is_exact_match)
            best_node is None if no child exceeds tau.
        """
        if not children:
            return None, 0.0, False

        if response_embedding is None:
            response_embedding = self.embedding_fn(response_text)

        best_node = None
        best_sim = -1.0
        is_exact = False

        for child in children:
            # Check exact lexical match first
            if child.response_text == response_text:
                return child, 1.0, True

            sim = self._cosine_similarity(response_embedding, child.response_embedding)
            if sim > best_sim:
                best_sim = sim
                best_node = child

        if best_sim >= self.tau:
            return best_node, best_sim, False
        return None, best_sim, False

    # =========================================================
    # Insert New Branch
    # =========================================================

    def insert_child(self, children: list[CacheNode], response_text: str,
                     response_embedding: list, next_bundle: dict) -> CacheNode:
        """Appends a new branch node to the given children list."""
        node = CacheNode(response_text, response_embedding, next_bundle)
        children.append(node)
        return node

    # =========================================================
    # High-Level: Process a Turn
    # =========================================================

    def process_turn(self, children: list[CacheNode], response_text: str,
                     conversation_history, validator_fn=None):
        """
        Main entry point per turn. Implements the caching decision from the thesis:

        If Sim(y_t, y') >= tau:
            - exact match with full path exact: bypass CG + CV
            - safeguard enabled: bypass CG, run CV on cached bundle
            - safeguard disabled: bypass CG + CV
        Else:
            return None (caller must run full CG+CV cycle and then call insert_child)

        Args:
            children: The children list of the current node (or root).
            response_text: The target system's response y_t.
            conversation_history: Current dialogue history H_t for CV validation.
            validator_fn: Callable(bundle, history) -> dict with 'correct' key.
                          Required when safeguard=True.

        Returns:
            dict with:
                'hit': bool — whether cache was used
                'bundle': the G_{t+1} bundle (or None if miss)
                'node': the matched CacheNode (or None)
                'similarity': float
                'embedding': the computed embedding of response_text (for reuse on miss)
        """
        response_embedding = self.embedding_fn(response_text)
        node, sim, is_exact = self.lookup(children, response_text, response_embedding)

        if node is None:
            return {
                'hit': False,
                'bundle': None,
                'node': None,
                'similarity': sim,
                'embedding': response_embedding
            }

        # Cache hit
        bundle = node.next_bundle

        # Exact lexical match -> always bypass
        if is_exact:
            return {
                'hit': True,
                'bundle': bundle,
                'node': node,
                'similarity': 1.0,
                'embedding': response_embedding
            }

        # Safeguard mode: validate cached bundle
        if self.safeguard and validator_fn is not None:
            validation = validator_fn(bundle, conversation_history)
            if validation and not validation.get('correct', True):
                # CV rejected -> treat as miss
                return {
                    'hit': False,
                    'bundle': None,
                    'node': None,
                    'similarity': sim,
                    'embedding': response_embedding
                }

        return {
            'hit': True,
            'bundle': bundle,
            'node': node,
            'similarity': sim,
            'embedding': response_embedding
        }

    # =========================================================
    # Persistence
    # =========================================================

    def save(self, path: str = None):
        path = path or self.persist_path
        if not path:
            return

        def serialize_node(node: CacheNode) -> dict:
            return {
                'response_text': node.response_text,
                'response_embedding': node.response_embedding,
                'next_bundle': node.next_bundle,
                'children': [serialize_node(c) for c in node.children]
            }

        data = {}
        for key, root in self.roots.items():
            data[key] = {
                'parent_doc_a': root.parent_doc_a,
                'parent_doc_b': root.parent_doc_b,
                'source_set': root.source_set,
                'initial_bundle': root.initial_bundle,
                'children': [serialize_node(c) for c in root.children]
            }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        def deserialize_node(d: dict) -> CacheNode:
            node = CacheNode(d['response_text'], d['response_embedding'], d['next_bundle'])
            node.children = [deserialize_node(c) for c in d.get('children', [])]
            return node

        for key, root_data in data.items():
            root = DialogueCacheRoot(
                root_data['parent_doc_a'], root_data['parent_doc_b'],
                root_data['source_set'], root_data['initial_bundle']
            )
            root.children = [deserialize_node(c) for c in root_data.get('children', [])]
            self.roots[key] = root

    # =========================================================
    # Stats
    # =========================================================

    def stats(self) -> dict:
        """Returns cache statistics."""
        total_nodes = 0

        def count(children):
            nonlocal total_nodes
            for c in children:
                total_nodes += 1
                count(c.children)

        for root in self.roots.values():
            count(root.children)

        return {
            'num_roots': len(self.roots),
            'total_nodes': total_nodes
        }