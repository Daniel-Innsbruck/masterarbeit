# Generated with Gemini 3.1 Pro, validated and checked by Daniel Hillebrand

import json
from .templates import CD_PROMPTS

class ContextDiscoverer:
    """
    Orchestrates the two-stage multi-hop context discovery process by utilizing filtered
    semantic search and subsequent validation through a Large Language Model.
    """

    def __init__(self, db_connector, llm_model, k=4):
        """
        Initializes the ContextDiscoverer with database and LLM connections.

        Args:
            db_connector: An instance of ChromaConnector.
            llm_model: An instance of the GEMINI wrapper class.
            k: The number of semantic neighbors to retrieve during the discovery phase.
        """
        self.db = db_connector
        self.llm = llm_model
        self.k = k

    def discover_valid_context(self, max_start_retries=10):
        """
        Executes the filtered semantic search and LLM validation to find a valid multi-hop context.

        Args:
            max_start_retries: The maximum number of different starting chunks to attempt.

        Returns:
            A dictionary containing 'chunk_a', 'chunk_b', and 't_bridge' if successful, or None otherwise.
        """
        for attempt in range(max_start_retries):
            raw_chunk_a = self.db.get_random_chunk()
            if not raw_chunk_a:
                return None
            db_result = self.db.collection.get(ids=[raw_chunk_a['id']])

            if not db_result['metadatas'] or not db_result['metadatas'][0]:
                print(f"[CD] Warning: Chunk {raw_chunk_a['id']} has no metadata in V_eval DB. Skipping")
                continue

            meta = db_result['metadatas'][0]

            if 'chunk_index' not in meta or 'total_chunks' not in meta:
                print(
                    f"[CD] Warning: 'chunk_index' or 'total_chunks' missing from article {meta.get('article_id', 'Unknown')}. Discrading chunk.")
                continue

            chunk_a = {
                'id': db_result['ids'][0],
                'text_snippet': db_result['documents'][0],
                'article_id': meta['article_id'],
                'chunk_index': meta['chunk_index'],
                'total_chunks': meta['total_chunks']
            }
            neighbors = self._get_top_k_neighbors(
                query_text=chunk_a['text_snippet'],
                exclude_article_id=chunk_a['article_id']
            )

            if not neighbors:
                continue

            for chunk_b in neighbors:
                validation_result = self._validate_with_llm(chunk_a, chunk_b)

                if validation_result and validation_result.get('score') == 1:
                    return {
                        'chunk_a': chunk_a,
                        'chunk_b': chunk_b,
                        't_bridge': validation_result.get('T_bridge', [])
                    }

        print("[CD] Error: Could not find valid contextual bridges after maximum retries.")
        return None

    def _get_top_k_neighbors(self, query_text, exclude_article_id):
        """
        Retrieves the top k semantic neighbors from the vector database, strictly filtering
        out chunks originating from the same parent document.

        Args:
            query_text: The text snippet of the initial chunk.
            exclude_article_id: The article ID of the initial chunk to exclude from results.

        Returns:
            A list of neighbor chunk dictionaries.
        """
        query_vector = self.db.get_gemini_embedding(query_text)

        results = self.db.collection.query(
            query_embeddings=[query_vector],
            n_results=self.k,
            where={"article_id": {"$ne": exclude_article_id}}
        )

        if not results['ids'] or not results['ids'][0]:
            return []

        neighbors = []
        for i in range(len(results['ids'][0])):
            neighbors.append({
                'id': results['ids'][0][i],
                'text_snippet': results['documents'][0][i],
                'article_id': results['metadatas'][0][i]['article_id'],
                'chunk_index': results['metadatas'][0][i]['chunk_index'],
                'total_chunks': results['metadatas'][0][i].get('total_chunks', 1)
            })
        return neighbors

    def _validate_with_llm(self, chunk_a, chunk_b):
        """
        Validates the thematic overlap of two chunks using the LLM.

        Args:
            chunk_a: The first chunk dictionary.
            chunk_b: The candidate bridging chunk dictionary.

        Returns:
            A parsed JSON dictionary containing the validation score and bridging topics,
            or None on parsing failure.
        """
        prompt = CD_PROMPTS['validate_bridge'].format(
            chunk_a=chunk_a['text_snippet'],
            chunk_b=chunk_b['text_snippet']
        )

        try:
            response_text = self.llm.prompt(prompt)
            clean_text = response_text.replace('```json', '').replace('```', '').strip()
            return json.loads(clean_text)
        except Exception as e:
            print(f"[CD] LLM Validation Error: {e}")
            return None