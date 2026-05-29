CD_PROMPTS = {
    'validate_bridge': """
You are an expert evaluator for a multi-hop reasoning dataset.
Your task is to determine if two text snippets share a strong enough thematic connection to formulate a complex, multi-hop question that requires BOTH snippets to answer.

Snippet A: {chunk_a}
Snippet B: {chunk_b}

Evaluate the connection. If there is a valid connection (e.g., causal link, shared event, temporal sequence, or comparative aspect), assign a score of 1 and suggest a list of 'T_bridge' (the bridging topics).
If they are completely unrelated or the connection is too weak/trivial to form a good question, assign a score of 0 and leave 'T_bridge' empty.

OUTPUT FORMAT (JSON ONLY):
{{
    "score": 1,
    "T_bridge": ["Topic 1", "Topic 2"]
}}
"""
}