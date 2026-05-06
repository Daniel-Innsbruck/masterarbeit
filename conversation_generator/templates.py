"""
Prompt templates for rag_to_be_tested evaluation
"""

CONVERSATION_PROMPTS = {
    'init_prompt': '''
        You are an expert evaluator of conversational AI systems, specializing in assessing Retrieval-Augmented Generation (rag_to_be_tested) models dynamically. You will conduct a multi-turn dialogue with the rag_to_be_tested system by initiating and maintaining a conversation. Your questions and follow-ups should be thoughtfully based on the provided documents to thoroughly evaluate the system's understanding and retrieval capabilities.
        
        **Your Role:** {Role}

        **Information Document: {Document}**

        - Adopt the persona and tone consistent with your assigned role.
        - Use language and style fitting your character—be direct and natural; excessive politeness or introductions are unnecessary unless dictated by your role.
        - Remember, you are communicating with an LLM; your questions should be clear, concise, and fully self-contained.
        - Base your questions and follow-ups on the provided documents, starting a natural conversation that probes the rag_to_be_tested system's knowledge and reasoning.
        - Be phrased as a **context-aware follow-up** that assumes shared knowledge from earlier turns (do **not** make it standalone).
        Provide the first turn (fully self-contained) of the conversation with the following format:
        Only provide the **first turn**. Do **not** generate multiple dialogue turns.
        **Output Format (exact JSON):**
        "rag_input": The initial input question given to the rag_to_be_tested system. It must be clear, context-rich, and include all details necessary to answer without ambiguity or additional context. Do not mention that a document was provided.
        "Question": Exactly the same as the rag_to_be_tested input. In the first turn, these should be identical.
        "Answer": The correct, ground-truth answer to the question.
    ''',
    'follow_up_prompt': '''
        This was the provided answer from the rag_to_be_tested system:

        **{RAG_answer}**
        
        Your task is to write the **next turn in the conversation** — a natural follow-up question that assumes a shared conversational context, but **does not refer to any documents, sources, or retrieval process.**

        You are evaluating how well the system can maintain internal consistency, resolve ambiguities, and reason based on prior conversation turns. Do **not** break character or refer to any underlying documents.
        Choose one of the following Types:
            - **Follow-up:** Builds on a previous answer (e.g., “What about…”, “How about…”).
            - **Clarification:** Seeks to resolve ambiguity (e.g., “You mean…?”, “Does that mean…?”).
            - **Correction:** Rectifies a misunderstanding or error (e.g., “No, that’s not what I meant.”).
            - **Comparative:** Requests comparison between two or more concepts (e.g., “How does this compare to…?”).
        **Output Format (exact JSON):**
        "rag_input": A concise, context-aware follow-up that continues the conversation naturally. It should sound like a human asking a follow-up question in a discussion, **not like a meta-evaluation.**
        "Question": A fully self-contained version of the rag_to_be_tested input, rewritten with all necessary context for standalone understanding.
        "Answer": The expected, correct answer based on the document(s).
        "Type": The type of question you are asking, choose from the following: Follow-up, Clarification, Correction, Comparative.
        ''',

    'rephrase_init_prompt': '''
        This was incorrect. Reason: {reason}

        Generate a completly new QA Set again, following this exact format without any additional text:

        **Output Format (exact JSON):**
        "rag_input": The initial input question given to the rag_to_be_tested system. It must be clear, context-rich, and include all details necessary to answer without ambiguity or additional context. Do not mention that a document was provided.
        "Question": Exactly the same as the rag_to_be_tested input. In the first turn, these should be identical.
        "Answer": The correct, ground-truth answer to the question.
        ''',
    'rephrase_follow_up_prompt': '''
        This was incorrect. Reason: {reason}

        Do it again, following this exact format  without any additional text:
         **Output Format (exact JSON):**
        "rag_input": A concise, context-aware follow-up that continues the conversation naturally. It should sound like a human asking a follow-up question in a discussion, **not like a meta-evaluation.**
        "Question": A fully self-contained version of the rag_to_be_tested input, rewritten with all necessary context for standalone understanding.
        "Answer": The expected, correct answer based on the document(s).
        "Type": The type of question you are asking, choose from the following: Follow-up, Clarification, Correction, Comparative.
    ''',
    'init_multihop_prompt': '''
        You are a SOTA evaluation assistant for RAG systems. Your goal is to create complex
        Multi-Hop questions.
        
        TASK:
        Given two context snippets from DIFFERENT news articles, identify their latent connection and formulate a question that CANNOT be answered by one snippet alone.
        
        QUESTION TYPE: {query_type}
        
        CONTEXT A: {chunk_a}
        CONTEXT B: {chunk_b}
        
        CONSTRAINTS:
        1. The answer MUST require facts from both Snippet A AND Snippet B.
        2. Focus on a "Thematic Bridge" (e.g. Causal link, shared event, or temporal sequence).
        3. The question must be natural (no mention of 'Snippet A').
        4. Provide the answer clearly based on the provided snippets.
        
        OUTPUT FORMAT (JSON):
        {{
            "question": "The multi-hop question",
            "answer": "The combined answer",
            "thematic_link": "Short explanation of the link between A and B",
            "logic_type": "{query_type}"
        }}
        
        REQUIRED JSON STRUCTURE:
        {{
            "question": "...",
            "answer": "...",
            "thematic_link": "...",
            "logic_type": "{query_type}"
        }}
    ''',
}