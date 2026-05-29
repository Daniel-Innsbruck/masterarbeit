"""
Prompt templates for rag_to_be_tested evaluation
"""

CONVERSATION_PROMPTS = {
    'init_multihop_prompt': '''
        You are an expert evaluator of conversational AI systems, specializing in assessing Retrieval-Augmented Generation (RAG) models dynamically. You will conduct a multi-turn dialogue with the RAG system by initiating and maintaining a conversation. 
        Your goal is to create a complex Multi-Hop question for the first turn of a conversation.

        **Your Role:** {Role}

        TASK:
        Given two context snippets from DIFFERENT source documents and a set of valid Bridging Topics, formulate a highly precise, incisive question that STRICTLY requires synthesizing information from BOTH snippets.

        VALID BRIDGING TOPICS (T_bridge): {t_bridge}

        CONTEXT A: 
        {chunk_a}

        CONTEXT B: 
        {chunk_b}

        CRITICAL CONSTRAINTS:
        1. ADOPT PERSONA: Use language and style fitting your assigned character. Be direct and natural; excessive politeness or introductions are unnecessary unless dictated by your role.
        2. STRICT MULTI-HOP: It must be IMPOSSIBLE to answer the question using only Context A or only Context B. The answer must require a logical synthesis of facts from both.
        3. NO COMPOUND QUESTIONS: Do NOT simply ask two separate questions joined by 'and'. The question must be a single, cohesive inquiry built around ONE OR MORE topics from the 'T_bridge' list.
        4. STRICT SCOPE: The question must be answerable EXACTLY and ENTIRELY within the boundaries of the provided snippets.
        5. COMPREHENSIVE ANSWER: The ground-truth "answer" captures and summarizes the core synthesized facts from both contexts.
        6. NATURAL FLOW: The question must sound natural (do not mention 'Context A' or 'Context B').
        7. LOGIC TYPE: Autonomously determine the logical reasoning required to answer your question ('inference', 'comparison' or 'temporal').
        8. Because this is the first turn, "rag_input" and "question" MUST be identical. "multi_hop_flag" MUST be 1, and "type" MUST be "Initial".

        OUTPUT FORMAT (exact JSON):
        {{
            "rag_input": "The initial natural question sent to the RAG system, phrased exactly in your assigned Role",
            "question": "Exactly the same as rag_input",
            "answer": "The combined, comprehensive ground-truth answer",
            "type": "Initial",
            "logic_type": "The logical reasoning type you chose (inference, comparison, or temporal)",
            "multi_hop_flag": 1,
            "bridging_topic": "A single string summarizing the bridging topic(s) you actually used"
        }}
    ''',

    'rephrase_init_prompt': '''
        This was incorrect. Reason: {reason}

        You MUST follow this exact JSON format without any additional text:

        OUTPUT FORMAT (exact JSON):
        {{
            "rag_input": "The initial natural question sent to the RAG system, phrased in your assigned Role",
            "question": "Exactly the same as rag_input",
            "answer": "The combined, comprehensive ground-truth answer",
            "type": "Initial",
            "logic_type": "The logical reasoning type you chose (inference, comparison, or temporal)",
            "multi_hop_flag": 1,
            "bridging_topic": "A single string summarizing the bridging topic(s) you used"
        }}
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
        
        Since the evaluation splits single-hop from multi-hop questions, you must specify the follow-up category:
            - **multi_hop_flag = 1**: Use this ONLY if your new question strictly requires combining distinct facts from BOTH Context A and Context B to form a single answer. 
                * STRICT RULE: Do NOT create compound questions (e.g., "What is X, and what is Y?"). The question must be singular, but require dual-context synthesis to be answered.
            - **multi_hop_flag = 0**: Use this if the question can be fully answered using only ONE context, or just the previous conversation history. 
            - If "multi_hop_flag" is 1, specify the "logic_type" (inference, comparison, or temporal). If "multi_hop_flag" is 0, set "logic_type" to "none".
        
        OUTPUT FORMAT (exact JSON):
        {{
            "rag_input": "A concise, context-aware follow-up in your persona, containing pronouns like 'this' or 'it' to continue the conversation naturally.",
            "question": "A fully self-contained version of the rag_input, rewritten with all necessary context for standalone understanding.",
            "answer": "The expected, comprehensive ground-truth answer.",
            "type": "Follow-up / Clarification / Correction / Comparative",
            "logic_type": "inference / comparison / temporal / none",
            "multi_hop_flag": 0 or 1
        }}
    ''',

    'rephrase_follow_up_prompt': '''
        This was incorrect. Reason: {reason}

        Do it again, following this exact format without any additional text:
        OUTPUT FORMAT (exact JSON):
        {{
            "rag_input": "A concise, context-aware follow-up in your persona.",
            "question": "A fully self-contained version of the rag_input.",
            "answer": "The expected, comprehensive ground-truth answer.",
            "type": "Follow-up / Clarification / Correction / Comparative",
            "logic_type": "inference / comparison / temporal / none",
            "multi_hop_flag": 0 or 1
        }}
    '''
}