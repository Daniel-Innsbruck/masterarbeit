"""
Prompt templates for rag_to_be_tested evaluation
"""

CONVERSATION_PROMPTS = {
    'init_multihop_prompt': '''
            Your task: Write the FIRST question to start a conversation with a RAG system. 
            You must create a complex Multi-Hop question based on the provided bridging topics and contexts.

            **Your Persona:** {Role}
            **Bridging Topics:** {t_bridge}

            **Context A:** 
            {chunk_a}

            **Context B:** 
            {chunk_b}

            ### RULES FOR THE QUESTION
            1. **Strict Multi-Hop:** The question MUST require synthesizing facts from BOTH Context A and B. It must be impossible to answer using only one context.
            2. **No Compound Questions:** Do NOT ask two separate questions joined by "and" (e.g., "What is X, and what is Y?"). Ask a single, cohesive question.
            3. **Natural Persona:** Sound like a human naturally starting a chat based on your Persona. Avoid stiff, academic exam-style phrasing. NEVER mention "Context A" or "Context B".
            4. **Scope:** The answer to your question must be entirely contained within the provided snippets.
            5. **No Premise Leakage:** Do not explicitly summarize the facts from both Context A and Context B in your question. A true multi-hop question asks for the connection WITHOUT giving it away. 
               - BAD: "Given that flour prices have tripled [Context B], how did the Iran attack [Context A] cause this?"
               - GOOD: "What direct impact did the recent escalation with Iran have on the pricing of basic baking ingredients like flour?"
                        
            ### PRECISION RULES
            
            - **No Vague Temporal Language:** NEVER use "recently", "lately", "not long ago", etc. Use **specific dates, months, or years** from the contexts (e.g., "in March 2024", "after the 14 April strike").
            - **Specific Answers:** The ground-truth answer must contain concrete facts (dates, numbers, names) that narrow down the possible reference ground truths to precisely the references sources that are given to you to formulate the question.
            
            ### CONFIGURATION
            - **Logic Type:** Choose the reasoning required: 'inference' (connecting premises), 'comparison' (comparing entities), or 'temporal' (timelines/sequences).
            - **First Turn Rule:** Because this is the very first message, `rag_input` and `question` MUST be strictly identical.

            ### OUTPUT FORMAT (Strict JSON)
            {{
                "rag_input": "Your initial, natural question matching your Persona.",
                "question": "Exactly the same as rag_input.",
                "answer": "The comprehensive ground-truth answer combining facts from both contexts.",
                "type": "Initial",
                "logic_type": "inference / comparison / temporal",
                "multi_hop_flag": 1,
                "bridging_topic": "A short summary of the specific bridging topic you used."
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