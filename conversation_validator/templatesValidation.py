"""
Prompt templates for rag_to_be_tested evaluation
"""

VALIDATION_PROMPTS = {''
    'validate_init_prompt': '''
        Validate the following initial prompt.  
        It contains a `RAG_input` used to test a Retrieval-Augmented Generation (rag_to_be_tested) system.  
        This should represent the start of a conversation with the rag_to_be_tested system.  

        Requirements:  
        1. The context of the RAG_input should be related to the provided document.  
        2. The question must be exactly the same as the `RAG_input`.  
        3. The answer must be correct.  
        4. Multi-Hop Integrity: The answer MUST require synthesising information from BOTH provided snippets (Context A and Context B). It must not be answerable by only one snippet and must not be a compound question.
        5. Query Type Alignment: The logical reasoning used must strictly match the declared `logic_type`.

        Input to validate:  
        {question}  
        Provided Documents (Contains Context A and Context B):
        {document}

        If any of the above requirements are not met, return the reason and set `"correct"` to `false`.  
        If all requirements are met, set `"correct"` to `true`.  

        Return the result as JSON in the following format:

            "correct": true/false,
            "reason": "Explanation if incorrect, otherwise empty"

    ''',

    'validate_follow_up_multihop_prompt': '''
         Validate the MULTI-HOP follow-up input against the rules below.

         1. RAG_input
            Must be clear, specific, and contextually connected. It must build on the prior turn and use conversational pronouns (e.g., this, it, that). Never mention documents.

         2. Question
            Must be a standalone rephrasing of the rag_input. Must not be identical to rag_input.

         3. Type
            Must be one of: Follow-up, Clarification, Correction, Comparative. Must match its declared conversational type.

         4. Answer
            Must be factually correct.

         5. STRICT Multi-Hop Integrity & Logic Type
            - DUAL CONTEXT REQUIREMENT: The question MUST require synthesizing facts from BOTH available contexts to form a complete answer. If the question can be fully answered using only one context, it fails.
            - NO COMPOUND QUESTIONS: It must NOT be two separate single-hop questions joined together (e.g., "What did A do, and what did B do?"). The synthesis must be required to answer a single underlying question
         
         Input to validate: 
         {question}

         Conversation History: 
         {conversation_history}

         Currently Active Context (Use this for factual verification and checking Multi-Hop Context Requirement):
         {active_context}

         Return the result as JSON in the following format:
         {{
            "correct": true/false,
            "reason": "Explanation of which requirement failed and why (leave empty if correct)"
         }}
    ''',

    'validate_follow_up_singlehop_prompt': '''
        Validate the SINGLE-HOP follow-up input against the rules below.
        
        1. RAG_input
        
        Must be clear, specific, and contextually connected.
        It must clearly build on the prior turn (not standalone)
        Never mention that a document was provided.
        
        2.  Question
        
        Must be a standalone rephrasing of the RAG_input (self-contained, understandable without history).
        Must include necessary context but not refer to “a document.”
        Must not be identical to RAG_input.
        
        3. Type
        
        Must be one of: Follow-up, Clarification, Correction, Comparative.
        Must match its declared type:
        Follow-up: continues or builds on the previous answer.
        Clarification: asks for clarification about the previous answer.
        Correction: points out or requests a correction to the previous answer.
        Comparative: asks to compare or contrast items from earlier in the conversation.
        
        
        4. Answer
        
        Must be factually correct and relevant to the RAG_input / standalone Question.
        
        
        5. STRICT Single-Hop Integrity
        
        The question must rely strictly on a SINGLE context or just the conversation history. It does NOT require multi-hop synthesis. 
        The `logic_type` MUST be "none".
        
        Input to validate: 
        {question}
        
        Conversation History: 
        {conversation_history}
        
        Currently Active Context (Use this for factual verification and checking Single-Hop Context Requirement):
        {active_context}
        
        Return the result as JSON in the following format:
        {{
            "correct": true/false,
            "reason": "Explanation of which requirement failed and why (leave empty if correct)"
        }}
    '''
}