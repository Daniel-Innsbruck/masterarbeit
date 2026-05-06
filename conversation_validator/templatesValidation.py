"""
Prompt templates for rag_to_be_tested evaluation
"""



VALIDATION_PROMPTS = {
    'validate_init_prompt': '''
            Validate the following initial multi-hop prompt used to test a Retrieval-Augmented Generation system.  
            This represents the start of a conversation.  

            Input to validate (JSON):  
            {question}  
            
            Provided Context: 
            {document}

            Requirements:  
            1. Cross-Document Reasoning: The question MUST require synthesizing facts from BOTH Snippet A and Snippet B. If the question can be fully answered using only one of the snippets, it is invalid.
            2. Factual Accuracy: The provided answer must be factually correct based on the combined context.
            3. Thematic Link: The identified 'thematic_link' must be logically sound, contextually well-founded, and genuinely connect the two snippets.
            4. Query Type Validation: The JSON contains a 'logic_type'. The formulated question MUST genuinely match the definition of this specific type:
               * inference: The question requires deducting a conclusion, underlying motive, or implicit fact by combining information from both snippets.
               * comparison: The question requires comparing, contrasting, or finding similarities between entities, events, or numbers mentioned across the two snippets.
               * temporal: The question requires establishing a chronological sequence, a timeline of events, or a cause-and-effect chain over time using both snippets.
            5. Natural Phrasing: The question must not explicitly mention "Snippet A", "Snippet B", or "the document".

            If ANY of the above requirements are not met, return the reason and set `"correct"` to `false`.  
            If ALL requirements are met, set `"correct"` to `true`.  

            **Output Format (exact JSON):**
            {{
                "correct": true,
                "reason": "Explanation of which requirement failed and why (leave empty if correct)"
            }}
        ''',
    
    'validate_follow_up_prompt': '''
             Validate the input against the rules below.
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
            Input: {question}
            History: {conversation_history}

                **Output Format (exact JSON):** 
                "correct": true/false, 
                "reason": "Explanation of which requirement failed and why (leave empty if correct)"
                    
        
        '''
}