"""
Parser for LLM responses in rag_to_be_tested evaluation
"""

import re
import json
from typing import Dict, Optional

class LLMResponseParser:

    def parse_and_validate(self, response: str) -> Optional[Dict[str, str]]:
        """
        Parses the output of the Conversation GENERATOR (CG).
        Checks for the questions, answers, and logic types.
        """
        try:
            if response.strip().startswith("```json"):
                response = re.sub(r"^```json\s*", "", response.strip())
                response = re.sub(r"\s*```$", "", response.strip())

            json_data = json.loads(response)

            if isinstance(json_data, dict):
                normalized = {}
                for k, v in json_data.items():
                    key = k.lower().replace(" ", "_")
                    normalized[key] = str(v).strip() if isinstance(v, str) else str(v)

                if 'question' in normalized and 'rag_input' not in normalized:
                    normalized['rag_input'] = normalized['question']

                is_initial_turn = 'bridging_topic' in normalized or str(normalized.get('type', '')).lower() == 'initial'

                if is_initial_turn:
                    required_fields = ['rag_input', 'question', 'answer', 'type', 'logic_type', 'multi_hop_flag',
                                       'bridging_topic']
                else:
                    required_fields = ['rag_input', 'question', 'answer', 'type', 'logic_type', 'multi_hop_flag']

                if all(field in normalized for field in required_fields):
                    # Safety cast for multi_hop_flag to ensure it is an integer (1 or 0)
                    try:
                        normalized['multi_hop_flag'] = int(normalized['multi_hop_flag'])
                    except (ValueError, TypeError):
                        normalized['multi_hop_flag'] = 0
                    return {field: normalized[field] for field in required_fields}
                else:
                    missing = [field for field in required_fields if field not in normalized]
                    print(f"Parser Warning: Missing fields in generation: {missing}")

        except json.JSONDecodeError:
            print("Parser Error: Generator LLM did not return valid JSON.")

        return None

    def parse_and_validate_validation(self, response: str) -> Optional[Dict]:
        """
        Parses the output of the Conversation VALIDATOR (CV).
        Strictly looks for {"correct": true/false, "reason": "..."}
        """
        try:
            if response.strip().startswith("```json"):
                response = re.sub(r"^```json\s*", "", response.strip())
                response = re.sub(r"\s*```$", "", response.strip())

            json_data = json.loads(response)
            if isinstance(json_data, dict):
                normalized = {}
                for k, v in json_data.items():
                    key = k.lower().replace(" ", "_")
                    # Keep booleans as booleans, strip strings
                    normalized[key] = v.strip() if isinstance(v, str) else v

                required_fields = ['correct', 'reason']

                if all(field in normalized for field in required_fields):
                    # Safety catch: if the LLM returned "true" as a string instead of a boolean
                    if isinstance(normalized['correct'], str):
                        normalized['correct'] = normalized['correct'].lower() == 'true'

                    return {field: normalized[field] for field in required_fields}

        except json.JSONDecodeError:
            pass

        print("Warning: Incomplete CV validation response received")
        return None