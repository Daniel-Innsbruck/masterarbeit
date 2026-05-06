"""
Parser for LLM responses in rag_to_be_tested evaluation
"""
from typing import Dict, Optional

class LLMResponseParser:
   
    
    def parse_and_validate(self, response: str) -> Optional[Dict[str, str]]:
        """
        Parse and validate response in one step
        
        Args:
            response (str): Raw LLM response
            
        Returns:
            Optional[Dict[str, str]]: Parsed data if valid, None otherwise
        """
        try:
            import json, re
            match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
            json_str = match.group(1) if match else response.strip()

            json_data = json.loads(json_str)

            if isinstance(json_data, dict):
                normalized = {}
                for k, v in json_data.items():
                    key = k.lower().replace(" ", "_")
                    if isinstance(v, str):
                        normalized[key] = v.strip()
                    elif isinstance(v, bool):
                        normalized[key] = v
                    else:
                        normalized[key] = str(v)

                if 'question' in normalized and 'rag_input' not in normalized:
                    normalized['rag_input'] = normalized['question']

                is_initial_turn = 'thematic_link' in normalized or 'logic_type' in normalized

                if is_initial_turn:
                    required_fields = ['rag_input', 'question', 'answer', 'thematic_link', 'logic_type']
                else:
                    required_fields = ['rag_input', 'question', 'answer', 'type']

                # Validation
                if all(field in normalized for field in required_fields):
                    return {field: normalized[field] for field in required_fields}
                else:
                    missing = [field for field in required_fields if field not in normalized]

                if missing:
                    print(f"Warning: Missing fields: {missing}")

        except json.JSONDecodeError:
            print("Error: Response is not valid JSON.")
            print("Response text:", response)

        return None