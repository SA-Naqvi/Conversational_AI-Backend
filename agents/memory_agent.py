import json
import re
from lm_studio.client import chat_completion

MEMORY_SYSTEM_PROMPT = """You are a medical data extraction assistant.
Your ONLY job is to extract structured information from a patient message and return a JSON object.

Extract the following fields if present in the message:
- patient_name (string or null)
- surgery_type (string or null)
- surgery_date (string or null — keep as stated by user)
- pain_level (integer 0-10 or null)
- temperature (float in Fahrenheit or null)
- new_symptoms (list of strings, e.g. ["swelling", "fever"])
- medications (string or null)

Rules:
- Return ONLY valid JSON. No explanation, no markdown, no backticks.
- If a field is not mentioned, set it to null or empty list.
- For pain, extract the number only.
- For temperature, extract the numeric value and convert to Fahrenheit if needed.

Example output:
{"patient_name": null, "surgery_type": "appendectomy", "surgery_date": "three days ago", "pain_level": 4, "temperature": null, "new_symptoms": ["mild swelling"], "medications": null}

Do not output internal reasoning tags such as <think> or similar markers.
Respond directly with the answer.
"""

async def extract_memory(user_message: str, current_patient_state: dict) -> dict:
    """Agent 1: Extract structured data from user message."""
    messages = [
        {"role": "system", "content": MEMORY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Current patient state:\n{json.dumps(current_patient_state, indent=2)}\n\nNew patient message:\n{user_message}"
        }
    ]

    raw = await chat_completion(messages)

    # Clean and parse JSON
    raw = raw.strip()
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: return empty extraction on parse failure
        return {
            "patient_name": None, "surgery_type": None, "surgery_date": None,
            "pain_level": None, "temperature": None, "new_symptoms": [], "medications": None
        }
