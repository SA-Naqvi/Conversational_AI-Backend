import re
from typing import Tuple

RED_FLAG_KEYWORDS = [
    "chest pain", "can't breathe", "cannot breathe", "difficulty breathing",
    "shortness of breath", "unconscious", "uncontrolled bleeding",
    "heavy bleeding", "won't stop bleeding"
]

def check_red_flags(message: str, patient_state: dict) -> Tuple[bool, str]:
    """
    Returns (is_red_flag, reason).
    Deterministic — does NOT rely on LLM.
    """
    msg_lower = message.lower()

    # Keyword check
    for keyword in RED_FLAG_KEYWORDS:
        if keyword in msg_lower:
            return True, f"Reported symptom: '{keyword}'"

    # Fever check — extract temperature
    temp_match = re.search(r'(\d{2,3}(?:\.\d)?)\s*°?[fF]', message)
    if temp_match:
        temp = float(temp_match.group(1))
        if temp > 102.0:
            return True, f"High fever detected: {temp}°F"

    # Severe pain check
    pain_match = re.search(r'\b([0-9]|10)\b', message)
    if pain_match:
        pain = int(pain_match.group(1))
        if pain >= 9:
            return True, f"Severe pain reported: {pain}/10"

    return False, ""
