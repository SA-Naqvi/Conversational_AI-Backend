"""
Deterministic Red Flag Engine — Safety Layer.
All safety logic is deterministic. The LLM is NEVER involved in safety decisions.

Checks:
- Critical vital signs (temperature, pain)
- Dangerous symptoms (bleeding, discharge, chest pain, breathing)
- Symptom combinations (fever + chills)
- Pain trends (worsening over time)
- Context-aware thresholds (post-op day matters)
"""
import re
from typing import Tuple

from config import (
    RED_FLAG_TEMP_CRITICAL,
    RED_FLAG_TEMP_HIGH,
    RED_FLAG_PAIN_CRITICAL,
    RED_FLAG_PAIN_HIGH,
)

# Critical symptom keywords checked against the raw message
RED_FLAG_KEYWORDS = [
    "chest pain", "can't breathe", "cannot breathe", "difficulty breathing",
    "shortness of breath", "unconscious", "uncontrolled bleeding",
    "heavy bleeding", "won't stop bleeding", "soaking through",
    "passed out", "fainted", "blacked out", "lost consciousness",
    "pus", "purulent", "yellow discharge", "green discharge",
]


def check_red_flags(message: str, patient_state: dict) -> Tuple[bool, str]:
    """
    Deterministic red flag detection.

    Args:
        message: the user's (filtered) message text
        patient_state: current structured patient state

    Returns:
        (is_red_flag: bool, reason: str)
    """
    msg_lower = message.lower()
    days_post_op = patient_state.get("days_post_op")

    # ── 1. Critical keyword check ──
    for keyword in RED_FLAG_KEYWORDS:
        if keyword in msg_lower:
            return True, f"CRITICAL_KEYWORD: '{keyword}'"

    # ── 2. Temperature checks ──
    flag, reason = check_critical_temp(message, patient_state, days_post_op)
    if flag:
        return True, reason

    # ── 3. Pain checks ──
    flag, reason = check_critical_pain(message, patient_state, days_post_op)
    if flag:
        return True, reason

    # ── 4. Infection signs (symptom combinations) ──
    flag, reason = check_infection_signs(patient_state)
    if flag:
        return True, reason

    # ── 5. Pain trend worsening ──
    flag, reason = check_pain_trend(patient_state, days_post_op)
    if flag:
        return True, reason

    # ── 6. Temperature trend worsening ──
    flag, reason = check_temperature_trend(patient_state)
    if flag:
        return True, reason

    # ── 7. Severe/spreading symptoms ──
    flag, reason = check_severe_symptoms(patient_state, days_post_op)
    if flag:
        return True, reason

    # ── 8. Rapid symptom accumulation ──
    flag, reason = check_symptom_worsening(patient_state)
    if flag:
        return True, reason

    return False, ""


# ── Individual Check Functions ──


def check_critical_temp(
    message: str, patient_state: dict, days_post_op: int = None
) -> Tuple[bool, str]:
    """Check for critical temperature readings."""
    # Check from message (inline temperature report)
    temp_match = re.search(r'(\d{2,3}(?:\.\d{1,2})?)\s*°?\s*[fF]', message)
    if temp_match:
        temp = float(temp_match.group(1))
        if temp > RED_FLAG_TEMP_CRITICAL:
            return True, f"CRITICAL_FEVER: {temp}°F (>{RED_FLAG_TEMP_CRITICAL}°F)"
        if days_post_op is not None and days_post_op >= 7 and temp > RED_FLAG_TEMP_HIGH:
            return True, f"LATE_FEVER: {temp}°F on day {days_post_op}"

    # Check from patient state history
    temp_history = patient_state.get("temperature_history", [])
    if temp_history:
        latest_temp = temp_history[-1]
        if isinstance(latest_temp, (int, float)):
            if latest_temp > RED_FLAG_TEMP_CRITICAL:
                return True, f"CRITICAL_FEVER: {latest_temp}°F"
            if days_post_op is not None and days_post_op >= 7:
                if latest_temp > RED_FLAG_TEMP_HIGH:
                    return True, f"LATE_FEVER: {latest_temp}°F on day {days_post_op}"

    return False, ""


def check_critical_pain(
    message: str, patient_state: dict, days_post_op: int = None
) -> Tuple[bool, str]:
    """Check for critical pain levels."""
    # Check from message
    pain_match = re.search(r'(?:pain\s*(?:is|level|:)?\s*)(\d{1,2})', message, re.IGNORECASE)
    if not pain_match:
        pain_match = re.search(r'\b(\d{1,2})\s*/\s*10\b', message)

    if pain_match:
        pain = int(pain_match.group(1))
        if pain >= RED_FLAG_PAIN_CRITICAL:
            return True, f"SEVERE_PAIN: {pain}/10"
        if days_post_op is not None and days_post_op > 3 and pain >= RED_FLAG_PAIN_HIGH:
            return True, f"UNEXPECTED_HIGH_PAIN: {pain}/10 on day {days_post_op}"

    # Check from patient state
    pain_history = patient_state.get("pain_history", [])
    if pain_history:
        latest_pain = pain_history[-1]
        if isinstance(latest_pain, (int, float)):
            if latest_pain >= RED_FLAG_PAIN_CRITICAL:
                return True, f"SEVERE_PAIN: {latest_pain}/10"
            if days_post_op is not None and days_post_op > 3:
                if latest_pain >= RED_FLAG_PAIN_HIGH:
                    return (
                        True,
                        f"UNEXPECTED_HIGH_PAIN: {latest_pain}/10 on day {days_post_op}",
                    )

    return False, ""


def check_infection_signs(patient_state: dict) -> Tuple[bool, str]:
    """Check for infection sign combinations (fever + chills, etc.)."""
    symptoms = patient_state.get("symptoms", [])
    symptom_names = set()
    for s in symptoms:
        if isinstance(s, str):
            symptom_names.add(s.lower())
        elif isinstance(s, dict):
            symptom_names.add(s.get("name", "").lower())

    temp_history = patient_state.get("temperature_history", [])
    has_fever = False
    if temp_history:
        latest = temp_history[-1]
        if isinstance(latest, (int, float)) and latest >= 100.4:
            has_fever = True

    if "fever" in symptom_names:
        has_fever = True

    # Fever + chills = infection concern
    if has_fever and "chills" in symptom_names:
        return True, "FEVER_WITH_CHILLS"

    # Discharge (pus) = infection
    if "discharge" in symptom_names:
        return True, "INFECTION_SIGNS: discharge/pus"

    return False, ""


def check_pain_trend(
    patient_state: dict, days_post_op: int = None
) -> Tuple[bool, str]:
    """Check if pain is trending upward (worsening)."""
    pain_history = patient_state.get("pain_history", [])
    if len(pain_history) < 3:
        return False, ""

    # Check last 3 readings
    recent = pain_history[-3:]
    if all(isinstance(v, (int, float)) for v in recent):
        if all(recent[i] < recent[i + 1] for i in range(len(recent) - 1)):
            # After day 3: critical red flag
            if days_post_op is not None and days_post_op > 3:
                return (
                    True,
                    f"PAIN_TREND_WORSENING: {recent} on day {days_post_op}",
                )
            # Before day 3: still escalate if pain is high
            if recent[-1] >= 7:
                return (
                    True,
                    f"PAIN_TREND_WORSENING_EARLY: {recent} (high pain, rising)",
                )

    return False, ""


def check_temperature_trend(patient_state: dict) -> Tuple[bool, str]:
    """Check if temperature is trending upward across consecutive readings."""
    temp_history = patient_state.get("temperature_history", [])
    if len(temp_history) < 3:
        return False, ""

    recent = temp_history[-3:]
    if all(isinstance(v, (int, float)) for v in recent):
        # Strictly increasing temperatures
        if all(recent[i] < recent[i + 1] for i in range(len(recent) - 1)):
            if recent[-1] >= 100.4:  # Latest is at least low-grade fever
                return (
                    True,
                    f"TEMPERATURE_TREND_RISING: {recent} (latest {recent[-1]}°F)",
                )

    return False, ""


def check_severe_symptoms(
    patient_state: dict, days_post_op: int = None
) -> Tuple[bool, str]:
    """Check for severe or spreading symptoms."""
    symptoms = patient_state.get("symptoms", [])
    symptom_names = set()
    for s in symptoms:
        if isinstance(s, str):
            symptom_names.add(s.lower())
        elif isinstance(s, dict):
            symptom_names.add(s.get("name", "").lower())

    # Swelling after day 7 is concerning
    if days_post_op is not None and days_post_op >= 7:
        if "swelling" in symptom_names:
            return True, f"LATE_SWELLING: day {days_post_op}"

    # Numbness after day 3 is concerning
    if days_post_op is not None and days_post_op >= 3:
        if "numbness" in symptom_names:
            return True, f"WORSENING_NUMBNESS: day {days_post_op}"

    return False, ""


def is_pain_trend_worsening(patient_state: dict, days: int = 2) -> bool:
    """Check if pain is trending upward over past N readings."""
    history = patient_state.get("pain_history", [])
    if len(history) < days + 1:
        return False

    recent = history[-(days + 1):]
    if all(isinstance(v, (int, float)) for v in recent):
        return all(recent[i] < recent[i + 1] for i in range(len(recent) - 1))
    return False


def is_fever_with_chills(patient_state: dict) -> bool:
    """Check for fever + chills combination."""
    flag, _ = check_infection_signs(patient_state)
    return flag


def check_symptom_worsening(patient_state: dict) -> Tuple[bool, str]:
    """
    Check if symptoms are accumulating rapidly.
    5+ distinct symptoms is a concern.
    """
    symptoms = patient_state.get("symptoms", [])
    if len(symptoms) >= 5:
        symptom_names = [
            s if isinstance(s, str) else s.get("name", "")
            for s in symptoms
        ]
        return (
            True,
            f"RAPID_SYMPTOM_ACCUMULATION: {len(symptoms)} symptoms ({', '.join(symptom_names[:5])})",
        )
    return False, ""
