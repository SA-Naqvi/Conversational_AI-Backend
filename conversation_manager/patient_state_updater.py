"""
Hybrid NLP + Regex patient state extraction.
Extracts pain levels, temperatures, symptoms, surgery dates, and surgery types
from user messages WITHOUT relying on an LLM.
"""
import re
from datetime import datetime
from typing import Optional, List

from nlp_utils.regex_patterns import (
    PAIN_DIRECT, PAIN_STANDALONE, PAIN_DESCRIPTORS,
    PAIN_WORSENED, PAIN_IMPROVED, PAIN_SAME,
    TEMP_FAHRENHEIT, TEMP_CELSIUS, TEMP_CONTEXTUAL, TEMP_FEVER_WORDS,
    DATE_FULL, SURGERY_TYPE_PATTERN, NAME_PATTERN,
)
from nlp_utils.extraction import (
    extract_relative_pain_change,
    extract_date_from_narrative,
    celsius_to_fahrenheit,
)
from nlp_utils.symptom_detector import detect_symptoms


# ============ Pain Extraction ============

def extract_pain_level(text: str, previous_pain: int = None) -> dict:
    """
    Extract pain level from text using multi-method approach.

    Returns:
        {"value": int (0-10), "confidence": float, "method": str}
    """
    text_lower = text.lower()

    # Method 1: Direct numeric — "my pain is 7", "pain level 8"
    match = PAIN_DIRECT.search(text_lower)
    if match:
        value = int(match.group(1))
        if 0 <= value <= 10:
            return {"value": value, "confidence": 0.95, "method": "direct_numeric"}

    # Method 2: Standalone with /10 — "it's a 7/10", "about 6 out of 10"
    match = PAIN_STANDALONE.search(text_lower)
    if match:
        value = int(match.group(1))
        if 0 <= value <= 10:
            return {"value": value, "confidence": 0.90, "method": "regex"}

    # Method 3: Word-based descriptors — "severe pain", "mild"
    for descriptor, value in sorted(
        PAIN_DESCRIPTORS.items(), key=lambda x: len(x[0]), reverse=True
    ):
        if descriptor in text_lower:
            return {"value": value, "confidence": 0.70, "method": "descriptor"}

    # Method 4: Relative expressions — "pain got worse"
    if previous_pain is not None:
        if PAIN_WORSENED.search(text_lower):
            return extract_relative_pain_change(text, previous_pain)
        if PAIN_IMPROVED.search(text_lower):
            return extract_relative_pain_change(text, previous_pain)
        if PAIN_SAME.search(text_lower):
            return {
                "value": previous_pain,
                "confidence": 0.85,
                "method": "unchanged",
            }

    return {"value": None, "confidence": 0.0, "method": "failed"}


# ============ Temperature Extraction ============

def extract_temperature(text: str) -> dict:
    """
    Extract temperature from text.

    Returns:
        {"value": float (Fahrenheit), "unit": str, "confidence": float, "method": str}
    """
    # Method 1: Fahrenheit — "101.5F", "101.5°F"
    match = TEMP_FAHRENHEIT.search(text)
    if match:
        value = float(match.group(1))
        if 90.0 <= value <= 115.0:  # Reasonable F range
            return {
                "value": value, "unit": "F",
                "confidence": 0.95, "method": "direct_numeric",
            }

    # Method 2: Celsius — "38.5C"
    match = TEMP_CELSIUS.search(text)
    if match:
        value = float(match.group(1))
        if 30.0 <= value <= 45.0:  # Reasonable C range
            f_value = round(celsius_to_fahrenheit(value), 1)
            return {
                "value": f_value, "unit": "F",
                "confidence": 0.95, "method": "celsius_converted",
            }

    # Method 3: Contextual — "temperature is 101.5"
    match = TEMP_CONTEXTUAL.search(text)
    if match:
        value = float(match.group(1))
        if 90.0 <= value <= 115.0:
            return {
                "value": value, "unit": "F",
                "confidence": 0.85, "method": "contextual",
            }

    # Method 4: Fever words (no numeric) — estimate range
    if TEMP_FEVER_WORDS.search(text):
        return {
            "value": None, "unit": "F",
            "confidence": 0.50, "method": "fever_words",
            "estimated_range": (100.4, 102.0),
        }

    return {"value": None, "unit": "F", "confidence": 0.0, "method": "failed"}


# ============ Symptoms Extraction ============

def extract_symptoms(text: str) -> List[dict]:
    """
    Extract symptoms from text using keyword matching.

    Returns:
        [{"name": str, "severity": str, "confidence": float, "spreading": bool}]
    """
    return detect_symptoms(text)


# ============ Surgery Date Extraction ============

def extract_surgery_date(text: str) -> dict:
    """
    Extract surgery date from text.

    Returns:
        {"date": str (YYYY-MM-DD) or None, "confidence": float, "method": str}
    """
    return extract_date_from_narrative(text)


# ============ Surgery Type Extraction ============

def extract_surgery_type(text: str) -> dict:
    """
    Extract surgery type from text.

    Returns:
        {"type": str or None, "confidence": float, "normalized": str}
    """
    match = SURGERY_TYPE_PATTERN.search(text.lower())
    if match:
        raw = match.group(1)
        normalized = _normalize_surgery_type(raw)
        return {
            "type": raw,
            "confidence": 0.90,
            "normalized": normalized,
        }

    # If no known surgery matched, check if they mentioned "surgery" at all
    if re.search(r'\b(?:surgery|operation|procedure)\b', text, re.IGNORECASE):
        # Try to capture surrounding words
        match = re.search(
            r'(\w+(?:\s+\w+)?)\s+(?:surgery|operation|procedure)',
            text, re.IGNORECASE,
        )
        if match:
            surgery_desc = match.group(0)
            return {
                "type": surgery_desc.strip(),
                "confidence": 0.60,
                "normalized": surgery_desc.strip().lower(),
            }

    return {"type": None, "confidence": 0.0, "normalized": ""}


def _normalize_surgery_type(raw: str) -> str:
    """Normalize surgery type names."""
    normalization = {
        "appendix removal": "appendectomy",
        "gallbladder removal": "cholecystectomy",
        "c-section": "cesarean section",
        "caesarean": "cesarean section",
        "acl repair": "acl reconstruction",
        "acl surgery": "acl reconstruction",
        "tonsil removal": "tonsillectomy",
        "cataract removal": "cataract surgery",
        "wisdom teeth": "wisdom tooth extraction",
        "tooth extraction": "dental extraction",
        "dental surgery": "dental procedure",
        "bypass surgery": "heart bypass",
        "gastric bypass": "bariatric surgery",
    }
    return normalization.get(raw.lower(), raw.lower())


# ============ Name Extraction ============

def extract_patient_name(text: str) -> dict:
    """
    Extract patient name from text.

    Returns:
        {"name": str or None, "confidence": float}
    """
    match = NAME_PATTERN.search(text)
    if match:
        return {"name": match.group(1).strip(), "confidence": 0.85}
    return {"name": None, "confidence": 0.0}


# ============ Days Post-Op Calculation ============

def calculate_days_post_op(surgery_date_str: str) -> Optional[int]:
    """Calculate days since surgery."""
    if not surgery_date_str:
        return None
    try:
        surgery_date = datetime.strptime(surgery_date_str, "%Y-%m-%d")
        delta = datetime.now() - surgery_date
        return max(delta.days, 0)
    except (ValueError, TypeError):
        return None


# ============ Unified Extraction ============

def extract_all(text: str, session: dict) -> dict:
    """
    Run all extractors on user text.

    Returns a dict with all extraction results:
        {
            "patient_name": {...},
            "surgery_type": {...},
            "surgery_date": {...},
            "pain_level": {...},
            "temperature": {...},
            "symptoms": [...],
        }
    """
    patient_state = session.get("patient_state", {})

    # Get previous pain for relative calculations
    pain_history = patient_state.get("pain_history", [])
    previous_pain = pain_history[-1] if pain_history else None

    return {
        "patient_name": extract_patient_name(text),
        "surgery_type": extract_surgery_type(text),
        "surgery_date": extract_surgery_date(text),
        "pain_level": extract_pain_level(text, previous_pain),
        "temperature": extract_temperature(text),
        "symptoms": extract_symptoms(text),
    }


def update_patient_state(session: dict, extraction: dict) -> None:
    """
    Apply extraction results to the session's patient state.
    Only updates fields where extraction confidence is above threshold.
    """
    patient_state = session.get("patient_state", {})

    # Name
    name_result = extraction.get("patient_name", {})
    if name_result.get("name") and name_result.get("confidence", 0) > 0.50:
        patient_state["patient_name"] = name_result["name"]

    # Surgery type
    surgery_result = extraction.get("surgery_type", {})
    if surgery_result.get("type") and surgery_result.get("confidence", 0) > 0.50:
        patient_state["surgery_type"] = surgery_result["type"]

    # Surgery date
    date_result = extraction.get("surgery_date", {})
    if date_result.get("date") and date_result.get("confidence", 0) > 0.50:
        patient_state["surgery_date"] = date_result["date"]
        patient_state["days_post_op"] = calculate_days_post_op(date_result["date"])

    # Pain level
    pain_result = extraction.get("pain_level", {})
    if pain_result.get("value") is not None and pain_result.get("confidence", 0) > 0.40:
        patient_state.setdefault("pain_history", []).append(pain_result["value"])

    # Temperature
    temp_result = extraction.get("temperature", {})
    if temp_result.get("value") is not None and temp_result.get("confidence", 0) > 0.50:
        patient_state.setdefault("temperature_history", []).append(
            temp_result["value"]
        )

    # Symptoms
    symptoms_result = extraction.get("symptoms", [])
    existing_symptoms = set(
        s if isinstance(s, str) else s.get("name", "")
        for s in patient_state.get("symptoms", [])
    )
    for symptom in symptoms_result:
        symptom_name = symptom.get("name", "")
        if symptom_name and symptom_name not in existing_symptoms:
            patient_state.setdefault("symptoms", []).append(symptom_name)
            existing_symptoms.add(symptom_name)

    session["patient_state"] = patient_state

    # Compute recovery analytics after every state update
    compute_recovery_analytics(patient_state)


# ============ Recovery Analytics ============

def compute_recovery_analytics(patient_state: dict) -> None:
    """
    Compute and attach recovery analytics to patient state:
    - recovery_score (0.0 – 1.0)
    - pain_trend ("improving", "worsening", "stable")
    - fever_status ("normal", "low_grade", "elevated", "critical")
    """
    # ── Pain trend ──
    pain_history = patient_state.get("pain_history", [])
    if len(pain_history) >= 2:
        if pain_history[-1] < pain_history[-2]:
            patient_state["pain_trend"] = "improving"
        elif pain_history[-1] > pain_history[-2]:
            patient_state["pain_trend"] = "worsening"
        else:
            patient_state["pain_trend"] = "stable"
    elif len(pain_history) == 1:
        patient_state["pain_trend"] = "stable"

    # ── Fever status ──
    temp_history = patient_state.get("temperature_history", [])
    if temp_history:
        latest_temp = temp_history[-1]
        if isinstance(latest_temp, (int, float)):
            if latest_temp >= 102.0:
                patient_state["fever_status"] = "critical"
            elif latest_temp >= 101.0:
                patient_state["fever_status"] = "elevated"
            elif latest_temp >= 100.4:
                patient_state["fever_status"] = "low_grade"
            else:
                patient_state["fever_status"] = "normal"

    # ── Recovery score (0.0 = critical, 1.0 = healthy) ──
    score = 1.0
    deductions = 0.0

    # Pain penalty (latest pain / 10)
    if pain_history:
        latest_pain = pain_history[-1]
        if isinstance(latest_pain, (int, float)):
            deductions += (latest_pain / 10) * 0.35  # max 0.35 deduction

    # Pain trend penalty
    if patient_state.get("pain_trend") == "worsening":
        deductions += 0.10

    # Temperature penalty
    if patient_state.get("fever_status") == "critical":
        deductions += 0.30
    elif patient_state.get("fever_status") == "elevated":
        deductions += 0.20
    elif patient_state.get("fever_status") == "low_grade":
        deductions += 0.10

    # Symptom penalty (each symptom = 0.05, max 0.25)
    symptom_count = len(patient_state.get("symptoms", []))
    deductions += min(symptom_count * 0.05, 0.25)

    # Red flag detected
    if patient_state.get("red_flag_detected"):
        deductions += 0.30

    score = max(0.0, score - deductions)
    patient_state["recovery_score"] = round(score, 2)
