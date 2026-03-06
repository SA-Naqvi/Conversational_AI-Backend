"""
Keyword-based symptom detection with severity estimation.
"""
import re
from typing import List, Dict


# Symptom keyword mapping: symptom_name -> list of keywords/phrases
SYMPTOM_KEYWORDS = {
    "fever": ["fever", "feverish", "febrile", "high temperature"],
    "chills": ["chills", "chilly", "shivering", "shaking"],
    "redness": ["red", "redness", "inflamed", "inflammation", "pink", "irritated"],
    "swelling": ["swelling", "swollen", "swelled", "puffed", "puffy", "edema"],
    "discharge": ["discharge", "pus", "drainage", "oozing", "draining", "leaking"],
    "bleeding": ["bleeding", "bleed", "blood", "hemorrhage", "hemorrhaging"],
    "numbness": ["numb", "numbness", "tingling", "pins and needles", "loss of sensation"],
    "nausea": ["nausea", "nauseous", "queasy", "sick to my stomach", "feel sick"],
    "vomiting": ["vomiting", "vomit", "throwing up", "threw up"],
    "dizziness": ["dizzy", "dizziness", "lightheaded", "light headed", "room spinning"],
    "headache": ["headache", "head hurts", "head ache", "head pain", "migraine"],
    "chest_pain": ["chest pain", "chest hurts", "chest tightness", "chest pressure"],
    "breathing_difficulty": [
        "difficulty breathing", "hard to breathe", "can't breathe", "cannot breathe",
        "shortness of breath", "short of breath", "trouble breathing", "breathless",
        "labored breathing",
    ],
    "loss_of_consciousness": [
        "unconscious", "passed out", "fainted", "blacked out",
        "lost consciousness", "collapse", "collapsed",
    ],
    "fatigue": ["fatigue", "fatigued", "exhausted", "tired", "no energy", "lethargic"],
    "stiffness": ["stiff", "stiffness", "tight", "tightness", "can't move"],
    "bruising": ["bruise", "bruising", "bruised", "discoloration"],
    "itching": ["itch", "itching", "itchy"],
    "constipation": ["constipation", "constipated", "can't go", "no bowel movement"],
    "diarrhea": ["diarrhea", "loose stool", "watery stool", "frequent bowel"],
    "insomnia": ["insomnia", "can't sleep", "trouble sleeping", "sleepless"],
    "loss_of_appetite": ["no appetite", "not hungry", "can't eat", "loss of appetite"],
}

# Severity indicators
SEVERITY_SEVERE = re.compile(
    r'\b(?:severe|extreme|terrible|horrible|awful|unbearable|intense|worst'
    r'|very\s+bad|really\s+bad|excruciating|acute|serious|critical'
    r'|uncontrolled|massive|heavy|a\s+lot\s+of|profuse|soaking)\b',
    re.IGNORECASE
)

SEVERITY_MODERATE = re.compile(
    r'\b(?:moderate|noticeable|significant|considerable|some|fairly|quite)\b',
    re.IGNORECASE
)

SEVERITY_MILD = re.compile(
    r'\b(?:mild|slight|minor|little|small|tiny|barely|faint|light|subtle)\b',
    re.IGNORECASE
)

# Spreading / worsening modifiers
SPREADING_PATTERN = re.compile(
    r'\b(?:spreading|spread|getting\s+worse|worsening|growing|expanding|increasing)\b',
    re.IGNORECASE
)


def detect_symptoms(text: str) -> List[Dict]:
    """
    Detect symptoms from text using keyword matching.

    Returns a list of dicts:
        [
            {
                "name": "redness",
                "severity": "mild",      # mild, moderate, severe
                "confidence": 0.85,
                "spreading": False
            }
        ]
    """
    text_lower = text.lower()
    detected = []
    seen = set()

    for symptom_name, keywords in SYMPTOM_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower and symptom_name not in seen:
                seen.add(symptom_name)

                # Determine severity from surrounding context
                severity = _estimate_severity(text_lower)

                # Check for spreading / worsening
                spreading = bool(SPREADING_PATTERN.search(text_lower))

                detected.append({
                    "name": symptom_name,
                    "severity": severity,
                    "confidence": 0.85,
                    "spreading": spreading,
                })
                break  # Found this symptom, move to next

    return detected


def _estimate_severity(text: str) -> str:
    """Estimate severity from descriptive words in the text."""
    if SEVERITY_SEVERE.search(text):
        return "severe"
    elif SEVERITY_MODERATE.search(text):
        return "moderate"
    elif SEVERITY_MILD.search(text):
        return "mild"
    return "mild"  # Default to mild if no descriptor


def is_critical_symptom(symptom_name: str) -> bool:
    """Check if a symptom name is considered critical (needs immediate attention)."""
    critical = {
        "chest_pain", "breathing_difficulty", "loss_of_consciousness",
        "bleeding", "discharge",
    }
    return symptom_name in critical
