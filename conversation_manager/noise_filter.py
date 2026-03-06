"""
Context-aware noise filtering for patient messages.
Two-stage filtering: safe normalization + selective medical context preservation.
"""
import re

from config import MAX_CHARS_AFTER_FILTER, AGGRESSIVE_FILTER_THRESHOLD

# Internet slang and filler words to remove
FILLER_WORDS = {
    "uh", "um", "uhm", "hmm", "hm", "like", "yeah", "yea", "yep",
    "nah", "ok", "okay", "alright", "well", "so", "basically",
    "actually", "literally", "lol", "lmao", "omg", "btw",
    "idk", "tbh", "smh", "brb", "imo", "imho",
}

# Medical keywords that should always be preserved
MEDICAL_KEYWORDS = {
    # Symptoms
    "pain", "ache", "hurts", "hurt", "sore", "tender",
    "fever", "temperature", "temp", "hot", "chills",
    "swelling", "swollen", "puffy", "edema",
    "redness", "red", "inflamed", "inflammation",
    "discharge", "pus", "drainage", "oozing",
    "bleeding", "blood", "bleed", "hemorrhage",
    "numb", "numbness", "tingling",
    "nausea", "vomiting", "dizzy", "dizziness",
    "headache", "migraine",
    "stiff", "stiffness",
    "bruise", "bruising",
    "itch", "itching",
    # Body parts
    "knee", "hip", "shoulder", "back", "chest", "stomach",
    "incision", "wound", "stitches", "sutures", "bandage",
    "joint", "muscle", "bone", "leg", "arm",
    # Activities
    "walk", "walking", "move", "movement", "exercise",
    "sleep", "sleeping", "rest", "resting",
    "eat", "eating", "drink", "drinking",
    # Medical terms
    "surgery", "operation", "procedure",
    "medication", "medicine", "pill", "pills",
    "doctor", "nurse", "clinic", "hospital", "er", "emergency",
    "appointment", "follow-up", "checkup",
    # Timing
    "today", "yesterday", "morning", "night", "evening",
    "week", "day", "hour", "ago", "since", "last",
    # Numbers (kept for pain/temp)
    "scale", "level", "degree", "degrees",
    # Recovery
    "better", "worse", "improved", "worsened", "same",
    "recovery", "healing", "progress",
}

# Timing words that should always be preserved
TIMING_WORDS = {
    "today", "yesterday", "morning", "night", "evening", "afternoon",
    "week", "day", "hour", "minute", "ago", "since", "last",
    "before", "after", "during", "when", "earlier",
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday",
}


def filter_text(text: str, context: dict = None) -> str:
    """
    Two-stage noise filtering.

    Stage 1: Safe filtering (always applied)
    Stage 2: Selective filtering (context-aware)

    Args:
        text: raw user input
        context: optional session context for smarter filtering

    Returns:
        Filtered text string
    """
    if not text:
        return ""

    # Stage 1: Safe filtering
    filtered = _safe_filter(text)

    # Stage 2: Context-aware filtering
    if len(filtered) > AGGRESSIVE_FILTER_THRESHOLD:
        filtered = aggressive_filter(filtered)
    else:
        filtered = _selective_filter(filtered)

    # Enforce max length
    if len(filtered) > MAX_CHARS_AFTER_FILTER:
        filtered = filtered[:MAX_CHARS_AFTER_FILTER]

    return filtered.strip()


def _safe_filter(text: str) -> str:
    """
    Stage 1: Always-apply safe normalization.
    - Lowercase
    - Normalize whitespace
    - Remove internet slang and filler words
    """
    text = text.lower().strip()

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove filler words (only standalone words, not substrings)
    words = text.split()
    filtered_words = [w for w in words if w not in FILLER_WORDS]

    return " ".join(filtered_words)


def _selective_filter(text: str) -> str:
    """
    Stage 2: Context-aware selective filtering.
    Preserves medical content and timing information.
    """
    # If the text contains medical keywords, keep it mostly intact
    if _contains_medical_content(text):
        return text

    # For non-medical content, extract only medically relevant parts
    words = text.split()
    kept = []
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word)
        if clean_word in MEDICAL_KEYWORDS or clean_word in TIMING_WORDS:
            kept.append(word)
        # Always keep numbers (could be pain levels, temperatures)
        elif re.match(r'\d+\.?\d*', clean_word):
            kept.append(word)

    if kept:
        return " ".join(kept)

    # If nothing medical was found, return original
    # (might be a greeting or general question — LLM handles those)
    return text


def _contains_medical_content(text: str) -> bool:
    """Check if text contains any medical keywords."""
    words = set(re.sub(r'[^\w\s]', '', text).split())
    return bool(words & MEDICAL_KEYWORDS)


def preserve_medical_context(text: str) -> bool:
    """Check if text should be kept in full (has medical context)."""
    return _contains_medical_content(text)


def is_medical_keyword(word: str) -> bool:
    """Check if a single word is a medical keyword."""
    return word.lower() in MEDICAL_KEYWORDS


def aggressive_filter(text: str) -> str:
    """
    Aggressive filtering for very long inputs.
    Extracts only medical keywords and numbers.
    """
    words = text.split()
    kept = []
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word).lower()
        if clean_word in MEDICAL_KEYWORDS or clean_word in TIMING_WORDS:
            kept.append(word)
        elif re.match(r'\d+\.?\d*', clean_word):
            kept.append(word)

    return " ".join(kept) if kept else text[:MAX_CHARS_AFTER_FILTER]
