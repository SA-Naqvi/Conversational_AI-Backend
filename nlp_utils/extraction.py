"""
NLP-based extraction functions using spaCy and dateutil.
Handles relative pain changes, date parsing from narrative text.
"""
import re
from datetime import datetime, timedelta
from typing import Optional

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
except Exception:
    _nlp = None

try:
    from dateutil import parser as date_parser
except ImportError:
    date_parser = None


def extract_relative_pain_change(text: str, previous_pain: int) -> dict:
    """
    Detect relative pain expressions using NLP.

    Examples:
        "pain got worse" + prev=6 → 7-8
        "pain improved" + prev=6 → 4-5
        "pain is the same" → no change
    """
    text_lower = text.lower()

    # Use regex-based detection (works without spaCy too)
    worsen_words = [
        "worse", "worsened", "worsening", "increased", "increasing",
        "higher", "more pain", "gone up", "getting worse",
        "intensified", "aggravated",
    ]
    improve_words = [
        "better", "improved", "improving", "decreased", "decreasing",
        "lower", "less pain", "gone down", "getting better",
        "relieved", "subsided", "eased",
    ]
    same_words = [
        "same", "unchanged", "no change", "stable", "constant",
    ]

    # Check with spaCy lemmatization if available
    if _nlp is not None:
        doc = _nlp(text_lower)
        lemmas = [token.lemma_ for token in doc]
        lemma_text = " ".join(lemmas)

        if any(w in lemma_text for w in ["worsen", "worse", "increase"]):
            return {
                "value": min(previous_pain + 2, 10),
                "confidence": 0.60,
                "method": "relative_worse",
            }
        if any(w in lemma_text for w in ["improve", "better", "decrease", "good"]):
            return {
                "value": max(previous_pain - 2, 0),
                "confidence": 0.60,
                "method": "relative_better",
            }

    # Fallback: simple keyword matching
    for word in worsen_words:
        if word in text_lower:
            return {
                "value": min(previous_pain + 2, 10),
                "confidence": 0.55,
                "method": "relative_worse",
            }

    for word in improve_words:
        if word in text_lower:
            return {
                "value": max(previous_pain - 2, 0),
                "confidence": 0.55,
                "method": "relative_better",
            }

    for word in same_words:
        if word in text_lower:
            return {
                "value": previous_pain,
                "confidence": 0.85,
                "method": "unchanged",
            }

    return {"value": previous_pain, "confidence": 0.40, "method": "unchanged"}


def extract_date_from_narrative(text: str) -> dict:
    """
    Extract dates from natural language.

    Examples:
        "last Monday" → infer from today
        "2 weeks ago" → calculate from today
        "March 5" → assume current year
    """
    now = datetime.now()

    # Try standard parsing first (most reliable)
    if date_parser is not None:
        try:
            parsed_date = date_parser.parse(text, fuzzy=True)
            # Sanity check: date should be in the past or today
            if parsed_date <= now:
                return {
                    "date": parsed_date.strftime("%Y-%m-%d"),
                    "confidence": 0.90,
                    "method": "direct_parse",
                }
        except (ValueError, OverflowError):
            pass

    text_lower = text.lower()

    # "yesterday"
    if "yesterday" in text_lower:
        d = now - timedelta(days=1)
        return {
            "date": d.strftime("%Y-%m-%d"),
            "confidence": 0.90,
            "method": "relative",
        }

    # "today"
    if "today" in text_lower:
        return {
            "date": now.strftime("%Y-%m-%d"),
            "confidence": 0.90,
            "method": "relative",
        }

    # "N days/weeks/months ago"
    ago_match = re.search(r'(\d+)\s*(day|week|month)s?\s*ago', text_lower)
    if ago_match:
        amount = int(ago_match.group(1))
        unit = ago_match.group(2)
        if unit == "day":
            d = now - timedelta(days=amount)
        elif unit == "week":
            d = now - timedelta(weeks=amount)
        elif unit == "month":
            d = now - timedelta(days=amount * 30)
        else:
            d = now
        return {
            "date": d.strftime("%Y-%m-%d"),
            "confidence": 0.80,
            "method": "relative",
        }

    # "last Monday/Tuesday/..."
    day_names = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6
    }
    last_match = re.search(
        r'last\s*(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
        text_lower,
    )
    if last_match:
        target_day = day_names[last_match.group(1)]
        current_day = now.weekday()
        days_back = (current_day - target_day) % 7
        if days_back == 0:
            days_back = 7  # "last Monday" when today is Monday means 7 days ago
        d = now - timedelta(days=days_back)
        return {
            "date": d.strftime("%Y-%m-%d"),
            "confidence": 0.70,
            "method": "relative",
        }

    # "last week"
    if "last week" in text_lower:
        d = now - timedelta(weeks=1)
        return {
            "date": d.strftime("%Y-%m-%d"),
            "confidence": 0.70,
            "method": "relative",
        }

    # "last month"
    if "last month" in text_lower:
        d = now - timedelta(days=30)
        return {
            "date": d.strftime("%Y-%m-%d"),
            "confidence": 0.60,
            "method": "relative",
        }

    # Try spaCy NER for DATE entities
    if _nlp is not None:
        doc = _nlp(text)
        for ent in doc.ents:
            if ent.label_ == "DATE":
                if date_parser is not None:
                    try:
                        parsed = date_parser.parse(ent.text, fuzzy=True)
                        return {
                            "date": parsed.strftime("%Y-%m-%d"),
                            "confidence": 0.75,
                            "method": "ner",
                        }
                    except (ValueError, OverflowError):
                        pass

    return {"date": None, "confidence": 0.0, "method": "failed"}


def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return (celsius * 9 / 5) + 32
