"""
Input validation and sanitization for user messages.
Validates length, forbidden characters, and applies rate limiting.
"""
import html
import re
import time
from typing import Dict, List

from config import (
    MAX_INPUT_LENGTH,
    MIN_INPUT_LENGTH,
    RATE_LIMIT_MESSAGES_PER_MINUTE,
    RATE_LIMIT_WINDOW,
    FORBIDDEN_CHARS,
)

# In-memory rate limiting store: session_id -> list of timestamps
_rate_limits: Dict[str, List[float]] = {}


def validate_input(text: str, session_id: str) -> dict:
    """
    Validate and sanitize user input.

    Returns:
        {
            "is_valid": bool,
            "sanitized_text": str,
            "validation_errors": [str],
            "flags": {
                "is_empty": bool,
                "is_too_long": bool,
                "contains_invalid_chars": bool,
                "rate_limited": bool
            }
        }
    """
    errors = []
    flags = {
        "is_empty": False,
        "is_too_long": False,
        "contains_invalid_chars": False,
        "rate_limited": False,
    }

    # Check empty / whitespace-only
    if not text or not text.strip():
        flags["is_empty"] = True
        errors.append("Message cannot be empty.")
        return {
            "is_valid": False,
            "sanitized_text": "",
            "validation_errors": errors,
            "flags": flags,
        }

    # Check length
    if len(text) > MAX_INPUT_LENGTH:
        flags["is_too_long"] = True
        errors.append(
            f"Message is too long ({len(text)} characters). "
            f"Maximum is {MAX_INPUT_LENGTH} characters."
        )

    if len(text.strip()) < MIN_INPUT_LENGTH:
        flags["is_empty"] = True
        errors.append("Message is too short.")

    # Check forbidden characters
    for char in FORBIDDEN_CHARS:
        if char in text:
            flags["contains_invalid_chars"] = True
            break

    # Rate limiting
    if _check_rate_limit(session_id):
        flags["rate_limited"] = True
        errors.append("You're sending messages too quickly. Please wait a moment.")

    # Sanitize
    sanitized = sanitize_text(text)

    is_valid = len(errors) == 0
    return {
        "is_valid": is_valid,
        "sanitized_text": sanitized,
        "validation_errors": errors,
        "flags": flags,
    }


def sanitize_text(text: str) -> str:
    """
    Sanitize user input:
    - HTML-escape dangerous characters
    - Normalize whitespace
    - Trim to max length
    """
    # HTML escape to neutralize injection
    text = html.escape(text)

    # Normalize whitespace (collapse multiple spaces, strip)
    text = re.sub(r'\s+', ' ', text).strip()

    # Truncate if still too long after sanitization
    if len(text) > MAX_INPUT_LENGTH:
        text = text[:MAX_INPUT_LENGTH]

    return text


def _check_rate_limit(session_id: str) -> bool:
    """
    Check if a session has exceeded the rate limit.
    Returns True if rate-limited.
    """
    now = time.time()

    if session_id not in _rate_limits:
        _rate_limits[session_id] = []

    # Remove timestamps outside the window
    _rate_limits[session_id] = [
        t for t in _rate_limits[session_id]
        if now - t < RATE_LIMIT_WINDOW
    ]

    # Check limit
    if len(_rate_limits[session_id]) >= RATE_LIMIT_MESSAGES_PER_MINUTE:
        return True

    # Record this message
    _rate_limits[session_id].append(now)
    return False


def detect_malicious_input(text: str) -> bool:
    """
    Detect potentially malicious input patterns.
    Returns True if suspicious.
    """
    suspicious_patterns = [
        r'<script',
        r'javascript:',
        r'on\w+\s*=',
        r'(?:SELECT|INSERT|UPDATE|DELETE|DROP)\s',
        r'(?:UNION\s+SELECT)',
        r'--\s*$',
    ]
    for pattern in suspicious_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False
