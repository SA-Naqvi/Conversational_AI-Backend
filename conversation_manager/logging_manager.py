"""
Structured JSON logging for the medical bot.
"""
import logging
import json
import os
from datetime import datetime

from config import LOG_LEVEL, LOG_FILE

# Ensure logs directory exists
os.makedirs(os.path.dirname(LOG_FILE) if os.path.dirname(LOG_FILE) else "logs", exist_ok=True)

# Configure logger
logger = logging.getLogger("medical_bot")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# File handler
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def log_extraction(session_id: str, extraction: dict, confidence: float):
    """Log an extraction event."""
    logger.info(json.dumps({
        "event": "extraction",
        "session_id": session_id,
        "extraction": _safe_serialize(extraction),
        "confidence": confidence,
        "timestamp": datetime.now().isoformat(),
    }))


def log_red_flag(session_id: str, category: str, details: dict):
    """Log a red flag detection event."""
    logger.warning(json.dumps({
        "event": "red_flag_detected",
        "session_id": session_id,
        "category": category,
        "details": _safe_serialize(details),
        "timestamp": datetime.now().isoformat(),
    }))


def log_message_processing(session_id: str, duration_ms: float, state: str):
    """Log a message processing event."""
    logger.info(json.dumps({
        "event": "message_processed",
        "session_id": session_id,
        "duration_ms": round(duration_ms, 2),
        "state": state,
        "timestamp": datetime.now().isoformat(),
    }))


def log_state_transition(session_id: str, from_state: str, to_state: str):
    """Log a state machine transition."""
    logger.info(json.dumps({
        "event": "state_transition",
        "session_id": session_id,
        "from_state": from_state,
        "to_state": to_state,
        "timestamp": datetime.now().isoformat(),
    }))


def log_error(session_id: str, error_type: str, message: str):
    """Log an error event."""
    logger.error(json.dumps({
        "event": "error",
        "session_id": session_id,
        "error_type": error_type,
        "message": message,
        "timestamp": datetime.now().isoformat(),
    }))


def _safe_serialize(obj):
    """Safely serialize an object for JSON logging."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)
