"""
Session management with persistence.
Sessions are kept in-memory for speed, but synced to disk for durability.
Saved to backend/data/sessions/{sid}.json
"""
import os
import json
import logging
from typing import Dict, List
from datetime import datetime, timedelta
from pathlib import Path
from models.schemas import Session

logger = logging.getLogger("medical_bot")

# Storage Configuration
DATA_DIR = Path(__file__).parent.parent / "data" / "sessions"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# In-memory store
_sessions: Dict[str, Session] = {}


def _get_session_path(session_id: str) -> Path:
    return DATA_DIR / f"{session_id}.json"


def _save_session_to_disk(sid: str):
    """Internal helper to write a session to its JSON file."""
    if sid not in _sessions:
        return
    try:
        path = _get_session_path(sid)
        with open(path, "w", encoding="utf-8") as f:
            # use session.json() or session.dict() with json.dump
            # Pydantic's .json() handles datetime etc automatically
            f.write(_sessions[sid].json())
    except Exception as e:
        logger.error(f"Failed to save session {sid} to disk: {e}")


def load_sessions_from_disk():
    """Load all stored JSON sessions into the in-memory cache."""
    try:
        files = list(DATA_DIR.glob("*.json"))
        count = 0
        for f in files:
            try:
                with open(f, "r", encoding="utf-8") as f_in:
                    data = json.load(f_in)
                    sid = data.get("session_id")
                    if sid:
                        _sessions[sid] = Session(**data)
                        count += 1
            except Exception as fe:
                logger.error(f"Error loading session file {f.name}: {fe}")
        if count > 0:
            logger.info(f"Loaded {count} sessions from disk persistence")
    except Exception as e:
        logger.error(f"Disk session load error: {e}")


# Initialize: Load data once
load_sessions_from_disk()


def get_session(session_id: str) -> dict:
    """Retrieve or create a session. Syncs activity time to disk."""
    if session_id not in _sessions:
        _sessions[session_id] = Session(session_id=session_id)
        _save_session_to_disk(session_id)
        
    _sessions[session_id].last_activity = datetime.now()
    # Note: We don't necessarily save on every get_session to avoid thrashing,
    # but handle_message usually triggers updates later.
    return _sessions[session_id].dict()


def create_session(session_id: str) -> dict:
    """Explicitly create and persist a new session."""
    _sessions[session_id] = Session(session_id=session_id)
    _save_session_to_disk(session_id)
    return _sessions[session_id].dict()


def update_session(session_id: str, updated: dict):
    """Update and persist a session."""
    updated["last_activity"] = datetime.now() # Let pydantic handle conversion
    _sessions[session_id] = Session(**updated)
    _save_session_to_disk(session_id)


def reset_session(session_id: str):
    """Reset and persist session state."""
    _sessions[session_id] = Session(session_id=session_id)
    _save_session_to_disk(session_id)


def append_history(session_id: str, role: str, content: str):
    """Append a message and persist. History limit increased to 20 for more context."""
    if session_id not in _sessions:
        _sessions[session_id] = Session(session_id=session_id)

    session = _sessions[session_id]
    entry = {"role": role, "content": content}
    session.history.append(entry)
    
    # Increased limit to 20 (10 exchanges) for better multi-turn experience
    session.history = session.history[-20:]
    _save_session_to_disk(session_id)


def get_conversation_context(session_id: str, turns: int = 4) -> list:
    """Get context with larger windowing support."""
    if session_id not in _sessions:
        return []
    history = _sessions[session_id].history
    return history[-(turns * 2):]


def clear_session(session_id: str):
    """Remove session and delete its file."""
    _sessions.pop(session_id, None)
    path = _get_session_path(session_id)
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass


def cleanup_expired_sessions(max_age_days: int = 7) -> int:
    """Session cleanup now defaults to 7 days for more persistent behavior."""
    now = datetime.now()
    expired = []
    for sid, session in _sessions.items():
        last_dt = session.last_activity
        if isinstance(last_dt, str):
            last_dt = datetime.fromisoformat(last_dt)
            
        if now - last_dt > timedelta(days=max_age_days):
            expired.append(sid)
            
    for sid in expired:
        clear_session(sid)
    return len(expired)


def list_sessions() -> List[dict]:
    """Return all currently loaded sessions."""
    return [s.dict() for s in _sessions.values()]


def get_active_session_count() -> int:
    return len(_sessions)
