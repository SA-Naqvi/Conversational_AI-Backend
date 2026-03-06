"""
In-memory session management.
Sessions are ephemeral — once patient disconnects, session is lost.
"""
from models.schemas import Session, PatientState
from typing import Dict, Optional
from datetime import datetime, timedelta

# In-memory store — keyed by session_id
_sessions: Dict[str, dict] = {}


def get_session(session_id: str) -> dict:
    """Retrieve or create a session."""
    if session_id not in _sessions:
        _sessions[session_id] = Session(session_id=session_id).dict()
    _sessions[session_id]["last_activity"] = datetime.now().isoformat()
    return _sessions[session_id]


def create_session(session_id: str) -> dict:
    """Explicitly create a new session."""
    _sessions[session_id] = Session(session_id=session_id).dict()
    return _sessions[session_id]


def update_session(session_id: str, updated: dict):
    """Update a session with new data."""
    updated["last_activity"] = datetime.now().isoformat()
    _sessions[session_id] = updated


def reset_session(session_id: str):
    """Reset a session to initial state."""
    _sessions[session_id] = Session(session_id=session_id).dict()


def append_history(session_id: str, role: str, content: str):
    """Append a message to the session history. Keeps last 8 turns."""
    entry = {
        "role": role,
        "content": content,
    }
    _sessions[session_id]["history"].append(entry)
    # Keep last 8 turns (4 exchanges)
    _sessions[session_id]["history"] = _sessions[session_id]["history"][-8:]


def get_conversation_context(session_id: str, turns: int = 4) -> list:
    """Get the last N turns of conversation history."""
    if session_id not in _sessions:
        return []
    history = _sessions[session_id].get("history", [])
    return history[-(turns * 2):]


def clear_session(session_id: str):
    """Remove a session entirely."""
    _sessions.pop(session_id, None)


def cleanup_expired_sessions(max_age_hours: int = 2):
    """Remove sessions older than max_age_hours."""
    now = datetime.now()
    expired = []
    for sid, session in _sessions.items():
        last_activity = session.get("last_activity")
        if last_activity:
            try:
                if isinstance(last_activity, str):
                    last_dt = datetime.fromisoformat(last_activity)
                else:
                    last_dt = last_activity
                if now - last_dt > timedelta(hours=max_age_hours):
                    expired.append(sid)
            except (ValueError, TypeError):
                pass
    for sid in expired:
        _sessions.pop(sid, None)
    return len(expired)


def get_active_session_count() -> int:
    """Return the number of active sessions."""
    return len(_sessions)
