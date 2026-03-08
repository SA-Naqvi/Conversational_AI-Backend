"""
In-memory session management using Pydantic models.
Sessions are ephemeral — once patient disconnects, session is lost.
"""
from models.schemas import Session, PatientState
from typing import Dict
from datetime import datetime, timedelta


# In-memory store — keyed by session_id, using Pydantic Session objects
_sessions: Dict[str, Session] = {}


def get_session(session_id: str) -> dict:
    """Retrieve or create a session. Returns dict for pipeline compatibility."""
    if session_id not in _sessions:
        _sessions[session_id] = Session(session_id=session_id)
    _sessions[session_id].last_activity = datetime.now()
    return _sessions[session_id].dict()


def create_session(session_id: str) -> dict:
    """Explicitly create a new session."""
    _sessions[session_id] = Session(session_id=session_id)
    return _sessions[session_id].dict()


def update_session(session_id: str, updated: dict):
    """Update a session with new data."""
    updated["last_activity"] = datetime.now().isoformat()
    # Reconstruct the Pydantic model from the updated dict
    _sessions[session_id] = Session(**updated)


def reset_session(session_id: str):
    """Reset a session to initial state."""
    _sessions[session_id] = Session(session_id=session_id)


def append_history(session_id: str, role: str, content: str):
    """Append a message to the session history. Keeps last 8 turns."""
    if session_id not in _sessions:
        _sessions[session_id] = Session(session_id=session_id)

    session = _sessions[session_id]
    entry = {"role": role, "content": content}
    session.history.append(entry)
    # Keep last 8 turns (4 exchanges)
    session.history = session.history[-8:]


def get_conversation_context(session_id: str, turns: int = 4) -> list:
    """Get the last N turns of conversation history."""
    if session_id not in _sessions:
        return []
    history = _sessions[session_id].history
    return history[-(turns * 2):]


def clear_session(session_id: str):
    """Remove a session entirely."""
    _sessions.pop(session_id, None)


def cleanup_expired_sessions(max_age_hours: int = 2) -> int:
    """Remove sessions older than max_age_hours."""
    now = datetime.now()
    expired = []
    for sid, session in _sessions.items():
        last_activity = session.last_activity
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
