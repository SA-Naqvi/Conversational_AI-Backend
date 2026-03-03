from models.schemas import Session, PatientState
from typing import Dict

# In-memory store — keyed by session_id
_sessions: Dict[str, dict] = {}

def get_session(session_id: str) -> dict:
    if session_id not in _sessions:
        _sessions[session_id] = Session(session_id=session_id).dict()
    return _sessions[session_id]

def update_session(session_id: str, updated: dict):
    _sessions[session_id] = updated

def reset_session(session_id: str):
    _sessions[session_id] = Session(session_id=session_id).dict()

def append_history(session_id: str, role: str, content: str):
    _sessions[session_id]["history"].append({"role": role, "content": content})
    # Keep last 8 turns (4 exchanges)
    _sessions[session_id]["history"] = _sessions[session_id]["history"][-8:]
