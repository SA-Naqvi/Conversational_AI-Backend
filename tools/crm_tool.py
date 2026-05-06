"""
CRM (Customer Relationship Management) Tool.

Stores and retrieves user/patient information keyed by session_id.
Persistence: JSON file at data/crm/users.json
Operations: get_user_info, update_user_info, create_user_profile
"""
import json
import logging
import pathlib
import asyncio
from datetime import datetime
from typing import Any, Dict, Optional

from tools.base import BaseTool

logger = logging.getLogger("medical_bot")

CRM_FILE = pathlib.Path(__file__).parent.parent / "data" / "crm" / "users.json"


def _load_crm() -> Dict[str, Any]:
    """Load the CRM JSON file, returning an empty dict if not found."""
    CRM_FILE.parent.mkdir(parents=True, exist_ok=True)
    if CRM_FILE.exists():
        try:
            return json.loads(CRM_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"CRM load error: {e}")
    return {}


def _save_crm(data: Dict[str, Any]) -> None:
    """Persist the CRM data to disk."""
    CRM_FILE.parent.mkdir(parents=True, exist_ok=True)
    CRM_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


class GetUserInfoTool(BaseTool):
    name = "get_user_info"
    description = (
        "Retrieve stored information about a patient/user by their session_id. "
        "Use this when a returning user says 'I'm back' or provides their name, "
        "to recall their name, surgery type, preferences, and past interaction notes."
    )
    parameters = {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "The unique session ID of the user to look up.",
            }
        },
        "required": ["session_id"],
    }

    async def execute(self, session_id: str, **kwargs) -> Dict[str, Any]:
        data = await asyncio.to_thread(_load_crm)
        user = data.get(session_id)
        if user:
            return {"found": True, "user": user}
        return {"found": False, "message": "No CRM record found for this session."}


class UpdateUserInfoTool(BaseTool):
    name = "update_user_info"
    description = (
        "Store or update a specific field in the patient's CRM profile. "
        "Use this when the user provides personal information such as their name, "
        "phone number, preferred appointment time, medication allergies, or "
        "other preferences that should be remembered for future sessions."
    )
    parameters = {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "The unique session ID of the user.",
            },
            "field": {
                "type": "string",
                "description": (
                    "The name of the field to update. Examples: "
                    "'name', 'phone', 'email', 'preferred_contact', "
                    "'medication_allergies', 'caregiver_name', 'notes'."
                ),
            },
            "value": {
                "type": "string",
                "description": "The value to store for that field.",
            },
        },
        "required": ["session_id", "field", "value"],
    }

    async def execute(self, session_id: str, field: str, value: str, **kwargs) -> Dict[str, Any]:
        data = await asyncio.to_thread(_load_crm)
        if session_id not in data:
            data[session_id] = {
                "created_at": datetime.utcnow().isoformat(),
                "interaction_count": 0,
            }
        data[session_id][field] = value
        data[session_id]["last_updated"] = datetime.utcnow().isoformat()
        await asyncio.to_thread(_save_crm, data)
        return {"success": True, "field": field, "value": value}


class RecordInteractionTool(BaseTool):
    name = "record_interaction"
    description = (
        "Log a note about a significant event in this session to the user's CRM history. "
        "Use this to record important milestones such as red flag events, "
        "appointment scheduling confirmations, or key patient disclosures."
    )
    parameters = {
        "type": "object",
        "properties": {
            "session_id": {"type": "string"},
            "note": {
                "type": "string",
                "description": "A brief note about the interaction to record.",
            },
        },
        "required": ["session_id", "note"],
    }

    async def execute(self, session_id: str, note: str, **kwargs) -> Dict[str, Any]:
        data = await asyncio.to_thread(_load_crm)
        if session_id not in data:
            data[session_id] = {
                "created_at": datetime.utcnow().isoformat(),
                "interaction_count": 0,
            }
        history = data[session_id].setdefault("interaction_history", [])
        history.append({"timestamp": datetime.utcnow().isoformat(), "note": note})
        data[session_id]["interaction_count"] = data[session_id].get("interaction_count", 0) + 1
        data[session_id]["last_updated"] = datetime.utcnow().isoformat()
        await asyncio.to_thread(_save_crm, data)
        return {"success": True, "note_recorded": note}


# ── Public helpers ──────────────────────────────────────────────────────────

def get_user_profile(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Synchronous helper used by the manager to pre-load user profile
    at the start of each message without making an LLM tool call.
    """
    data = _load_crm()
    return data.get(session_id)


def auto_increment_session(session_id: str) -> None:
    """Increment interaction count on each message (called by manager)."""
    try:
        data = _load_crm()
        if session_id in data:
            data[session_id]["interaction_count"] = data[session_id].get("interaction_count", 0) + 1
            data[session_id]["last_seen"] = datetime.utcnow().isoformat()
            _save_crm(data)
    except Exception:
        pass
