"""
Pydantic models for session, patient state, and API schemas.
Simplified patient state for the NLP extraction pipeline.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ─── Core Models ───


class PatientState(BaseModel):
    # Basic Info
    patient_name: Optional[str] = None
    surgery_type: Optional[str] = None
    surgery_date: Optional[str] = None          # YYYY-MM-DD string
    days_post_op: Optional[int] = None

    # Vital Signs History (simple value lists for NLP pipeline)
    pain_history: List[int] = []                 # e.g. [7, 6, 4]
    temperature_history: List[float] = []        # e.g. [98.6, 99.1]

    # Symptoms (simple list of symptom name strings)
    symptoms: List[str] = []                     # e.g. ["swelling", "redness"]

    # Medications
    medications: Optional[str] = None

    # Red Flags
    red_flag_detected: bool = False
    red_flag_details: Optional[Dict[str, Any]] = None
    escalation_time: Optional[datetime] = None

    # Recovery Analytics
    recovery_score: Optional[float] = None        # 0.0-1.0 overall score
    pain_trend: Optional[str] = None              # "improving", "worsening", "stable"
    fever_status: Optional[str] = None            # "normal", "low_grade", "elevated", "critical"


class SessionMetadata(BaseModel):
    total_turns: int = 0
    red_flags_count: int = 0
    extraction_confidence: Dict[str, float] = {}


class Session(BaseModel):
    session_id: str
    title: str = "New Chat"
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    conversation_stage: str = "INIT"
    patient_state: PatientState = Field(default_factory=PatientState)
    history: List[dict] = []  # {"role": "user"/"assistant", "content": "..."}
    metadata: SessionMetadata = Field(default_factory=SessionMetadata)


# ─── API Schemas ───


class ChatMessage(BaseModel):
    session_id: str
    message: str


class ValidationResult(BaseModel):
    is_valid: bool
    sanitized_text: str = ""
    validation_errors: List[str] = []
    flags: Dict[str, bool] = {}


class ExtractionResult(BaseModel):
    value: Any = None
    confidence: float = 0.0
    method: str = "failed"
    raw_match: Optional[str] = None
