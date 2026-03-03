from pydantic import BaseModel
from typing import Optional, List

class ChatMessage(BaseModel):
    session_id: str
    message: str

class PatientState(BaseModel):
    patient_name: Optional[str] = None
    surgery_type: Optional[str] = None
    surgery_date: Optional[str] = None
    days_post_op: Optional[int] = None
    pain_history: List[int] = []
    temperature_history: List[float] = []
    symptoms: List[str] = []
    medications: Optional[str] = None
    red_flag_detected: bool = False

class Session(BaseModel):
    session_id: str
    conversation_stage: str = "INIT"
    patient_state: PatientState = PatientState()
    history: List[dict] = []  # {"role": "user"/"assistant", "content": "..."}
