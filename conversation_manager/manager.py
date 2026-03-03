import os
import httpx
from agents.memory_agent import extract_memory
from agents.response_agent import generate_response
from conversation_manager.session_store import (
    get_session, update_session, append_history
)
from conversation_manager.state_machine import next_stage, STAGE_QUESTIONS
from conversation_manager.red_flag_engine import check_red_flags
from typing import AsyncGenerator

CLINIC_PHONE = os.getenv("CLINIC_PHONE", "XXX-XXX-XXXX")

EMERGENCY_MESSAGE = (
    f"⚠️ I am concerned about your symptoms. "
    f"Please contact your clinic immediately at {CLINIC_PHONE} "
    f"or go to the nearest emergency department. "
    f"Do not wait."
)

async def handle_message(session_id: str, user_message: str) -> AsyncGenerator[str, None]:
    """Main entry point. Orchestrates Agent 1 → safety check → Agent 2."""

    session = get_session(session_id)
    patient_state = session["patient_state"]
    stage = session["conversation_stage"]

    # --- ESCALATED: Conversation locked ---
    if stage == "ESCALATED":
        yield EMERGENCY_MESSAGE
        return

    # --- STEP 1: Agent 1 — Extract memory from user message ---
    try:
        extracted = await extract_memory(user_message, patient_state)
        _apply_extraction(patient_state, extracted)
    except (httpx.ReadTimeout, httpx.ConnectError) as e:
        print(f"Memory extraction failed due to LLM error: {e}")
        pass  # non-fatal, proceed with existing state

    # --- STEP 2: Deterministic Red Flag Check ---
    is_red_flag, reason = check_red_flags(user_message, patient_state)
    if is_red_flag:
        patient_state["red_flag_detected"] = True
        session["conversation_stage"] = "ESCALATED"
        append_history(session_id, "user", user_message)
        append_history(session_id, "assistant", EMERGENCY_MESSAGE)
        update_session(session_id, session)
        yield EMERGENCY_MESSAGE
        return

    # --- STEP 3: Advance stage ---
    new_stage = next_stage(stage, patient_state)
    session["conversation_stage"] = new_stage

    # Determine if we need to prompt next intake question
    next_question = STAGE_QUESTIONS.get(new_stage) if new_stage != stage else None

    # --- STEP 4: Agent 2 — Generate response (streaming) ---
    append_history(session_id, "user", user_message)

    full_response = ""
    try:
        async for chunk in generate_response(
            user_message=user_message,
            patient_state=patient_state,
            conversation_stage=new_stage,
            history=session["history"],
            next_stage_question=next_question
        ):
            full_response += chunk
            yield chunk
    except (httpx.ReadTimeout, httpx.ConnectError) as e:
        print(f"Response generation failed due to LLM error: {e}")
        err_msg = "\n\n⚠️ System Alert: The medical AI took too long to respond or failed to connect. Please try your message again."
        full_response += err_msg
        yield err_msg

    append_history(session_id, "assistant", full_response)
    session["patient_state"] = patient_state
    update_session(session_id, session)


def _apply_extraction(patient_state: dict, extracted: dict):
    """Apply Agent 1 output to patient state."""
    if extracted.get("patient_name"):
        patient_state["patient_name"] = extracted["patient_name"]
    if extracted.get("surgery_type"):
        patient_state["surgery_type"] = extracted["surgery_type"]
    if extracted.get("surgery_date"):
        patient_state["surgery_date"] = extracted["surgery_date"]
    if extracted.get("pain_level") is not None:
        patient_state["pain_history"].append(extracted["pain_level"])
    if extracted.get("temperature") is not None:
        patient_state["temperature_history"].append(extracted["temperature"])
    if extracted.get("new_symptoms"):
        for s in extracted["new_symptoms"]:
            if s not in patient_state["symptoms"]:
                patient_state["symptoms"].append(s)
    if extracted.get("medications"):
        patient_state["medications"] = extracted["medications"]
