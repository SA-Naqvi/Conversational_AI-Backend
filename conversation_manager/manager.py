"""
Main Conversation Manager — orchestrates the full processing pipeline.

Pipeline:
1. Input validation & sanitization
2. Retrieve/create session
3. Noise filtering
4. NLP-based extraction + Red flag check (PARALLEL)
5. State machine transition
6. Intent routing (skip LLM when possible)
7. LLM prompt building (with dynamic token budget)
8. LLM response generation (streaming, buffered)
9. History & state update
"""
import os
import time
import asyncio
from datetime import datetime
from typing import AsyncGenerator

from conversation_manager import input_validator
from conversation_manager import noise_filter
from conversation_manager import patient_state_updater
from conversation_manager.red_flag_engine import check_red_flags
from conversation_manager.prompt_builder import build_prompt, get_max_tokens
from conversation_manager.session_store import (
    get_session, update_session, append_history,
)
from conversation_manager.state_machine import (
    next_stage, STAGE_QUESTIONS,
    get_state_prompt_guidance, is_intake_requirement_met, should_transition,
)
from conversation_manager.logging_manager import (
    log_extraction, log_red_flag, log_message_processing,
    log_state_transition, log_error, log_generation_metrics,
)
from llm_engine.llama_interface import get_llm

CLINIC_PHONE = os.getenv("CLINIC_PHONE", "XXX-XXX-XXXX")

EMERGENCY_MESSAGE = (
    f"🚨 I am concerned about your symptoms.\n\n"
    f"Based on your report, you need immediate medical attention.\n\n"
    f"Please contact your clinic immediately or seek urgent medical care:\n"
    f"- Call: {CLINIC_PHONE}\n"
    f"- Go to: Nearest Emergency Room\n"
    f"- Call 911 if: Severe chest pain, difficulty breathing, loss of consciousness\n\n"
    f"Your safety is my priority. Medical professionals need to evaluate you now.\n\n"
    f"[This conversation will now be locked for your safety]"
)


async def handle_message(session_id: str, user_message: str) -> AsyncGenerator[str, None]:
    """
    Main entry point. Orchestrates the full processing pipeline.
    Yields streaming response chunks.
    """
    start_time = time.time()

    # --- STEP 1: Input Validation ---
    validation = input_validator.validate_input(user_message, session_id)
    if not validation["is_valid"]:
        error_msg = (
            "I couldn't process your message:\n"
            + "\n".join(f"- {e}" for e in validation["validation_errors"])
            + "\nPlease try again."
        )
        yield error_msg
        return

    sanitized = validation["sanitized_text"]

    # --- STEP 2: Retrieve or create session ---
    session = get_session(session_id)
    patient_state = session["patient_state"]
    stage = session["conversation_stage"]

    # --- ESCALATED: Conversation locked ---
    if stage == "ESCALATED":
        yield EMERGENCY_MESSAGE
        return

    # --- STEP 3: Noise filtering ---
    filtered_text = noise_filter.filter_text(sanitized, context=session)

    # --- STEP 4 & 5: Parallel NLP extraction + Red flag check ---
    extraction = {}
    is_red_flag = False
    red_flag_reason = ""

    async def _run_extraction():
        nonlocal extraction
        try:
            extraction = patient_state_updater.extract_all(filtered_text, session)
            patient_state_updater.update_patient_state(session, extraction)
        except Exception as e:
            log_error(session_id, "extraction_error", str(e))

    async def _run_red_flag():
        nonlocal is_red_flag, red_flag_reason
        is_red_flag, red_flag_reason = check_red_flags(filtered_text, patient_state)

    # Run extraction and red-flag check in parallel
    await asyncio.gather(_run_extraction(), _run_red_flag())

    # Update patient_state reference after extraction
    patient_state = session["patient_state"]

    # Log extraction
    if extraction:
        avg_confidence = _avg_confidence(extraction)
        log_extraction(session_id, extraction, avg_confidence)

    # --- Handle Red Flags ---
    if is_red_flag:
        patient_state["red_flag_detected"] = True
        session["conversation_stage"] = "ESCALATED"

        append_history(session_id, "user", user_message)
        append_history(session_id, "assistant", EMERGENCY_MESSAGE)
        update_session(session_id, session)

        log_red_flag(session_id, red_flag_reason, {"message": filtered_text})

        yield EMERGENCY_MESSAGE
        return

    # Also re-check red flags after extraction updated state
    is_red_flag_post, red_flag_reason_post = check_red_flags(filtered_text, patient_state)
    if is_red_flag_post:
        patient_state["red_flag_detected"] = True
        session["conversation_stage"] = "ESCALATED"

        append_history(session_id, "user", user_message)
        append_history(session_id, "assistant", EMERGENCY_MESSAGE)
        update_session(session_id, session)

        log_red_flag(session_id, red_flag_reason_post, {"message": filtered_text})

        yield EMERGENCY_MESSAGE
        return

    # --- STEP 6: State machine transition ---
    old_stage = stage
    new_stage = next_stage(stage, patient_state)
    session["conversation_stage"] = new_stage

    if new_stage != old_stage:
        log_state_transition(session_id, old_stage, new_stage)

    # Determine if we need to prompt next intake question
    next_question = STAGE_QUESTIONS.get(new_stage) if new_stage != stage else None

    # --- STEP 7: Check intake requirements ---
    if new_stage in ["INTAKE_SURGERY", "INTAKE_DATE", "INTAKE_BASELINE"]:
        if not is_intake_requirement_met(new_stage, patient_state):
            # Still need info for this stage — guidance will be in the prompt
            pass

    # --- STEP 8: Build LLM prompt (with dynamic token budget) ---
    state_guidance = get_state_prompt_guidance(new_stage)
    if next_question:
        state_guidance = (state_guidance or "") + f"\nNext required intake question: {next_question}"

    prompt_messages = build_prompt(session, sanitized, state_guidance)
    max_tokens = get_max_tokens(new_stage)

    # --- STEP 10: LLM response generation (streaming) ---
    append_history(session_id, "user", user_message)

    full_response = ""
    llm = get_llm()

    try:
        async for chunk in llm.generate_response_stream(
            prompt_messages, max_tokens=max_tokens
        ):
            full_response += chunk
            yield chunk
    except Exception as e:
        log_error(session_id, "llm_error", str(e))
        err_msg = (
            "\n\n⚠️ System Alert: The medical AI took too long to respond "
            "or failed to connect. Please try your message again."
        )
        full_response += err_msg
        yield err_msg

    # --- STEP 11: Update history & state ---
    append_history(session_id, "assistant", full_response)
    session["patient_state"] = patient_state
    update_session(session_id, session)

    # Log processing time and generation metrics
    duration_ms = (time.time() - start_time) * 1000
    log_message_processing(session_id, duration_ms, new_stage)

    metrics = llm.last_metrics
    if metrics:
        log_generation_metrics(
            session_id=session_id,
            prompt_tokens=metrics.get("prompt_tokens", 0),
            response_tokens=metrics.get("completion_tokens", 0),
            generation_time=metrics.get("generation_time", 0),
            tokens_per_second=metrics.get("tokens_per_second", 0),
            stage=new_stage,
        )


def _avg_confidence(extraction: dict) -> float:
    """Calculate average confidence across all extraction results."""
    confidences = []
    for key, val in extraction.items():
        if isinstance(val, dict) and "confidence" in val:
            confidences.append(val["confidence"])
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict) and "confidence" in item:
                    confidences.append(item["confidence"])
    return sum(confidences) / len(confidences) if confidences else 0.0
