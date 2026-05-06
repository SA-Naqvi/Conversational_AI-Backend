"""
Main Conversation Manager — orchestrates the full processing pipeline.

Pipeline:
1.  Input validation & sanitization
2.  Retrieve/create session
3.  CRM profile lookup (pre-load returning user data)
4.  Noise filtering
5.  NLP-based extraction + Red flag check (PARALLEL)
6.  RAG retrieval (parallel with step 5)
7.  State machine transition
8.  Build LLM prompt (patient state + CRM context + RAG context)
9.  LLM call with tools:
      a. Streaming with tool detection
      b. If tool_calls detected → execute tools → stream follow-up
      c. Otherwise → stream text directly
10. History & state update
"""
import os
import time
import json
import asyncio
import logging
from datetime import datetime
from typing import AsyncGenerator, Optional

from conversation_manager import input_validator
from conversation_manager import noise_filter
from conversation_manager import patient_state_updater
from conversation_manager.red_flag_engine import check_red_flags
from conversation_manager.prompt_builder import build_prompt
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
from llm_engine.qwen_interface import get_llm
from rag.retriever import retrieve, format_context
from tools.tool_orchestrator import get_orchestrator
from tools.crm_tool import get_user_profile, auto_increment_session

logger = logging.getLogger("medical_bot")

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

# Maximum number of tool call rounds to prevent infinite loops
MAX_TOOL_ITERATIONS = 3


def _format_crm_context(profile: Optional[dict]) -> Optional[str]:
    """Format CRM profile into a brief system prompt snippet."""
    if not profile:
        return None
    lines = ["[RETURNING USER PROFILE]"]
    for field in ("name", "phone", "email", "preferred_contact",
                  "medication_allergies", "caregiver_name", "notes"):
        val = profile.get(field)
        if val:
            lines.append(f"  {field.replace('_', ' ').title()}: {val}")
    interactions = profile.get("interaction_count", 0)
    if interactions:
        lines.append(f"  Previous interactions: {interactions}")
    history = profile.get("interaction_history", [])
    if history:
        last = history[-1]
        lines.append(f"  Last note: {last.get('note', '')} ({last.get('timestamp', '')[:10]})")
    lines.append("[END USER PROFILE]")
    return "\n".join(lines)


def _format_rag_sources(rag_chunks: list) -> Optional[str]:
    """
    Build a user-visible source attribution block for retrieved RAG chunks.
    """
    if not rag_chunks:
        return None

    lines = ["\n\nSources (RAG):"]
    for i, chunk in enumerate(rag_chunks[:4], 1):
        source = chunk.get("source", "unknown")
        chunk_index = chunk.get("chunk_index", "?")
        score = chunk.get("score")
        if isinstance(score, (float, int)):
            lines.append(f"{i}. {source} (chunk {chunk_index}, score {score:.3f})")
        else:
            lines.append(f"{i}. {source} (chunk {chunk_index})")
    return "\n".join(lines)


async def handle_message(session_id: str, user_message: str, eval_mode: bool = False) -> AsyncGenerator[str, None]:
    """
    Main entry point. Orchestrates the full processing pipeline.
    Yields streaming response chunks.
    """
    start_time = time.time()
    _eval_metrics = {
        "ttft_ms": None, "total_ms": None, "retrieval_ms": None,
        "tool_ms": {}, "red_flag_triggered": False,
        "state_transition": None, "tokens_generated": 0,
    }
    _first_token_sent = False

    # ── STEP 1: Input Validation ──────────────────────────────────────────
    validation = input_validator.validate_input(user_message, session_id, skip_rate_limit=eval_mode)
    if not validation["is_valid"]:
        error_msg = (
            "I couldn't process your message:\n"
            + "\n".join(f"- {e}" for e in validation["validation_errors"])
            + "\nPlease try again."
        )
        yield error_msg
        return

    sanitized = validation["sanitized_text"]

    # ── STEP 2: Retrieve or create session ───────────────────────────────
    session = get_session(session_id)

    if session.get("title") == "New Chat":
        clean_text = user_message.strip().replace("\n", " ")
        title = clean_text[:30] + ("..." if len(clean_text) > 30 else "")
        session["title"] = title
        update_session(session_id, session)

    patient_state = session["patient_state"]
    stage = session["conversation_stage"]

    # ── ESCALATED check ──────────────────────────────────────────────────
    if stage == "ESCALATED":
        yield EMERGENCY_MESSAGE
        return

    # ── STEP 3: CRM profile lookup ────────────────────────────────────────
    crm_profile = get_user_profile(session_id)
    crm_context = _format_crm_context(crm_profile)
    auto_increment_session(session_id)

    # ── STEP 4: Noise filtering ───────────────────────────────────────────
    filtered_text = noise_filter.filter_text(sanitized, context=session)

    # ── STEP 5+6: Parallel NLP extraction, Red-flag check, and RAG ───────
    extraction = {}
    is_red_flag = False
    red_flag_reason = ""
    rag_chunks = []

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

    async def _run_rag():
        nonlocal rag_chunks
        _rag_start = time.time()
        try:
            rag_chunks = await retrieve(filtered_text, top_k=4)
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
            rag_chunks = []
        _eval_metrics["retrieval_ms"] = round((time.time() - _rag_start) * 1000, 2)

    # Run extraction, red-flag check, and RAG retrieval all in parallel
    await asyncio.gather(_run_extraction(), _run_red_flag(), _run_rag())

    patient_state = session["patient_state"]

    if extraction:
        avg_confidence = _avg_confidence(extraction)
        log_extraction(session_id, extraction, avg_confidence)

    # ── Handle Red Flags ──────────────────────────────────────────────────
    if is_red_flag:
        patient_state["red_flag_detected"] = True
        session["conversation_stage"] = "ESCALATED"
        append_history(session_id, "user", user_message)
        append_history(session_id, "assistant", EMERGENCY_MESSAGE)
        update_session(session_id, session)
        log_red_flag(session_id, red_flag_reason, {"message": filtered_text})
        _eval_metrics["red_flag_triggered"] = True
        _eval_metrics["state_transition"] = f"{stage} -> ESCALATED"
        _eval_metrics["total_ms"] = round((time.time() - start_time) * 1000, 2)
        yield EMERGENCY_MESSAGE
        if eval_mode:
            import json as _json
            yield _json.dumps({"type": "metrics", "data": _eval_metrics})
        return

    is_red_flag_post, red_flag_reason_post = check_red_flags(filtered_text, patient_state)
    if is_red_flag_post:
        patient_state["red_flag_detected"] = True
        session["conversation_stage"] = "ESCALATED"
        append_history(session_id, "user", user_message)
        append_history(session_id, "assistant", EMERGENCY_MESSAGE)
        update_session(session_id, session)
        log_red_flag(session_id, red_flag_reason_post, {"message": filtered_text})
        _eval_metrics["red_flag_triggered"] = True
        _eval_metrics["state_transition"] = f"{stage} -> ESCALATED"
        _eval_metrics["total_ms"] = round((time.time() - start_time) * 1000, 2)
        yield EMERGENCY_MESSAGE
        if eval_mode:
            import json as _json
            yield _json.dumps({"type": "metrics", "data": _eval_metrics})
        return

    # ── STEP 7: State machine transition ──────────────────────────────────
    old_stage = stage
    new_stage = next_stage(stage, patient_state)
    session["conversation_stage"] = new_stage

    _eval_metrics["state_transition"] = f"{old_stage} -> {new_stage}"
    if new_stage != old_stage:
        log_state_transition(session_id, old_stage, new_stage)

    next_question = STAGE_QUESTIONS.get(new_stage) if new_stage != stage else None

    if new_stage in ["INTAKE_SURGERY", "INTAKE_DATE", "INTAKE_BASELINE"]:
        if not is_intake_requirement_met(new_stage, patient_state):
            pass

    # ── STEP 8: Build LLM prompt ──────────────────────────────────────────
    state_guidance = get_state_prompt_guidance(new_stage)
    if next_question:
        state_guidance = (state_guidance or "") + f"\nNext required intake question: {next_question}"

    rag_context = format_context(rag_chunks) if rag_chunks else None

    prompt_messages = build_prompt(
        session, sanitized, state_guidance,
        rag_context=rag_context,
        crm_context=crm_context,
    )
    # ── STEP 9: LLM call with tools + streaming ───────────────────────────
    append_history(session_id, "user", user_message)

    full_response = ""
    llm = get_llm()
    orchestrator = get_orchestrator()
    tools_schema = orchestrator.get_tools_schema()

    # Inject session_id into user message context for tools that need it
    # (we prepend it silently in the last user message)
    working_messages = prompt_messages.copy()
    # Attach session_id as a hidden system note so tools can use it
    working_messages[-1] = {
        "role": "user",
        "content": sanitized + f"\n\n[session_id: {session_id}]",
    }

    try:
        tool_iteration = 0
        tool_calls_this_turn = None

        async for chunk in llm.generate_stream_with_tools(working_messages, tools_schema):
            if isinstance(chunk, dict) and chunk.get("type") == "tool_calls":
                # Model wants to call tools
                tool_calls_this_turn = chunk["calls"]
            elif isinstance(chunk, str):
                if not _first_token_sent:
                    _first_token_sent = True
                    _eval_metrics["ttft_ms"] = round((time.time() - start_time) * 1000, 2)
                _eval_metrics["tokens_generated"] += 1
                full_response += chunk
                yield chunk

        # ── Tool call loop ────────────────────────────────────────────────
        while tool_calls_this_turn and tool_iteration < MAX_TOOL_ITERATIONS:
            tool_iteration += 1
            call_names = [tc["function"]["name"] for tc in tool_calls_this_turn]
            logger.info(f"Tool calls requested (iter {tool_iteration}): {call_names}")
            _tool_start = time.time()

            # Yield a brief status indicator to the user
            tool_indicator = f"\n\n🔧 *Using tools: {', '.join(call_names)}...*\n\n"
            yield tool_indicator
            full_response += tool_indicator

            # Build the assistant tool-call message
            working_messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls_this_turn,
            })

            # Execute all tool calls concurrently
            tool_result_messages = await orchestrator.execute_tool_calls(tool_calls_this_turn)
            working_messages.extend(tool_result_messages)
            for _cn in call_names:
                _eval_metrics["tool_ms"][_cn] = round((time.time() - _tool_start) * 1000, 2)

            # Reset for next iteration
            tool_calls_this_turn = None

            # Stream the follow-up response (with tools still available)
            async for chunk in llm.generate_stream_with_tools(working_messages, tools_schema):
                if isinstance(chunk, dict) and chunk.get("type") == "tool_calls":
                    tool_calls_this_turn = chunk["calls"]
                elif isinstance(chunk, str):
                    if not _first_token_sent:
                        _first_token_sent = True
                        _eval_metrics["ttft_ms"] = round((time.time() - start_time) * 1000, 2)
                    _eval_metrics["tokens_generated"] += 1
                    full_response += chunk
                    yield chunk

    except Exception as e:
        log_error(session_id, "llm_error", str(e))
        err_msg = (
            "\n\n⚠️ System Alert: The AI took too long to respond or encountered an error. "
            "Please try again."
        )
        full_response += err_msg
        yield err_msg

    # Append visible citation block so users can see where RAG context came from.
    rag_sources_block = _format_rag_sources(rag_chunks)
    if rag_sources_block:
        full_response += rag_sources_block
        yield rag_sources_block

    # ── STEP 10: Update history & state ──────────────────────────────────
    append_history(session_id, "assistant", full_response)
    session["patient_state"] = patient_state
    update_session(session_id, session)

    duration_ms = (time.time() - start_time) * 1000
    _eval_metrics["total_ms"] = round(duration_ms, 2)
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

    # ── Eval mode: yield metrics as final frame ───────────────────────────
    if eval_mode:
        import json as _json
        yield _json.dumps({"type": "metrics", "data": _eval_metrics})


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
