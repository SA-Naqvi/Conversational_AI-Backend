from lm_studio.client import chat_completion_stream
from typing import AsyncGenerator

NURSE_PERSONA = """You are Nurse GPT-E, a professional post-operative recovery companion.
Your role is to:
- Guide patients through their recovery in a warm, professional tone
- Follow structured recovery protocols strictly
- Ask one question at a time during intake
- Never diagnose conditions — only provide recovery guidance
- Remind patients to contact their clinic for anything outside normal recovery

You must NEVER:
- Invent medical facts
- Provide diagnosis
- Ignore symptoms the patient reports
- Skip required intake questions

Always stay in character as Nurse GPT-E.
Respond directly with the answer.
"""

async def generate_response(
    user_message: str,
    patient_state: dict,
    conversation_stage: str,
    history: list,
    next_stage_question: str = None
) -> AsyncGenerator[str, None]:
    """Agent 2: Generate natural language response, streamed."""

    import json

    state_summary = f"""
Current Patient State:
- Name: {patient_state.get('patient_name') or 'Not yet provided'}
- Surgery: {patient_state.get('surgery_type') or 'Not yet provided'}
- Surgery Date: {patient_state.get('surgery_date') or 'Not yet provided'}
- Days Post-Op: {patient_state.get('days_post_op') or 'Unknown'}
- Pain History: {patient_state.get('pain_history') or []}
- Symptoms: {patient_state.get('symptoms') or []}
- Medications: {patient_state.get('medications') or 'Not stated'}

Conversation Stage: {conversation_stage}
"""
    if next_stage_question:
        state_summary += f"\nNext required intake question to ask: {next_stage_question}"

    system_prompt = NURSE_PERSONA + "\n\n" + state_summary

    messages = [{"role": "system", "content": system_prompt}]
    messages += history[-6:]  # Last 3 exchanges
    messages.append({"role": "user", "content": user_message})

    in_think_block = False

    async for chunk in chat_completion_stream(messages):
        # Handle streaming chunks that might contain <think> tags piece by piece
        # This is a bit tricky with streaming, so we'll collect characters and filter
        # but a simple string replacement on the yielded chunk is safest if tags are split.
        # We will keep a small buffer to catch `<think>` and `</think>` tags.
        pass # Re-written below

    buffer = ""
    async for chunk in chat_completion_stream(messages):
        buffer += chunk
        
        while True:
            if in_think_block:
                end_idx = buffer.find("</think>")
                if end_idx != -1:
                    in_think_block = False
                    buffer = buffer[end_idx + 8:]
                    continue
                else:
                    # We are in a think block and haven't found the end.
                    # We can discard the buffer up to the last 8 chars (in case </think> is cut off)
                    if len(buffer) > 8:
                        buffer = buffer[-8:]
                    break
            else:
                start_idx = buffer.find("<think>")
                if start_idx != -1:
                    # Yield everything before the think block
                    if start_idx > 0:
                        yield buffer[:start_idx]
                    in_think_block = True
                    buffer = buffer[start_idx + 7:]
                    continue
                else:
                    # We are not in a think block and no <think> tag was found.
                    # Yield the buffer, except the last 7 chars in case a <think> tag is cut off
                    if len(buffer) > 7:
                        yield buffer[:-7]
                        buffer = buffer[-7:]
                    else:
                        break

    # Yield any remaining buffer if we are not in a think block
    if buffer and not in_think_block and "<think" not in buffer:
        yield buffer
