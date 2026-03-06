"""
LLM prompt construction with token budgeting.
Builds structured prompts for Qwen3 4B GGUF with patient context.
"""
from llm_engine.token_manager import count_tokens, trim_history

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
Do not output internal reasoning tags such as <think> or similar markers.
Respond directly with the answer.
"""


def build_prompt(session: dict, user_message: str, state_guidance: str = None) -> list:
    """
    Build the complete prompt for the LLM.

    Returns a list of message dicts compatible with the OpenAI chat format:
        [{"role": "system", "content": "..."}, {"role": "user"/"assistant", ...}]
    """
    patient_state = session.get("patient_state", {})
    history = session.get("history", [])
    stage = session.get("conversation_stage", "INIT")

    # Build system prompt with patient state
    system_content = NURSE_PERSONA + "\n\n"
    system_content += format_patient_state(patient_state)
    system_content += f"\nConversation Stage: {stage}\n"

    if state_guidance:
        system_content += f"\n{state_guidance}\n"

    system_content += "\nRespond naturally and supportively. Provide guidance within scope."
    system_content += "\nAsk for clarification if needed."

    messages = [{"role": "system", "content": system_content}]

    # Add conversation history (trimmed to token budget)
    trimmed = trim_history(history, max_tokens=1200)
    messages.extend(trimmed)

    # Add current user message
    messages.append({"role": "user", "content": user_message})

    return messages


def format_patient_state(patient_state: dict) -> str:
    """Format patient state into a readable string for the system prompt."""
    lines = ["Patient Information:"]

    name = patient_state.get("patient_name")
    surgery = patient_state.get("surgery_type")
    surgery_date = patient_state.get("surgery_date")
    days_post_op = patient_state.get("days_post_op")

    lines.append(f"- Name: {name or 'Not yet provided'}")
    lines.append(f"- Surgery: {surgery or 'Not yet provided'}")
    lines.append(f"- Surgery Date: {surgery_date or 'Not yet provided'}")
    lines.append(f"- Days Post-Op: {days_post_op or 'Unknown'}")

    # Pain history
    pain_history = patient_state.get("pain_history", [])
    if pain_history:
        lines.append(f"- Pain History: {format_vital_history(pain_history)}")
        lines.append(f"- Current Pain: {pain_history[-1]}/10")
    else:
        lines.append("- Pain: Not yet reported")

    # Temperature history
    temp_history = patient_state.get("temperature_history", [])
    if temp_history:
        lines.append(f"- Temperature History: {format_vital_history(temp_history)}")
        lines.append(f"- Current Temperature: {temp_history[-1]}°F")
    else:
        lines.append("- Temperature: Not yet reported")

    # Symptoms
    symptoms = patient_state.get("symptoms", [])
    if symptoms:
        symptom_names = [
            s if isinstance(s, str) else s.get("name", str(s))
            for s in symptoms
        ]
        lines.append(f"- Symptoms: {', '.join(symptom_names)}")
    else:
        lines.append("- Symptoms: None reported")

    # Medications
    medications = patient_state.get("medications")
    if medications:
        lines.append(f"- Medications: {medications}")

    return "\n".join(lines)


def format_vital_history(history: list, max_entries: int = 5) -> str:
    """Format a vital signs history (pain or temperature) for display."""
    if not history:
        return "No data"

    # Show last N entries
    recent = history[-max_entries:]
    return " → ".join(str(v) for v in recent)
