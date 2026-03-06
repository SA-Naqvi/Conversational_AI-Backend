"""
Conversation state machine controller.
Controls strict conversation workflow. Prevents skipping steps.

States:
    INIT → INTAKE_SURGERY → INTAKE_DATE → INTAKE_BASELINE → MONITORING
    MONITORING → ESCALATED (red flag)
    MONITORING → SUMMARY (user requests)
"""

STAGES = [
    "INIT",
    "INTAKE_SURGERY",
    "INTAKE_DATE",
    "INTAKE_BASELINE",
    "MONITORING",
    "ESCALATED",
    "SUMMARY",
]

STAGE_QUESTIONS = {
    "INIT": None,
    "INTAKE_SURGERY": "What type of surgery did you have?",
    "INTAKE_DATE": "When was your surgery performed?",
    "INTAKE_BASELINE": "On a scale from 0 to 10, what is your current pain level?",
    "MONITORING": None,
    "ESCALATED": None,
    "SUMMARY": None,
}

# Valid transitions
VALID_TRANSITIONS = {
    "INIT": {"INTAKE_SURGERY"},
    "INTAKE_SURGERY": {"INTAKE_DATE"},
    "INTAKE_DATE": {"INTAKE_BASELINE"},
    "INTAKE_BASELINE": {"MONITORING"},
    "MONITORING": {"ESCALATED", "SUMMARY"},
    "ESCALATED": set(),  # Terminal state
    "SUMMARY": {"MONITORING"},
}

# Guidance prompts for each state
STATE_GUIDANCE = {
    "INIT": (
        "The patient has just connected. Greet them warmly and ask what type "
        "of surgery they had. Keep it brief and welcoming."
    ),
    "INTAKE_SURGERY": (
        "We need to collect the patient's surgery type. If they haven't "
        "provided it yet, ask: 'What type of surgery did you have?' "
        "Be patient and supportive."
    ),
    "INTAKE_DATE": (
        "We have the surgery type. Now we need the surgery date. "
        "Ask: 'When was your surgery performed?' Accept various date formats."
    ),
    "INTAKE_BASELINE": (
        "We have surgery type and date. Now collect baseline vitals. "
        "Ask: 'On a scale of 0 to 10, what is your current pain level?' "
        "Also ask about temperature if appropriate."
    ),
    "MONITORING": (
        "The patient is in the monitoring phase. Provide supportive recovery "
        "guidance. Track symptoms, pain levels, and temperature changes. "
        "Remind them of healthy recovery milestones. Ask follow-up questions "
        "about their recovery progress."
    ),
    "ESCALATED": None,  # Handled by emergency message, LLM not called
    "SUMMARY": (
        "The patient wants a recovery summary. Provide a comprehensive "
        "overview of their recovery progress."
    ),
}


def next_stage(current_stage: str, patient_state: dict) -> str:
    """Determine the next stage based on what data is still missing."""
    if current_stage == "ESCALATED":
        return "ESCALATED"  # Locked — terminal state

    if current_stage == "INIT":
        return "INTAKE_SURGERY"

    if current_stage == "INTAKE_SURGERY":
        if patient_state.get("surgery_type"):
            return "INTAKE_DATE"
        return "INTAKE_SURGERY"

    if current_stage == "INTAKE_DATE":
        if patient_state.get("surgery_date"):
            return "INTAKE_BASELINE"
        return "INTAKE_DATE"

    if current_stage == "INTAKE_BASELINE":
        if patient_state.get("pain_history"):
            return "MONITORING"
        return "INTAKE_BASELINE"

    return current_stage  # Stay in MONITORING by default


def can_transition(from_state: str, to_state: str) -> bool:
    """Check if a transition from one state to another is valid."""
    return to_state in VALID_TRANSITIONS.get(from_state, set())


def get_next_state(current_state: str, context: dict) -> str:
    """Alias for next_stage for compatibility with the plan."""
    return next_stage(current_state, context)


def get_state_prompt_guidance(state: str) -> str:
    """Get the prompt guidance string for a given state."""
    return STATE_GUIDANCE.get(state, "")


def is_intake_complete(patient_state: dict) -> bool:
    """Check if all required intake information has been collected."""
    return bool(
        patient_state.get("surgery_type")
        and patient_state.get("surgery_date")
        and patient_state.get("pain_history")
    )


def is_intake_requirement_met(state: str, patient_state: dict) -> bool:
    """Check if the current intake stage's requirement is met."""
    if state == "INTAKE_SURGERY":
        return bool(patient_state.get("surgery_type"))
    elif state == "INTAKE_DATE":
        return bool(patient_state.get("surgery_date"))
    elif state == "INTAKE_BASELINE":
        return bool(patient_state.get("pain_history"))
    return True


def should_transition(current_state: str, patient_state: dict) -> bool:
    """Check if a transition should happen from the current state."""
    candidate = next_stage(current_state, patient_state)
    return candidate != current_state


def block_after_escalation(state: str) -> bool:
    """Check if the conversation should be blocked (escalated)."""
    return state == "ESCALATED"
