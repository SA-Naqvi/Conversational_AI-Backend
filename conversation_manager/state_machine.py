STAGES = [
    "INIT",
    "INTAKE_SURGERY",
    "INTAKE_DATE",
    "INTAKE_BASELINE",
    "MONITORING",
    "ESCALATED",
    "SUMMARY"
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

def next_stage(current_stage: str, patient_state: dict) -> str:
    """Determine the next stage based on what data is still missing."""
    if current_stage == "ESCALATED":
        return "ESCALATED"  # Locked

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
