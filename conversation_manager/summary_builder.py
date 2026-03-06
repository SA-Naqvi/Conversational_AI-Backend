"""
Deterministic patient summary generation (no LLM).
Generates formatted recovery summaries from structured patient state.
"""


def build_patient_summary(patient_state: dict) -> str:
    """
    Build a complete patient recovery summary from structured state.

    Returns a formatted summary string.
    """
    lines = []
    lines.append("📋 Your Recovery Summary")
    lines.append("=" * 30)
    lines.append("")

    # Surgery info
    surgery = patient_state.get("surgery_type", "Not provided")
    days = patient_state.get("days_post_op")
    lines.append(f"Surgery: {surgery}")
    if days is not None:
        lines.append(f"Days Post-Op: {days}")
    lines.append("")

    # Pain progress
    lines.append("Pain Progress:")
    lines.append(build_pain_summary(patient_state.get("pain_history", [])))
    lines.append("")

    # Temperature
    lines.append("Temperature:")
    lines.append(build_temp_summary(patient_state.get("temperature_history", [])))
    lines.append("")

    # Symptoms
    lines.append("Symptoms:")
    lines.append(build_symptoms_summary(patient_state.get("symptoms", [])))
    lines.append("")

    # Overall assessment
    status = calculate_recovery_status(patient_state)
    lines.append(f"Overall: {status}")
    lines.append("Continue monitoring and contact your clinic if symptoms change.")

    return "\n".join(lines)


def build_pain_summary(pain_history: list) -> str:
    """Build a summary of pain history."""
    if not pain_history:
        return "  No pain data recorded."

    first = pain_history[0]
    current = pain_history[-1]

    lines = []
    lines.append(f"  - Started at: {first}/10")
    lines.append(f"  - Current: {current}/10")

    # Determine trend
    if len(pain_history) >= 2:
        if current < first:
            lines.append("  - Trend: Improving ✓")
        elif current > first:
            lines.append("  - Trend: Worsening ⚠")
        else:
            lines.append("  - Trend: Stable →")

    return "\n".join(lines)


def build_temp_summary(temp_history: list) -> str:
    """Build a summary of temperature history."""
    if not temp_history:
        return "  No temperature data recorded."

    highest = max(temp_history)
    current = temp_history[-1]

    lines = []
    lines.append(f"  - Highest: {highest}°F")
    lines.append(f"  - Current: {current}°F")

    if current < 100.4:
        lines.append("  - Status: Normal ✓")
    elif current < 101.0:
        lines.append("  - Status: Low-grade fever ⚠")
    else:
        lines.append("  - Status: Elevated ⚠")

    return "\n".join(lines)


def build_symptoms_summary(symptoms: list) -> str:
    """Build a summary of symptoms."""
    if not symptoms:
        return "  No symptoms reported."

    lines = []
    for s in symptoms:
        if isinstance(s, str):
            lines.append(f"  - {s.replace('_', ' ').title()}")
        elif isinstance(s, dict):
            name = s.get("name", "unknown").replace("_", " ").title()
            severity = s.get("severity", "")
            lines.append(f"  - {name}" + (f" ({severity})" if severity else ""))

    return "\n".join(lines)


def calculate_recovery_status(patient_state: dict) -> str:
    """Calculate overall recovery status from patient state."""
    pain_history = patient_state.get("pain_history", [])
    temp_history = patient_state.get("temperature_history", [])
    symptoms = patient_state.get("symptoms", [])

    concerns = []

    # Check pain trend
    if pain_history and len(pain_history) >= 2:
        if pain_history[-1] > pain_history[0]:
            concerns.append("pain increasing")
        if pain_history[-1] >= 8:
            concerns.append("high pain level")

    # Check temperature
    if temp_history:
        if temp_history[-1] >= 100.4:
            concerns.append("elevated temperature")

    # Check critical symptoms
    critical_symptoms = {
        "chest_pain", "breathing_difficulty", "loss_of_consciousness",
        "bleeding", "discharge",
    }
    for s in symptoms:
        name = s if isinstance(s, str) else s.get("name", "")
        if name in critical_symptoms:
            concerns.append(f"{name.replace('_', ' ')}")

    if not concerns:
        return "Recovery is progressing as expected. ✓"
    else:
        return (
            "Some concerns noted: " + ", ".join(concerns) + ". "
            "Please discuss with your healthcare provider."
        )
