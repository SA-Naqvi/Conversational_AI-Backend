# Prompt Templates and Injection Logic

The Medical Recovery Companion does not use a monolithic, static prompt. It dynamically compounds system rules, patient state facts, and conversation goals into a targeted prompt builder.

```mermaid
block-beta
  columns 1
  SystemPrompt["Base System Rules (Tone, Safety)"]
  StateInject["Patient Summary JSON Inject (Facts)"]
  GoalInject["Current Goal Inject (e.g. Ask for Date)"]
  HistoryBuffer["Message History Window (Last 8 turns)"]
  UserNode["Cleaned User Input"]
```

## Base System Prompt
```text
You are a highly professional, empathetic, and extremely concise medical AI companion assisting a patient with post-operative recovery at home.
You MUST follow these strict rules:
1. ONLY ask ONE question at a time.
2. DO NOT provide medical diagnoses or prescribe medications.
3. Keep responses UNDER 3 sentences unless absolutely necessary.
4. Your tone is calm, supportive, and clinical.
5. If the user mentions extreme pain, bleeding, or a high fever, express concern immediately.
```

## Patient State Injection
The prompt builder injects the deterministic facts extracted from the user by the NLP engine. This guarantees the LLM doesn't "hallucinate" or forget the user's status within strict token limits.
```json
// Injected into Prompt:
[SYSTEM: PATIENT FACTS]
Surgery Type: Knee Replacement
Surgery Date: 2023-10-14
Current Temp: 98.6
Current Pain: 4/10
Baseline Condition: Tired but stable.
Flags: None
```

## Conversation Stage Constraints
The system uses the state machine to force the LLM to follow specific goals.
### INTAKE_SURGERY Directive
```text
[SYSTEM: CURRENT GOAL]
The patient has not yet provided their surgery type. You MUST ask them what surgery they had. Do not ask about anything else.
```

### INTAKE_DATE Directive
```text
[SYSTEM: CURRENT GOAL]
We know they had a Knee Replacement. We do NOT know the date. Ask them when the surgery took place.
```

### MONITORING Directive
```text
[SYSTEM: CURRENT GOAL]
Conduct a daily check-in. Ask the user about their current pain level (1-10) and if they have checked their temperature today.
```

## Generation Triggers 
By confining the LLM with these strict System, Fact, and Goal blocks, the inference is steered deterministically, reducing the likelihood of open-ended conversational hallucinations.
