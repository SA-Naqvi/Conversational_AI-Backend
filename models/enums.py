"""
Enumerations for conversation stages, symptom severity, and extraction methods.
"""
from enum import Enum


class ConversationStage(str, Enum):
    INIT = "INIT"
    INTAKE_SURGERY = "INTAKE_SURGERY"
    INTAKE_DATE = "INTAKE_DATE"
    INTAKE_BASELINE = "INTAKE_BASELINE"
    MONITORING = "MONITORING"
    ESCALATED = "ESCALATED"
    SUMMARY = "SUMMARY"


class SymptomSeverity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class ExtractionMethod(str, Enum):
    DIRECT_NUMERIC = "direct_numeric"
    REGEX = "regex"
    RELATIVE = "relative"
    NLP_NER = "nlp_ner"
    KEYWORD = "keyword"
    DIRECT_PARSE = "direct_parse"
    UNCHANGED = "unchanged"
    FAILED = "failed"


class RedFlagCategory(str, Enum):
    CRITICAL_FEVER = "CRITICAL_FEVER"
    SEVERE_PAIN = "SEVERE_PAIN"
    UNEXPECTED_HIGH_PAIN = "UNEXPECTED_HIGH_PAIN"
    PAIN_TREND_WORSENING = "PAIN_TREND_WORSENING"
    FEVER_WITH_CHILLS = "FEVER_WITH_CHILLS"
    UNCONTROLLED_BLEEDING = "UNCONTROLLED_BLEEDING"
    INFECTION_SIGNS = "INFECTION_SIGNS"
    CRITICAL_CARDIO = "CRITICAL_CARDIO"
    CRITICAL_NEURO = "CRITICAL_NEURO"
    CRITICAL_KEYWORD = "CRITICAL_KEYWORD"
