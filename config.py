"""
Centralized configuration & constants for the Medical Recovery Companion.
"""
import os
from datetime import timedelta

# ============ LLM Configuration ============
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "qwen/qwen3-4b")

# llama.cpp server configuration
LLAMA_CPP_BASE_URL = os.getenv("LLAMA_CPP_BASE_URL", "http://localhost:8080")
LLAMA_CPP_MODEL = os.getenv("LLAMA_CPP_MODEL", "qwen3-4b")

LLM_CONTEXT_LENGTH = 4096
LLM_MAX_TOKENS = 512
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.9
LLM_TOP_K = 15
LLM_REPEAT_PENALTY = 1.1
LLM_TIMEOUT_SECONDS = 60

# ============ Dynamic Token Budgets (per stage) ============
STAGE_MAX_TOKENS = {
    "INIT": 120,
    "INTAKE_SURGERY": 80,
    "INTAKE_DATE": 80,
    "INTAKE_BASELINE": 120,
    "MONITORING": 120,
    "SUMMARY": 200,
    "ESCALATED": 80,
}
DEFAULT_STAGE_MAX_TOKENS = 120

# ============ Streaming ============
STREAM_BUFFER_SIZE = int(os.getenv("STREAM_BUFFER_SIZE", "6"))

# ============ Prompt Caching ============
PROMPT_CACHE_TTL = int(os.getenv("PROMPT_CACHE_TTL", "300"))  # seconds

# ============ Session Configuration ============
SESSION_TTL = timedelta(hours=2)
MAX_SESSIONS = 100
SESSION_CLEANUP_INTERVAL = 300  # seconds
MAX_HISTORY_TURNS = 8  # 4 exchanges (user + assistant)
MAX_HISTORY_TOKENS = 600  # token-based history budget

# ============ Input Validation ============
MAX_INPUT_LENGTH = 2000
MIN_INPUT_LENGTH = 1
RATE_LIMIT_MESSAGES_PER_MINUTE = 30
RATE_LIMIT_WINDOW = 60  # seconds
FORBIDDEN_CHARS = ['<', '>', '&']

# ============ Patient State Configuration ============
MAX_PAIN_HISTORY_LENGTH = 100
MAX_TEMP_HISTORY_LENGTH = 100
MAX_SYMPTOM_HISTORY_LENGTH = 50

# ============ Red Flag Thresholds ============
RED_FLAG_TEMP_CRITICAL = 102.0  # °F
RED_FLAG_TEMP_HIGH = 101.0  # °F (concerning at day 7+)
RED_FLAG_TEMP_FEVER = 100.4  # °F (low-grade fever)
RED_FLAG_PAIN_CRITICAL = 9
RED_FLAG_PAIN_HIGH = 8

# ============ Extraction Confidence Thresholds ============
CONFIDENCE_HIGH = 0.80
CONFIDENCE_MEDIUM = 0.50
CONFIDENCE_LOW = 0.30

# ============ Noise Filter Settings ============
MAX_CHARS_AFTER_FILTER = 500
AGGRESSIVE_FILTER_THRESHOLD = 1500  # chars; triggers aggressive mode

# ============ Prompt Token Budgets ============
TOKEN_BUDGET_SYSTEM = 500
TOKEN_BUDGET_STATE = 400
TOKEN_BUDGET_HISTORY = 1200
TOKEN_BUDGET_USER = 400
TOKEN_BUDGET_BUFFER = 1596

# ============ Logging ============
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = "logs/medical_bot.log"

# ============ Clinic Contact ============
CLINIC_PHONE = os.getenv("CLINIC_PHONE", "XXX-XXX-XXXX")
ESCALATION_EMERGENCY_PHONE = "911"

# ============ Server ============
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:8080")
