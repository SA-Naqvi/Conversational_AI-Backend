"""
Pytest configuration and shared fixtures for the eval suite.
"""
import os
import sys
import json
import asyncio
import subprocess
import time
import pytest
import httpx

# Add backend root to path so we can import app modules directly
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHATBOT_HOST = os.getenv("CHATBOT_HOST", "localhost")
CHATBOT_PORT = int(os.getenv("CHATBOT_PORT", "8000"))
CHATBOT_URL = os.getenv("CHATBOT_URL", f"http://{CHATBOT_HOST}:{CHATBOT_PORT}")
WS_URL = f"ws://{CHATBOT_HOST}:{CHATBOT_PORT}"

DATA_DIR = os.path.join(os.path.dirname(__file__), "tests", "data")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")


def pytest_addoption(parser):
    parser.addoption("--skip-llm", action="store_true", default=False,
                     help="Skip tests that require the LLM server")
    parser.addoption("--full-perf", action="store_true", default=False,
                     help="Run full 30-trial performance tests instead of 5")
    parser.addoption("--reset", action="store_true", default=False,
                     help="Force rebuild of all fixtures")


@pytest.fixture(scope="session")
def skip_llm(request):
    return request.config.getoption("--skip-llm")


@pytest.fixture(scope="session")
def full_perf(request):
    return request.config.getoption("--full-perf")


@pytest.fixture(scope="session")
def chatbot_url():
    return CHATBOT_URL


@pytest.fixture(scope="session")
def ws_url():
    return WS_URL


@pytest.fixture(scope="session")
def data_dir():
    return DATA_DIR


@pytest.fixture(scope="session")
def reports_dir():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    return REPORTS_DIR


def is_server_running(url: str) -> bool:
    """Check if the backend server is reachable."""
    try:
        resp = httpx.get(f"{url}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="session", autouse=True)
def ensure_server(chatbot_url):
    """Ensure the backend server is running before tests."""
    if not is_server_running(chatbot_url):
        pytest.skip(
            f"Backend server not running at {chatbot_url}. "
            "Start it with: uvicorn main:app --host 0.0.0.0 --port 8000"
        )


def load_test_data(filename: str) -> dict | list:
    """Load a JSON test data file from tests/data/."""
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def conversations_data():
    return load_test_data("conversations.json")


@pytest.fixture(scope="session")
def rag_queries_data():
    return load_test_data("rag_queries.json")


@pytest.fixture(scope="session")
def tool_invocations_data():
    return load_test_data("tool_invocations.json")
