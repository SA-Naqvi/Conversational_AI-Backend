"""
Module 5 — Latency and Throughput Benchmarks.

Tests:
  5.1 — Single-user latency (4 scenarios × N trials)
  5.2 — Concurrency ramp (1–10 users)
  5.3 — Hardware context capture
"""
import os
import sys
import json
import time
import asyncio
import platform
import uuid
import statistics
import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

try:
    import websockets
except ImportError:
    websockets = None

try:
    import psutil
except ImportError:
    psutil = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_hardware_context() -> dict:
    """Capture hardware and runtime information."""
    ctx = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    if psutil:
        ctx["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
        ctx["cpu_percent"] = psutil.cpu_percent(interval=1)
    return ctx


async def run_single_trial(ws_url: str, messages: list, session_id: str = None,
                           timeout: float = 180.0) -> dict:
    """
    Run a single trial: send messages via eval-mode WebSocket.
    Returns timing metrics from the last message.
    """
    sid = session_id or f"eval-perf-{uuid.uuid4().hex[:8]}"
    uri = f"{ws_url}/ws/chat/{sid}?eval_mode=true"
    last_metrics = None

    async with websockets.connect(uri, close_timeout=5) as ws:
        for msg in messages:
            await ws.send(msg)
            chunks = []

            while True:
                try:
                    resp = await asyncio.wait_for(ws.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    break
                if resp == "[END]":
                    break
                try:
                    parsed = json.loads(resp)
                    if isinstance(parsed, dict) and parsed.get("type") == "metrics":
                        last_metrics = parsed["data"]
                        continue
                except (json.JSONDecodeError, TypeError):
                    pass
                chunks.append(resp)

    return last_metrics or {}


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

# Do intake first, then the actual test message
INTAKE_MESSAGES = [
    "I had knee surgery",
    "January 15, 2025",
    "My pain is 4/10",
]

SCENARIOS = {
    "simple_dialogue": {
        "description": "Simple question — no RAG, no tool",
        "messages": INTAKE_MESSAGES + ["Thank you, I'm feeling okay today"],
    },
    "rag_only": {
        "description": "Medical question that triggers RAG retrieval",
        "messages": INTAKE_MESSAGES + ["How should I care for my surgical wound?"],
    },
    "tool_only": {
        "description": "Drug info question that triggers OpenFDA tool",
        "messages": INTAKE_MESSAGES + ["What are the side effects of ibuprofen?"],
    },
    "mixed": {
        "description": "Complex question that may trigger both RAG and tool",
        "messages": INTAKE_MESSAGES + ["Tell me about FDA warnings for oxycodone and recovery tips"],
    },
}


# ---------------------------------------------------------------------------
# 5.1 — Single-User Latency Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(websockets is None, reason="websockets not installed")
class TestSingleUserLatency:
    """Measure TTFT, total latency, and inter-token latency."""

    @pytest.fixture(autouse=True)
    def _check_llm(self, skip_llm):
        if skip_llm:
            pytest.skip("Skipping LLM-dependent test (--skip-llm)")

    @pytest.mark.asyncio
    async def test_latency_all_scenarios(self, ws_url, full_perf, reports_dir):
        """Run latency benchmarks for all 4 scenarios."""
        n_trials = 30 if full_perf else 3
        all_results = {}
        hw_context = get_hardware_context()

        for scenario_name, scenario in SCENARIOS.items():
            trials = []
            for trial in range(n_trials):
                try:
                    sid = f"eval-perf-{scenario_name}-{trial}-{uuid.uuid4().hex[:6]}"
                    metrics = await run_single_trial(
                        ws_url, scenario["messages"], session_id=sid
                    )
                    if metrics:
                        trials.append({
                            "trial": trial,
                            "ttft_ms": metrics.get("ttft_ms"),
                            "total_ms": metrics.get("total_ms"),
                            "tokens_generated": metrics.get("tokens_generated", 0),
                            "retrieval_ms": metrics.get("retrieval_ms"),
                            "tool_ms": metrics.get("tool_ms", {}),
                        })
                except Exception as e:
                    trials.append({"trial": trial, "error": str(e)})

            # Compute stats
            ttfts = [t["ttft_ms"] for t in trials if t.get("ttft_ms") is not None]
            totals = [t["total_ms"] for t in trials if t.get("total_ms") is not None]
            tokens = [t["tokens_generated"] for t in trials if t.get("tokens_generated")]

            inter_token = []
            for t in trials:
                if t.get("total_ms") and t.get("ttft_ms") and t.get("tokens_generated", 0) > 1:
                    gen_time = t["total_ms"] - t["ttft_ms"]
                    if gen_time > 0:
                        inter_token.append(gen_time / (t["tokens_generated"] - 1))

            def calc_stats(values):
                if not values:
                    return {"mean": None, "median": None, "p90": None, "p99": None, "min": None, "max": None}
                s = sorted(values)
                n = len(s)
                return {
                    "mean": round(statistics.mean(s), 2),
                    "median": round(statistics.median(s), 2),
                    "p90": round(s[int(n * 0.9)] if n > 1 else s[0], 2),
                    "p99": round(s[int(n * 0.99)] if n > 1 else s[0], 2),
                    "min": round(min(s), 2),
                    "max": round(max(s), 2),
                }

            all_results[scenario_name] = {
                "description": scenario["description"],
                "n_trials": n_trials,
                "successful_trials": len(ttfts),
                "ttft_ms": calc_stats(ttfts),
                "total_ms": calc_stats(totals),
                "inter_token_ms": calc_stats(inter_token),
                "tokens_generated": calc_stats(tokens),
                "raw_trials": trials,
            }

        summary = {
            "hardware": hw_context,
            "scenarios": all_results,
        }

        report_path = os.path.join(reports_dir, "module5_performance.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# 5.2 — Concurrency Ramp
# ---------------------------------------------------------------------------

@pytest.mark.skipif(websockets is None, reason="websockets not installed")
class TestConcurrencyRamp:
    """Test system behavior under concurrent load."""

    @pytest.fixture(autouse=True)
    def _check_llm(self, skip_llm):
        if skip_llm:
            pytest.skip("Skipping LLM-dependent test (--skip-llm)")

    @pytest.mark.asyncio
    async def test_concurrency_ramp(self, ws_url, full_perf, reports_dir):
        """Ramp concurrent users from 1 to N and measure latency degradation."""
        max_users = 10 if full_perf else 5
        ramp_results = []

        script = INTAKE_MESSAGES + ["How should I manage my recovery?"]

        for n_users in range(1, max_users + 1):
            tasks = []
            for u in range(n_users):
                sid = f"eval-concurrency-{n_users}-{u}-{uuid.uuid4().hex[:6]}"
                tasks.append(run_single_trial(ws_url, script, session_id=sid, timeout=300))

            start = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            wall_time = (time.time() - start) * 1000

            ttfts = []
            totals = []
            errors = 0
            for r in results:
                if isinstance(r, Exception):
                    errors += 1
                elif isinstance(r, dict):
                    if r.get("ttft_ms") is not None:
                        ttfts.append(r["ttft_ms"])
                    if r.get("total_ms") is not None:
                        totals.append(r["total_ms"])
                else:
                    errors += 1

            ramp_results.append({
                "concurrent_users": n_users,
                "wall_time_ms": round(wall_time, 2),
                "median_ttft_ms": round(statistics.median(ttfts), 2) if ttfts else None,
                "median_total_ms": round(statistics.median(totals), 2) if totals else None,
                "errors": errors,
                "successful": len(ttfts),
            })

        summary = {
            "max_concurrent_users": max_users,
            "ramp_results": ramp_results,
        }

        report_path = os.path.join(reports_dir, "module5_concurrency.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# 5.3 — Hardware Context
# ---------------------------------------------------------------------------

class TestHardwareContext:
    """Capture and report hardware context."""

    def test_hardware_info(self, reports_dir):
        """Record hardware information for the test environment."""
        ctx = get_hardware_context()

        report_path = os.path.join(reports_dir, "hardware_context.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(ctx, f, indent=2)

        assert ctx["cpu_count"] is not None
        assert ctx["python_version"] is not None
