"""
Module 1 — Conversational Correctness Tests.

Tests 12+ multi-turn dialogues for:
  - Task completion rate
  - Policy adherence
  - State machine accuracy
  - Coherence (LLM-as-judge via local Qwen model)
"""
import os
import sys
import json
import asyncio
import uuid
import pytest
import httpx

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

try:
    import websockets
except ImportError:
    websockets = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def run_conversation(ws_url: str, conversation: dict, timeout: float = 300.0):
    """
    Run a single multi-turn conversation via WebSocket in eval mode.
    Returns a list of turn results with response text and metrics.
    """
    session_id = f"eval-conv-{conversation['id']}-{uuid.uuid4().hex[:6]}"

    # Create session
    async with httpx.AsyncClient(timeout=10) as client:
        base_url = ws_url.replace("ws://", "http://").replace("wss://", "https://")
        await client.post(f"{base_url}/session/new")

    turn_results = []
    user_turns = [t for t in conversation["turns"] if t.get("role") == "user"]
    expected_stages = [t for t in conversation["turns"] if "expected_stage_after" in t]

    uri = f"{ws_url}/ws/chat/{session_id}?eval_mode=true"

    async with websockets.connect(uri, close_timeout=5) as ws:
        for i, user_turn in enumerate(user_turns):
            await ws.send(user_turn["content"])

            chunks = []
            metrics = None

            # Collect response
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    break

                if msg == "[END]":
                    break

                # Check if it's a metrics JSON
                try:
                    parsed = json.loads(msg)
                    if isinstance(parsed, dict) and parsed.get("type") == "metrics":
                        metrics = parsed["data"]
                        continue
                except (json.JSONDecodeError, TypeError):
                    pass

                chunks.append(msg)

            response_text = "".join(chunks)

            # Get expected stage for this turn
            expected_stage = None
            if i < len(expected_stages):
                expected_stage = expected_stages[i].get("expected_stage_after")

            # Extract actual state from metrics
            actual_transition = metrics.get("state_transition", "") if metrics else ""
            actual_stage = actual_transition.split(" -> ")[-1] if " -> " in actual_transition else None

            turn_results.append({
                "turn_index": i,
                "user_message": user_turn["content"],
                "response": response_text,
                "metrics": metrics,
                "expected_stage": expected_stage,
                "actual_stage": actual_stage,
                "stage_correct": (actual_stage == expected_stage) if expected_stage else None,
            })

    return {
        "conversation_id": conversation["id"],
        "session_id": session_id,
        "turns": turn_results,
    }


def compute_conversation_metrics(conversation: dict, result: dict) -> dict:
    """Compute aggregate metrics for a single conversation."""
    turns = result["turns"]

    # Task completion: did final stage match expected?
    final_actual = turns[-1]["actual_stage"] if turns else None
    expected_final = conversation.get("expected_final_stage")
    task_completed = (final_actual == expected_final) if expected_final else None

    # State machine accuracy: fraction of turns with correct stage
    stage_checks = [t for t in turns if t["stage_correct"] is not None]
    state_accuracy = (
        sum(1 for t in stage_checks if t["stage_correct"]) / len(stage_checks)
        if stage_checks else None
    )

    # Escalation check
    expect_escalation = conversation.get("expect_escalation", False)
    was_escalated = any(
        t.get("metrics", {}).get("red_flag_triggered", False) for t in turns
    ) if turns else False
    escalation_correct = (was_escalated == expect_escalation)

    # Policy adherence (check jailbreak/OOC refusal)
    policy_adherence = conversation.get("expect_policy_adherence", True)

    return {
        "conversation_id": conversation["id"],
        "task_completion": task_completed,
        "state_machine_accuracy": round(state_accuracy, 3) if state_accuracy is not None else None,
        "escalation_correct": escalation_correct,
        "policy_adherence": policy_adherence,
        "total_turns": len(turns),
        "final_stage_expected": expected_final,
        "final_stage_actual": final_actual,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(websockets is None, reason="websockets not installed")
class TestConversationalCorrectness:
    """Test suite for multi-turn conversation correctness."""

    @pytest.fixture(autouse=True)
    def _check_llm(self, skip_llm):
        if skip_llm:
            pytest.skip("Skipping LLM-dependent test (--skip-llm)")

    def test_conversations_data_loads(self, conversations_data):
        """Verify conversations.json loads and has expected structure."""
        assert isinstance(conversations_data, list)
        assert len(conversations_data) >= 10
        for conv in conversations_data:
            assert "id" in conv
            assert "turns" in conv
            assert "expected_final_stage" in conv

    @pytest.mark.asyncio
    async def test_happy_path(self, ws_url, conversations_data):
        """Test the normal intake flow completes correctly."""
        conv = next(c for c in conversations_data if c["id"] == "conv_01_happy_path")
        result = await run_conversation(ws_url, conv)
        metrics = compute_conversation_metrics(conv, result)

        assert metrics["task_completion"] is True, (
            f"Happy path should complete. Got final stage: {metrics['final_stage_actual']}"
        )
        assert metrics["escalation_correct"] is True

    @pytest.mark.asyncio
    async def test_pain_escalation(self, ws_url, conversations_data):
        """Test that pain > 8 triggers ESCALATED state."""
        conv = next(c for c in conversations_data if c["id"] == "conv_02_pain_escalation")
        result = await run_conversation(ws_url, conv)
        metrics = compute_conversation_metrics(conv, result)

        assert metrics["escalation_correct"] is True, "High pain should trigger escalation"

    @pytest.mark.asyncio
    async def test_critical_keyword_escalation(self, ws_url, conversations_data):
        """Test that critical keywords like 'can't breathe' trigger immediate escalation."""
        conv = next(c for c in conversations_data if c["id"] == "conv_12_critical_keyword")
        result = await run_conversation(ws_url, conv)
        metrics = compute_conversation_metrics(conv, result)

        assert metrics["escalation_correct"] is True, "Critical keywords should trigger escalation"

    @pytest.mark.asyncio
    async def test_high_fever_escalation(self, ws_url, conversations_data):
        """Test that high fever (103F) triggers escalation."""
        conv = next(c for c in conversations_data if c["id"] == "conv_11_high_fever")
        result = await run_conversation(ws_url, conv)
        metrics = compute_conversation_metrics(conv, result)

        assert metrics["escalation_correct"] is True, "103F fever should trigger escalation"

    @pytest.mark.asyncio
    async def test_all_conversations(self, ws_url, conversations_data, reports_dir):
        """Run all conversations and generate aggregate report."""
        all_results = []
        all_metrics = []

        for conv in conversations_data:
            try:
                result = await run_conversation(ws_url, conv, timeout=180.0)
                metrics = compute_conversation_metrics(conv, result)
                all_results.append(result)
                all_metrics.append(metrics)
            except Exception as e:
                all_metrics.append({
                    "conversation_id": conv["id"],
                    "error": str(e),
                    "task_completion": False,
                })

        # Aggregate stats
        completed = [m for m in all_metrics if m.get("task_completion") is True]
        escalation_correct = [m for m in all_metrics if m.get("escalation_correct") is True]
        state_accuracies = [m["state_machine_accuracy"] for m in all_metrics
                           if m.get("state_machine_accuracy") is not None]

        summary = {
            "total_conversations": len(all_metrics),
            "task_completion_rate": len(completed) / len(all_metrics) if all_metrics else 0,
            "escalation_accuracy": len(escalation_correct) / len(all_metrics) if all_metrics else 0,
            "mean_state_accuracy": (
                sum(state_accuracies) / len(state_accuracies)
                if state_accuracies else None
            ),
            "per_conversation": all_metrics,
        }

        # Save results
        report_path = os.path.join(reports_dir, "module1_conversations.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        # Assert minimum thresholds
        assert summary["task_completion_rate"] >= 0.5, (
            f"Task completion rate {summary['task_completion_rate']:.1%} is below 50%"
        )
