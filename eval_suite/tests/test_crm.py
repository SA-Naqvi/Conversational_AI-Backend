"""
Module 3 — CRM Tool Evaluation.

Tests:
  3.1 — Direct CRUD unit tests (no LLM)
  3.2 — LLM tool-call accuracy (requires LLM server)
"""
import os
import sys
import json
import asyncio
import uuid
import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

try:
    import websockets
except ImportError:
    websockets = None


# ---------------------------------------------------------------------------
# 3.1 — Direct CRUD Unit Tests (No LLM)
# ---------------------------------------------------------------------------

class TestCRMDirect:
    """Unit tests for CRM tool functions — no LLM needed."""

    def _get_crm_functions(self):
        """Import CRM functions."""
        from tools.crm_tool import _load_crm, _save_crm, CRM_FILE
        return _load_crm, _save_crm, CRM_FILE

    def test_create_and_read_profile(self):
        """Create a CRM profile and read it back."""
        load, save, crm_file = self._get_crm_functions()
        test_sid = f"test-crm-{uuid.uuid4().hex[:8]}"

        try:
            data = load()
            data[test_sid] = {
                "name": "Test Patient",
                "phone": "555-TEST",
                "created_at": "2025-01-01T00:00:00",
                "interaction_count": 0,
            }
            save(data)

            # Read back
            data2 = load()
            assert test_sid in data2
            assert data2[test_sid]["name"] == "Test Patient"
            assert data2[test_sid]["phone"] == "555-TEST"
        finally:
            # Cleanup
            data = load()
            data.pop(test_sid, None)
            save(data)

    def test_update_profile_field(self):
        """Update a specific field in a CRM profile."""
        load, save, _ = self._get_crm_functions()
        test_sid = f"test-crm-{uuid.uuid4().hex[:8]}"

        try:
            data = load()
            data[test_sid] = {"name": "Original Name", "interaction_count": 0}
            save(data)

            # Update
            data = load()
            data[test_sid]["name"] = "Updated Name"
            data[test_sid]["email"] = "updated@test.com"
            save(data)

            # Verify
            data = load()
            assert data[test_sid]["name"] == "Updated Name"
            assert data[test_sid]["email"] == "updated@test.com"
        finally:
            data = load()
            data.pop(test_sid, None)
            save(data)

    def test_delete_profile(self):
        """Delete a CRM profile."""
        load, save, _ = self._get_crm_functions()
        test_sid = f"test-crm-{uuid.uuid4().hex[:8]}"

        data = load()
        data[test_sid] = {"name": "To Delete"}
        save(data)

        # Delete
        data = load()
        del data[test_sid]
        save(data)

        # Verify gone
        data = load()
        assert test_sid not in data

    def test_interaction_recording(self):
        """Test recording interaction history."""
        load, save, _ = self._get_crm_functions()
        test_sid = f"test-crm-{uuid.uuid4().hex[:8]}"

        try:
            data = load()
            data[test_sid] = {
                "name": "History Test",
                "interaction_count": 0,
                "interaction_history": [],
            }
            save(data)

            # Record interaction
            data = load()
            data[test_sid]["interaction_history"].append({
                "timestamp": "2025-01-01T10:00:00",
                "note": "Test interaction note",
            })
            data[test_sid]["interaction_count"] += 1
            save(data)

            # Verify
            data = load()
            assert len(data[test_sid]["interaction_history"]) == 1
            assert data[test_sid]["interaction_count"] == 1
            assert data[test_sid]["interaction_history"][0]["note"] == "Test interaction note"
        finally:
            data = load()
            data.pop(test_sid, None)
            save(data)

    @pytest.mark.asyncio
    async def test_tool_classes_execute(self):
        """Test CRM tool classes can execute directly."""
        from tools.crm_tool import GetUserInfoTool, UpdateUserInfoTool, RecordInteractionTool

        test_sid = f"test-crm-{uuid.uuid4().hex[:8]}"

        # Get — should not find
        get_tool = GetUserInfoTool()
        result = await get_tool.execute(session_id=test_sid)
        assert result["found"] is False

        # Update — should create and update
        update_tool = UpdateUserInfoTool()
        result = await update_tool.execute(session_id=test_sid, field="name", value="Async Test")
        assert result["success"] is True

        # Get — should now find
        result = await get_tool.execute(session_id=test_sid)
        assert result["found"] is True
        assert result["user"]["name"] == "Async Test"

        # Record interaction
        record_tool = RecordInteractionTool()
        result = await record_tool.execute(session_id=test_sid, note="Test note via tool")
        assert result["success"] is True

        # Cleanup
        from tools.crm_tool import _load_crm, _save_crm
        data = _load_crm()
        data.pop(test_sid, None)
        _save_crm(data)

    def test_auto_increment_session(self):
        """Test auto_increment_session helper."""
        from tools.crm_tool import auto_increment_session, _load_crm, _save_crm

        test_sid = f"test-crm-{uuid.uuid4().hex[:8]}"

        try:
            data = _load_crm()
            data[test_sid] = {"interaction_count": 0}
            _save_crm(data)

            auto_increment_session(test_sid)

            data = _load_crm()
            assert data[test_sid]["interaction_count"] == 1
            assert "last_seen" in data[test_sid]
        finally:
            data = _load_crm()
            data.pop(test_sid, None)
            _save_crm(data)

    def test_invalid_session_get(self):
        """Getting a nonexistent session should return None."""
        from tools.crm_tool import get_user_profile
        result = get_user_profile(f"nonexistent-{uuid.uuid4().hex}")
        assert result is None


# ---------------------------------------------------------------------------
# 3.2 — LLM Tool-Call Accuracy (Requires LLM Server)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(websockets is None, reason="websockets not installed")
class TestCRMToolCallAccuracy:
    """Test LLM's ability to correctly invoke CRM tools."""

    @pytest.fixture(autouse=True)
    def _check_llm(self, skip_llm):
        if skip_llm:
            pytest.skip("Skipping LLM-dependent test (--skip-llm)")

    @pytest.mark.asyncio
    async def test_crm_tool_invocations(self, ws_url, tool_invocations_data, reports_dir):
        """Test that the LLM correctly calls CRM tools for relevant utterances."""
        crm_tests = tool_invocations_data.get("crm", [])
        if not crm_tests:
            pytest.skip("No CRM test data")

        results = []
        tp, fp, fn = 0, 0, 0

        for test_case in crm_tests:
            session_id = f"eval-crm-{test_case['id']}-{uuid.uuid4().hex[:6]}"
            uri = f"{ws_url}/ws/chat/{session_id}?eval_mode=true"

            try:
                async with websockets.connect(uri, close_timeout=5) as ws:
                    # Send the test utterance
                    await ws.send(test_case["utterance"])

                    chunks = []
                    metrics = None
                    while True:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=300)
                        except asyncio.TimeoutError:
                            break
                        if msg == "[END]":
                            break
                        try:
                            parsed = json.loads(msg)
                            if isinstance(parsed, dict) and parsed.get("type") == "metrics":
                                metrics = parsed["data"]
                                continue
                        except (json.JSONDecodeError, TypeError):
                            pass
                        chunks.append(msg)

                response = "".join(chunks)
                tool_ms = metrics.get("tool_ms", {}) if metrics else {}
                tool_was_called = len(tool_ms) > 0
                called_tools = list(tool_ms.keys())

                expected_tool = test_case.get("expected_tool")
                should_fire = test_case.get("should_fire", True)

                # Check if expected tool was called
                tool_correct = False
                if should_fire and expected_tool:
                    if expected_tool in called_tools:
                        tp += 1
                        tool_correct = True
                    else:
                        fn += 1
                elif not should_fire:
                    if tool_was_called:
                        fp += 1
                    else:
                        tp += 1
                        tool_correct = True

                results.append({
                    "id": test_case["id"],
                    "utterance": test_case["utterance"],
                    "expected_tool": expected_tool,
                    "called_tools": called_tools,
                    "tool_correct": tool_correct,
                })

            except Exception as e:
                results.append({
                    "id": test_case["id"],
                    "error": str(e),
                    "tool_correct": False,
                })

        total = tp + fp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        summary = {
            "total_tests": len(crm_tests),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "per_test": results,
        }

        report_path = os.path.join(reports_dir, "module3_crm.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
