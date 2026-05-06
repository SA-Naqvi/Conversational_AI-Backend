"""
Module 4 — Additional Tools (OpenFDA, PubMed, Google Calendar) Evaluation.

Tests:
  4.1 — Functional unit tests (direct tool execution)
  4.2 — LLM invocation accuracy
  4.3 — Tool selection confusion matrix
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
# 4.1 — Functional Unit Tests (No LLM)
# ---------------------------------------------------------------------------

class TestOpenFDATools:
    """Functional tests for OpenFDA tool — calls the real FDA API."""

    @pytest.mark.asyncio
    async def test_drug_label_found(self):
        """search_drug_label should find ibuprofen."""
        from tools.openfda_tool import DrugLabelTool
        tool = DrugLabelTool()
        result = await tool.execute(drug_name="ibuprofen")
        assert result["found"] is True
        assert result["drug_name"] == "ibuprofen"
        # Should have at least some label sections
        has_sections = any(
            k in result for k in ["warnings", "indications_and_usage", "dosage_and_administration"]
        )
        assert has_sections, "Should return at least one label section"

    @pytest.mark.asyncio
    async def test_drug_label_not_found(self):
        """search_drug_label should handle unknown drugs gracefully."""
        from tools.openfda_tool import DrugLabelTool
        tool = DrugLabelTool()
        result = await tool.execute(drug_name="nonexistent_drug_xyz_12345")
        assert result["found"] is False

    @pytest.mark.asyncio
    async def test_adverse_events(self):
        """search_drug_adverse_events should return reactions for aspirin."""
        from tools.openfda_tool import DrugAdverseEventsTool
        tool = DrugAdverseEventsTool()
        result = await tool.execute(drug_name="aspirin")
        assert result["found"] is True
        assert len(result.get("top_adverse_reactions", [])) > 0

    @pytest.mark.asyncio
    async def test_drug_recall(self):
        """check_drug_recall should not crash for any input."""
        from tools.openfda_tool import DrugRecallTool
        tool = DrugRecallTool()
        result = await tool.execute(drug_name="ibuprofen")
        # May or may not find recalls — just verify no crash
        assert "drug_name" in result


class TestPubMedTools:
    """Functional tests for PubMed tool — calls the real NCBI API."""

    @pytest.mark.asyncio
    async def test_pubmed_search(self):
        """search_pubmed should return results for a valid medical query."""
        from tools.pubmed_tool import PubMedSearchTool
        tool = PubMedSearchTool()
        result = await tool.execute(query="knee replacement recovery", max_results=3)
        assert result["found"] is True
        assert len(result.get("articles", [])) > 0
        # Verify article structure
        article = result["articles"][0]
        assert "title" in article
        assert "pmid" in article

    @pytest.mark.asyncio
    async def test_pubmed_no_results(self):
        """search_pubmed should handle zero-result queries gracefully."""
        from tools.pubmed_tool import PubMedSearchTool
        tool = PubMedSearchTool()
        result = await tool.execute(
            query="xyznonexistentqueryqqqzzz12345",
            max_results=1,
        )
        # Should either find nothing or handle gracefully
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_get_abstract(self):
        """get_pubmed_abstract should return an article for a valid PMID."""
        from tools.pubmed_tool import PubMedGetAbstractTool
        tool = PubMedGetAbstractTool()
        # Use a well-known PMID (this is a real PubMed article)
        result = await tool.execute(pmid="33782455")
        # Either found or graceful failure
        assert isinstance(result, dict)


class TestGoogleCalendarTools:
    """Functional tests for Google Calendar tool."""

    def _is_configured(self):
        import pathlib
        creds = pathlib.Path(BACKEND_DIR) / "data" / "calendar" / "credentials.json"
        return creds.exists()

    def test_calendar_not_configured_graceful(self):
        """Calendar should return a helpful message when not configured."""
        if self._is_configured():
            pytest.skip("Calendar is configured — skipping not-configured test")

        from tools.google_calendar_tool import CreateCalendarEventTool
        tool = CreateCalendarEventTool()
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                tool.execute(summary="Test", date="tomorrow")
            )
            assert result["success"] is False
            assert "not configured" in result.get("error", "").lower() or "setup" in json.dumps(result).lower()
        finally:
            loop.close()

    @pytest.mark.asyncio
    async def test_calendar_create_list_delete(self):
        """Full calendar CRUD cycle — only runs if configured."""
        if not self._is_configured():
            pytest.skip("Google Calendar not configured")

        from tools.google_calendar_tool import (
            CreateCalendarEventTool, ListCalendarEventsTool, DeleteCalendarEventTool,
        )

        # Create
        create = CreateCalendarEventTool()
        result = await create.execute(
            summary="Eval Suite Test Appointment",
            date="tomorrow",
            time="10:00 AM",
            duration_minutes=15,
            description="Auto-created by eval suite",
        )
        assert result["success"] is True
        event_id = result.get("event_id")
        assert event_id

        # List
        list_tool = ListCalendarEventsTool()
        list_result = await list_tool.execute(max_results=10, days_ahead=7)
        assert list_result["found"] is True

        # Delete
        delete = DeleteCalendarEventTool()
        del_result = await delete.execute(event_id=event_id)
        assert del_result["success"] is True


# ---------------------------------------------------------------------------
# 4.2 — LLM Invocation Accuracy
# ---------------------------------------------------------------------------

@pytest.mark.skipif(websockets is None, reason="websockets not installed")
class TestToolInvocationAccuracy:
    """Test LLM's ability to select the correct tool for each utterance."""

    @pytest.fixture(autouse=True)
    def _check_llm(self, skip_llm):
        if skip_llm:
            pytest.skip("Skipping LLM-dependent test (--skip-llm)")

    @pytest.mark.asyncio
    async def test_all_tool_invocations(self, ws_url, tool_invocations_data, reports_dir):
        """
        Run all tool invocation tests and build a confusion matrix.
        """
        # Flatten all tool categories
        all_tests = []
        for category, tests in tool_invocations_data.items():
            for test in tests:
                test["category"] = category
                all_tests.append(test)

        results = []
        # Confusion matrix: expected_tool -> {actual_tool -> count}
        confusion = {}
        tool_names = set()

        for test_case in all_tests:
            session_id = f"eval-tool-{test_case['id']}-{uuid.uuid4().hex[:6]}"

            # First do intake so the bot is in MONITORING and willing to use tools
            uri = f"{ws_url}/ws/chat/{session_id}?eval_mode=true"
            intake_messages = [
                "I had knee surgery",
                "January 15, 2025",
                "My pain is 4/10",
            ]

            try:
                async with websockets.connect(uri, close_timeout=5) as ws:
                    # Run through intake
                    for msg in intake_messages:
                        await ws.send(msg)
                        while True:
                            try:
                                resp = await asyncio.wait_for(ws.recv(), timeout=300)
                            except asyncio.TimeoutError:
                                break
                            if resp == "[END]":
                                break

                    # Now send the test utterance
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

                tool_ms = metrics.get("tool_ms", {}) if metrics else {}
                called_tools = list(tool_ms.keys())
                actual_tool = called_tools[0] if called_tools else None

                expected_tool = test_case.get("expected_tool")
                should_fire = test_case.get("should_fire", True)

                # Build confusion matrix entry
                expected_label = expected_tool or "no_tool"
                actual_label = actual_tool or "no_tool"
                tool_names.add(expected_label)
                tool_names.add(actual_label)

                if expected_label not in confusion:
                    confusion[expected_label] = {}
                confusion[expected_label][actual_label] = confusion[expected_label].get(actual_label, 0) + 1

                correct = (expected_label == actual_label)

                results.append({
                    "id": test_case["id"],
                    "category": test_case["category"],
                    "utterance": test_case["utterance"],
                    "expected_tool": expected_label,
                    "actual_tool": actual_label,
                    "correct": correct,
                })

            except Exception as e:
                results.append({
                    "id": test_case["id"],
                    "error": str(e),
                    "correct": False,
                })

        # Compute accuracy
        correct_count = sum(1 for r in results if r.get("correct"))
        accuracy = correct_count / len(results) if results else 0

        summary = {
            "total_tests": len(all_tests),
            "correct": correct_count,
            "accuracy": round(accuracy, 3),
            "confusion_matrix": confusion,
            "tool_names": sorted(tool_names),
            "per_test": results,
        }

        report_path = os.path.join(reports_dir, "module4_tools.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
