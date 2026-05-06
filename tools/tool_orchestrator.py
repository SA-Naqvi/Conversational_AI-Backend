"""
Tool Orchestrator — central registry and executor for all callable tools.

Registered tools:
  CRM (mandatory):
    get_user_info, update_user_info, record_interaction

  OpenFDA (drug info from FDA):
    search_drug_label, search_drug_adverse_events, check_drug_recall

  PubMed (biomedical research):
    search_pubmed, get_pubmed_abstract

  Google Calendar (appointment scheduling):
    create_calendar_event, list_calendar_events, delete_calendar_event
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from tools.base import BaseTool
from tools.crm_tool import GetUserInfoTool, UpdateUserInfoTool, RecordInteractionTool
from tools.openfda_tool import DrugLabelTool, DrugAdverseEventsTool, DrugRecallTool
from tools.pubmed_tool import PubMedSearchTool, PubMedGetAbstractTool
from tools.google_calendar_tool import (
    CreateCalendarEventTool,
    ListCalendarEventsTool,
    DeleteCalendarEventTool,
)

logger = logging.getLogger("medical_bot")

TOOL_TIMEOUT_SECONDS = 20   # Google Calendar / PubMed can be slower


class ToolOrchestrator:
    """Registry and async executor for all available tools."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._register_all()

    def _register_all(self):
        tools = [
            # ── CRM (mandatory) ──────────────────────────────────────────
            GetUserInfoTool(),
            UpdateUserInfoTool(),
            RecordInteractionTool(),
            # ── OpenFDA ──────────────────────────────────────────────────
            DrugLabelTool(),
            DrugAdverseEventsTool(),
            DrugRecallTool(),
            # ── PubMed ───────────────────────────────────────────────────
            PubMedSearchTool(),
            PubMedGetAbstractTool(),
            # ── Google Calendar ──────────────────────────────────────────
            CreateCalendarEventTool(),
            ListCalendarEventsTool(),
            DeleteCalendarEventTool(),
        ]
        for tool in tools:
            self._tools[tool.name] = tool
        logger.info(f"Registered {len(self._tools)} tools: {list(self._tools.keys())}")

    def get_tools_schema(self) -> List[Dict]:
        """Return OpenAI-compatible function-calling schemas for all tools."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def get_tool_names(self) -> List[str]:
        return list(self._tools.keys())

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Execute a single tool by name.
        Returns a JSON string result (or error message).
        Enforces a timeout to prevent blocking.
        """
        tool = self._tools.get(tool_name)
        if not tool:
            error = {
                "error": f"Unknown tool: '{tool_name}'.",
                "available_tools": self.get_tool_names(),
            }
            logger.warning(f"Tool not found: {tool_name}")
            return json.dumps(error)

        try:
            result = await asyncio.wait_for(
                tool.execute(**arguments),
                timeout=TOOL_TIMEOUT_SECONDS,
            )
            logger.info(f"Tool '{tool_name}' executed successfully")
            return json.dumps(result, ensure_ascii=False, default=str)
        except asyncio.TimeoutError:
            error = {"error": f"Tool '{tool_name}' timed out after {TOOL_TIMEOUT_SECONDS}s"}
            logger.error(f"Tool timeout: {tool_name}")
            return json.dumps(error)
        except Exception as e:
            error = {"error": f"Tool '{tool_name}' failed: {str(e)}"}
            logger.error(f"Tool error ({tool_name}): {e}", exc_info=True)
            return json.dumps(error)

    async def execute_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        """
        Execute a list of LLM-requested tool calls concurrently.
        Returns OpenAI-format tool result messages.
        """
        coros = []
        for tc in tool_calls:
            name = tc.get("function", {}).get("name", "")
            args_str = tc.get("function", {}).get("arguments", "{}")
            tc_id = tc.get("id", "unknown")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                args = {}
            coros.append((tc_id, name, self.execute_tool(name, args)))

        results = []
        for tc_id, name, coro in coros:
            result_str = await coro
            results.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "name": name,
                "content": result_str,
            })
            logger.debug(f"Tool result '{name}': {result_str[:200]}")

        return results


# ── Module-level singleton ────────────────────────────────────────────────────

_orchestrator: Optional[ToolOrchestrator] = None


def get_orchestrator() -> ToolOrchestrator:
    """Return the singleton ToolOrchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ToolOrchestrator()
    return _orchestrator
