"""
Qwen3 GGUF interface via LM Studio's OpenAI-compatible API.
Provides streaming, non-streaming, and tool-calling response generation.
Includes <think> block filtering and graceful fallback.
"""
import httpx
import json
import os
import logging
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, List, Optional, Any
from dotenv import load_dotenv

logger = logging.getLogger("medical_bot")

load_dotenv()

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "qwen/qwen3-4b"

FALLBACK_RESPONSES = {
    "timeout": "I'm processing your information. Please wait a moment or try again.",
    "connection_error": (
        "I'm temporarily unable to generate a response. "
        "Your information has been saved. Please try again."
    ),
    "generic": "I'm having trouble right now. Please try again in a moment.",
}


@dataclass
class ToolCallBuffer:
    """Accumulates streaming deltas for a single tool call."""
    id: str = ""
    name: str = ""
    arguments: str = ""


class ToolCallsDetected(Exception):
    """Raised mid-stream when the model signals it wants to call tools."""
    def __init__(self, tool_calls: List[ToolCallBuffer]):
        self.tool_calls = tool_calls


def _build_request_body(
    messages: list,
    model: str,
    temperature: float,
    top_p: float,
    stream: bool,
    tools: Optional[List[Dict]] = None,
    tool_choice: str = "auto",
    max_tokens: Optional[int] = None,
) -> Dict:
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": 15,
        "repeat_penalty": 1.1,
        "stream": stream,
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if tools:
        body["tools"] = tools
        body["tool_choice"] = tool_choice
    return body


class QwenInterface:
    """Interface to Qwen3 GGUF model via LM Studio (OpenAI-compatible API)."""

    def __init__(self, base_url: str = None, model: str = None, timeout: float = None):
        self.base_url = base_url or os.getenv("LM_STUDIO_BASE_URL", DEFAULT_BASE_URL)
        self.model = model or os.getenv("LM_STUDIO_MODEL", DEFAULT_MODEL)
        self.last_metrics: Dict = {}

    # ── Non-streaming (for tool detection / short outputs) ──────────────────

    async def generate_response(
        self,
        messages: list,
        max_tokens: Optional[int] = None,
        temperature: float = 0.5,
        top_p: float = 0.8,
        tools: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Non-streaming call.
        Returns a dict: {content: str | None, tool_calls: list | None}
        """
        body = _build_request_body(
            messages, self.model, temperature, top_p, False, tools, max_tokens=max_tokens
        )
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                resp = await client.post(f"{self.base_url}/chat/completions", json=body)
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                message = choice.get("message", {})
                return {
                    "content": message.get("content"),
                    "tool_calls": message.get("tool_calls"),
                    "finish_reason": choice.get("finish_reason"),
                }
            except httpx.TimeoutException:
                logger.error("LLM timeout (non-streaming)")
                return {"content": FALLBACK_RESPONSES["timeout"], "tool_calls": None}
            except httpx.ConnectError:
                logger.error("LLM server unreachable")
                return {"content": FALLBACK_RESPONSES["connection_error"], "tool_calls": None}
            except Exception as e:
                logger.error(f"LLM error: {e}")
                return {"content": FALLBACK_RESPONSES["generic"], "tool_calls": None}

    # ── Streaming (text only, no tools) ─────────────────────────────────────

    async def generate_response_stream(
        self,
        messages: list,
        max_tokens: Optional[int] = None,
        temperature: float = 0.5,
        top_p: float = 0.8,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming text generation.
        Filters <think>…</think> blocks (Qwen3 chain-of-thought).
        """
        body = _build_request_body(
            messages, self.model, temperature, top_p, True, max_tokens=max_tokens
        )
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream(
                    "POST", f"{self.base_url}/chat/completions", json=body
                ) as resp:
                    resp.raise_for_status()
                    buffer = ""
                    in_think = False

                    async for line in resp.aiter_lines():
                        if not line or "[DONE]" in line:
                            continue
                        if not line.startswith("data: "):
                            continue
                        try:
                            chunk = json.loads(line[6:])
                            delta = chunk["choices"][0]["delta"]
                            text = delta.get("content", "")
                            if not text:
                                continue
                            buffer += text
                            # Filter <think> blocks
                            while True:
                                if in_think:
                                    end = buffer.find("</think>")
                                    if end != -1:
                                        in_think = False
                                        buffer = buffer[end + 8:]
                                    else:
                                        buffer = buffer[-8:] if len(buffer) > 8 else buffer
                                        break
                                else:
                                    start = buffer.find("<think>")
                                    if start != -1:
                                        if start > 0:
                                            yield buffer[:start]
                                        in_think = True
                                        buffer = buffer[start + 7:]
                                    else:
                                        if len(buffer) > 7:
                                            yield buffer[:-7]
                                            buffer = buffer[-7:]
                                        break
                        except Exception:
                            continue

                    if buffer and not in_think and "<think" not in buffer:
                        yield buffer

            except httpx.TimeoutException:
                logger.error("LLM timeout (streaming)")
                yield FALLBACK_RESPONSES["timeout"]
            except httpx.ConnectError:
                logger.error("LLM unreachable (streaming)")
                yield FALLBACK_RESPONSES["connection_error"]
            except Exception as e:
                logger.error(f"LLM streaming error: {e}")
                yield FALLBACK_RESPONSES["generic"]

    # ── Streaming + tool detection ───────────────────────────────────────────

    async def generate_stream_with_tools(
        self,
        messages: list,
        tools: List[Dict],
        max_tokens: Optional[int] = None,
        temperature: float = 0.5,
        top_p: float = 0.8,
    ) -> AsyncGenerator[Any, None]:
        """
        Streaming call that detects tool calls.

        Yields:
          - str chunks for regular text responses
          - {"type": "tool_calls", "calls": [...]} dict when tool calls are detected

        The caller should:
          1. Collect text chunks and stream them to the user.
          2. On receiving a tool_calls dict: execute the tools, then make
             a follow-up streaming call with generate_response_stream().
        """
        body = _build_request_body(
            messages, self.model, temperature, top_p, True, tools, max_tokens=max_tokens
        )
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream(
                    "POST", f"{self.base_url}/chat/completions", json=body
                ) as resp:
                    resp.raise_for_status()

                    text_buffer = ""
                    in_think = False
                    tool_calls_map: Dict[int, ToolCallBuffer] = {}
                    has_tool_calls = False
                    has_content = False

                    async for line in resp.aiter_lines():
                        if not line or "[DONE]" in line:
                            continue
                        if not line.startswith("data: "):
                            continue
                        try:
                            chunk = json.loads(line[6:])
                            choice = chunk["choices"][0]
                            delta = choice.get("delta", {})
                            finish_reason = choice.get("finish_reason")

                            # ── Handle tool_call deltas ────────────────────
                            tc_deltas = delta.get("tool_calls")
                            if tc_deltas:
                                has_tool_calls = True
                                for tc in tc_deltas:
                                    idx = tc.get("index", 0)
                                    if idx not in tool_calls_map:
                                        tool_calls_map[idx] = ToolCallBuffer()
                                    buf = tool_calls_map[idx]
                                    if tc.get("id"):
                                        buf.id += tc["id"]
                                    fn = tc.get("function", {})
                                    if fn.get("name"):
                                        buf.name += fn["name"]
                                    if fn.get("arguments"):
                                        buf.arguments += fn["arguments"]

                            # ── Handle content deltas ──────────────────────
                            text = delta.get("content", "")
                            if text:
                                has_content = True
                                text_buffer += text
                                # Filter <think> blocks and stream
                                while True:
                                    if in_think:
                                        end = text_buffer.find("</think>")
                                        if end != -1:
                                            in_think = False
                                            text_buffer = text_buffer[end + 8:]
                                        else:
                                            text_buffer = text_buffer[-8:] if len(text_buffer) > 8 else text_buffer
                                            break
                                    else:
                                        start = text_buffer.find("<think>")
                                        if start != -1:
                                            if start > 0:
                                                yield text_buffer[:start]
                                            in_think = True
                                            text_buffer = text_buffer[start + 7:]
                                        else:
                                            if len(text_buffer) > 7:
                                                yield text_buffer[:-7]
                                                text_buffer = text_buffer[-7:]
                                            break

                        except Exception:
                            continue

                    # ── Stream complete ────────────────────────────────────
                    # Flush remaining text buffer
                    if text_buffer and not in_think and "<think" not in text_buffer:
                        yield text_buffer

                    # If tool calls were detected, signal the caller
                    if has_tool_calls and tool_calls_map:
                        calls = []
                        for idx in sorted(tool_calls_map.keys()):
                            buf = tool_calls_map[idx]
                            calls.append({
                                "id": buf.id or f"call_{idx}",
                                "type": "function",
                                "function": {
                                    "name": buf.name,
                                    "arguments": buf.arguments or "{}",
                                },
                            })
                        yield {"type": "tool_calls", "calls": calls}

            except httpx.TimeoutException:
                logger.error("LLM timeout (stream+tools)")
                yield FALLBACK_RESPONSES["timeout"]
            except httpx.ConnectError:
                logger.error("LLM unreachable (stream+tools)")
                yield FALLBACK_RESPONSES["connection_error"]
            except Exception as e:
                logger.error(f"LLM stream+tools error: {e}")
                yield FALLBACK_RESPONSES["generic"]


# ── Module-level singleton ───────────────────────────────────────────────────

_default_interface: Optional[QwenInterface] = None


def get_llm() -> QwenInterface:
    """Return the default QwenInterface singleton."""
    global _default_interface
    if _default_interface is None:
        _default_interface = QwenInterface()
    return _default_interface
