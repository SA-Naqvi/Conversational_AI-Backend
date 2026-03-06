"""
Qwen3 GGUF interface via LM Studio's OpenAI-compatible API.
Provides both non-streaming and streaming response generation.
Includes fallback responses when LLM is unavailable.
"""
import httpx
import json
import os
import logging
from typing import AsyncGenerator

logger = logging.getLogger("medical_bot")

BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
MODEL = os.getenv("LM_STUDIO_MODEL", "qwen/qwen3-4b")

# Fallback responses when LLM is unavailable
FALLBACK_RESPONSES = {
    "timeout": "I'm processing your information. Please wait a moment or try again.",
    "connection_error": (
        "I'm temporarily unable to generate a response. "
        "Your information has been saved. Please try again."
    ),
    "malformed_input": "I didn't understand that. Could you rephrase?",
    "rate_limited": "You're sending messages too quickly. Please wait a moment.",
    "generic": "I'm having trouble right now. Please try again in a moment.",
}


class QwenInterface:
    """Interface to Qwen3 GGUF model via LM Studio server."""

    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        timeout: float = 60.0,
    ):
        self.base_url = base_url or BASE_URL
        self.model = model or MODEL
        self.timeout = timeout

    async def generate_response(
        self,
        messages: list,
        max_tokens: int = 512,
        temperature: float = 0.5,
        top_p: float = 0.8,
    ) -> str:
        """
        Non-streaming response generation.
        Used for short, structured outputs.
        """
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout)
        ) as client:
            try:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": 15,
                        "repeat_penalty": 1.1,
                        "max_tokens": max_tokens,
                        "stream": False,
                    },
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except httpx.TimeoutException:
                logger.error("LLM timeout during non-streaming generation")
                return FALLBACK_RESPONSES["timeout"]
            except httpx.ConnectError:
                logger.error("LLM server unreachable")
                return FALLBACK_RESPONSES["connection_error"]
            except Exception as e:
                logger.error(f"LLM error: {e}")
                return FALLBACK_RESPONSES["generic"]

    async def generate_response_stream(
        self,
        messages: list,
        max_tokens: int = 512,
        temperature: float = 0.5,
        top_p: float = 0.8,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming response generation.
        Yields text chunks as they arrive.
        Filters out <think>...</think> blocks from the stream.
        """
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout)
        ) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": 15,
                        "repeat_penalty": 1.1,
                        "max_tokens": max_tokens,
                        "stream": True,
                    },
                ) as resp:
                    resp.raise_for_status()

                    buffer = ""
                    in_think_block = False

                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data: ") and "[DONE]" not in line:
                            try:
                                chunk = json.loads(line[6:])
                                delta = (
                                    chunk["choices"][0]["delta"]
                                    .get("content", "")
                                )
                                if not delta:
                                    continue

                                buffer += delta

                                # Filter <think>...</think> blocks
                                while True:
                                    if in_think_block:
                                        end_idx = buffer.find("</think>")
                                        if end_idx != -1:
                                            in_think_block = False
                                            buffer = buffer[end_idx + 8:]
                                            continue
                                        else:
                                            if len(buffer) > 8:
                                                buffer = buffer[-8:]
                                            break
                                    else:
                                        start_idx = buffer.find("<think>")
                                        if start_idx != -1:
                                            if start_idx > 0:
                                                yield buffer[:start_idx]
                                            in_think_block = True
                                            buffer = buffer[start_idx + 7:]
                                            continue
                                        else:
                                            if len(buffer) > 7:
                                                yield buffer[:-7]
                                                buffer = buffer[-7:]
                                            break
                            except Exception:
                                continue

                    # Yield remaining buffer
                    if buffer and not in_think_block and "<think" not in buffer:
                        yield buffer

            except httpx.TimeoutException:
                logger.error("LLM timeout during streaming")
                yield FALLBACK_RESPONSES["timeout"]
            except httpx.ConnectError:
                logger.error("LLM server unreachable during streaming")
                yield FALLBACK_RESPONSES["connection_error"]
            except Exception as e:
                logger.error(f"LLM streaming error: {e}")
                yield FALLBACK_RESPONSES["generic"]


# Module-level singleton for convenience
_default_interface = None


def get_llm() -> QwenInterface:
    """Get the default QwenInterface singleton."""
    global _default_interface
    if _default_interface is None:
        _default_interface = QwenInterface()
    return _default_interface
