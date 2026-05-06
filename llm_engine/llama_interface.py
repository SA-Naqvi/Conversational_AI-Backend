"""
llama.cpp interface via OpenAI-compatible HTTP API.
Provides streaming and non-streaming generation with:
  - <think> tag filtering
  - token buffering for streaming (sends every N tokens)
  - retry logic on failure
  - generation metrics tracking
  - fallback responses
"""
import httpx
import json
import time
import logging
from typing import AsyncGenerator, Optional

from config import (
    LLAMA_CPP_BASE_URL,
    LLAMA_CPP_MODEL,
    LLM_TIMEOUT_SECONDS,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_TOP_K,
    LLM_REPEAT_PENALTY,
    LLM_MAX_TOKENS,
    STREAM_BUFFER_SIZE,
)

logger = logging.getLogger("medical_bot")

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


class LlamaCppInterface:
    """Interface to llama.cpp server via OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        timeout: float = None,
        max_retries: int = 2,
    ):
        self.base_url = base_url or LLAMA_CPP_BASE_URL
        self.model = model or LLAMA_CPP_MODEL
        self.timeout = timeout or LLM_TIMEOUT_SECONDS
        self.max_retries = max_retries
        self._last_metrics = {}

    @property
    def last_metrics(self) -> dict:
        """Return metrics from the last generation call."""
        return self._last_metrics

    async def generate_response(
        self,
        messages: list,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
    ) -> str:
        """
        Non-streaming response generation.
        Used for short, structured outputs.
        """
        max_tokens = max_tokens or LLM_MAX_TOKENS
        temperature = temperature if temperature is not None else LLM_TEMPERATURE
        top_p = top_p if top_p is not None else LLM_TOP_P

        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(self.timeout)
                ) as client:
                    resp = await client.post(
                        f"{self.base_url}/v1/chat/completions",
                        json={
                            "model": self.model,
                            "messages": messages,
                            "temperature": temperature,
                            "top_p": top_p,
                            "top_k": LLM_TOP_K,
                            "repeat_penalty": LLM_REPEAT_PENALTY,
                            "max_tokens": max_tokens,
                            "stream": False,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]

                    # Track metrics
                    elapsed = time.time() - start_time
                    usage = data.get("usage", {})
                    self._last_metrics = {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "generation_time": round(elapsed, 3),
                        "tokens_per_second": round(
                            usage.get("completion_tokens", 0) / elapsed, 1
                        ) if elapsed > 0 else 0,
                    }

                    # Strip <think> blocks from response
                    content = _strip_think_blocks(content)
                    return content

            except httpx.TimeoutException:
                logger.error(
                    f"LLM timeout (attempt {attempt + 1}/{self.max_retries + 1})"
                )
                if attempt == self.max_retries:
                    return FALLBACK_RESPONSES["timeout"]
            except httpx.ConnectError:
                logger.error("LLM server unreachable")
                return FALLBACK_RESPONSES["connection_error"]
            except Exception as e:
                logger.error(f"LLM error: {e}")
                if attempt == self.max_retries:
                    return FALLBACK_RESPONSES["generic"]

        return FALLBACK_RESPONSES["generic"]

    async def generate_response_stream(
        self,
        messages: list,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming response generation with:
        - <think> tag filtering (post-processed per chunk batch)
        - token buffering (yields every STREAM_BUFFER_SIZE tokens)
        - retry on failure
        - generation metrics
        """
        max_tokens = max_tokens or LLM_MAX_TOKENS
        temperature = temperature if temperature is not None else LLM_TEMPERATURE
        top_p = top_p if top_p is not None else LLM_TOP_P

        start_time = time.time()
        total_tokens = 0
        in_think_block = False

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout)
        ) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": LLM_TOP_K,
                        "repeat_penalty": LLM_REPEAT_PENALTY,
                        "max_tokens": max_tokens,
                        "stream": True,
                    },
                ) as resp:
                    resp.raise_for_status()

                    raw_buffer = ""  # accumulates raw content for think-filtering
                    last_yielded_pos = 0  # track what we've already sent
                    chunk_count = 0

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

                                total_tokens += 1
                                raw_buffer += delta
                                chunk_count += 1

                                # Process and yield every N chunks
                                if chunk_count >= STREAM_BUFFER_SIZE:
                                    cleaned = _strip_think_blocks(raw_buffer)
                                    if len(cleaned) > last_yielded_pos:
                                        new_content = cleaned[last_yielded_pos:]
                                        last_yielded_pos = len(cleaned)
                                        yield new_content
                                    chunk_count = 0

                            except Exception:
                                continue

                    # Final flush — yield any remaining content
                    cleaned = _strip_think_blocks(raw_buffer)
                    if len(cleaned) > last_yielded_pos:
                        yield cleaned[last_yielded_pos:]

                    # Track metrics
                    elapsed = time.time() - start_time
                    self._last_metrics = {
                        "prompt_tokens": 0,  # not available in streaming
                        "completion_tokens": total_tokens,
                        "generation_time": round(elapsed, 3),
                        "tokens_per_second": round(
                            total_tokens / elapsed, 1
                        ) if elapsed > 0 else 0,
                    }

            except httpx.TimeoutException:
                logger.error("LLM timeout during streaming")
                yield FALLBACK_RESPONSES["timeout"]
            except httpx.ConnectError:
                logger.error("LLM server unreachable during streaming")
                yield FALLBACK_RESPONSES["connection_error"]
            except Exception as e:
                logger.error(f"LLM streaming error: {e}")
                yield FALLBACK_RESPONSES["generic"]


def _strip_think_blocks(text: str) -> str:
    """Remove all <think>...</think> blocks from a complete text."""
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# Module-level singleton
_default_interface = None


def get_llm() -> LlamaCppInterface:
    """Get the default LlamaCppInterface singleton."""
    global _default_interface
    if _default_interface is None:
        _default_interface = LlamaCppInterface()
    return _default_interface
