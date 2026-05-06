import httpx
import os
import json
from typing import AsyncGenerator

BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://host.docker.internal:1234/v1")
MODEL = os.getenv("LM_STUDIO_MODEL", "qwen/qwen3-0.6b")

# ---------- Non-streaming Agent (Memory Extraction) ----------
async def chat_completion(messages: list) -> str:
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        resp = await client.post(
            f"{BASE_URL}/chat/completions",
            json={
                "model": MODEL,
                "messages": messages,
                "temperature": 0.5,
                "top_p": 0.8,
                "top_k": 15,
                "repeat_penalty": 1.1,
                "max_tokens": 256,
                "stream": False,
            },
        )

        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ---------- Streaming Agent (Chat Response Generator) ----------
async def chat_completion_stream(messages: list) -> AsyncGenerator[str, None]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        async with client.stream(
            "POST",
            f"{BASE_URL}/chat/completions",
            json={
                "model": MODEL,
                "messages": messages,
                "temperature": 0.5,
                "top_p": 0.8,
                "top_k": 15,
                "repeat_penalty": 1.1,
                "max_tokens": 256,
                "stream": True,
            },
        ) as resp:

            resp.raise_for_status()

            async for line in resp.aiter_lines():
                if not line:
                    continue

                if line.startswith("data: ") and "[DONE]" not in line:
                    try:
                        chunk = json.loads(line[6:])
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except:
                        continue