import asyncio
import httpx
import json

async def main():
    async with httpx.AsyncClient() as client:
        try:
            async with client.stream(
                "POST", "http://127.0.0.1:1234/v1/chat/completions",
                json={
                    "model": "qwen/qwen3-4b",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True
                }
            ) as resp:
                with open("test_lm_output.txt", "w", encoding="utf-8") as f:
                    async for chunk in resp.aiter_lines():
                        f.write(repr(chunk) + "\n")
        except Exception as e:
            with open("test_lm_output.txt", "a", encoding="utf-8") as f:
                f.write(f"Error: {e}\n")

if __name__ == "__main__":
    asyncio.run(main())
