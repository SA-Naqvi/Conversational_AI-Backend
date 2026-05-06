"""
Locustfile for load testing the chatbot WebSocket endpoint.

Usage:
  locust -f locustfile.py --host=ws://localhost:8000 --users 10 --spawn-rate 2
"""
import json
import time
import uuid
import asyncio
from locust import User, task, between, events

try:
    import websockets
except ImportError:
    websockets = None


class ChatbotUser(User):
    """Simulates a patient interacting with the chatbot."""
    wait_time = between(1, 3)

    INTAKE_SCRIPT = [
        "I had knee surgery",
        "January 15, 2025",
        "My pain is 4/10",
    ]

    MONITORING_QUESTIONS = [
        "How should I care for my wound?",
        "What exercises can I do?",
        "What are the side effects of ibuprofen?",
        "My temperature is 99.5F today",
        "When should I schedule a follow-up?",
    ]

    def on_start(self):
        """Setup — initialize session."""
        self.session_id = f"locust-{uuid.uuid4().hex[:8]}"
        self._intake_done = False
        self._loop = asyncio.new_event_loop()

    def on_stop(self):
        self._loop.close()

    def _run_async(self, coro):
        return self._loop.run_until_complete(coro)

    async def _send_and_receive(self, message: str) -> dict:
        """Send a message via WebSocket and collect response + timing."""
        host = self.host or "ws://localhost:8000"
        uri = f"{host}/ws/chat/{self.session_id}?eval_mode=true"

        start = time.time()
        response_text = ""
        metrics = None

        try:
            async with websockets.connect(uri, close_timeout=5) as ws:
                await ws.send(message)

                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=120)
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
                    response_text += msg

            elapsed_ms = (time.time() - start) * 1000
            return {
                "success": True,
                "elapsed_ms": elapsed_ms,
                "response_length": len(response_text),
                "metrics": metrics,
            }
        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            return {
                "success": False,
                "elapsed_ms": elapsed_ms,
                "error": str(e),
            }

    @task(3)
    def chat_message(self):
        """Send a chat message and record timing."""
        import random

        if not self._intake_done:
            # Do intake first
            for msg in self.INTAKE_SCRIPT:
                result = self._run_async(self._send_and_receive(msg))
                events.request.fire(
                    request_type="WebSocket",
                    name="intake_turn",
                    response_time=result["elapsed_ms"],
                    response_length=result.get("response_length", 0),
                    exception=None if result["success"] else Exception(result.get("error")),
                )
            self._intake_done = True
            return

        # Random monitoring question
        question = random.choice(self.MONITORING_QUESTIONS)
        result = self._run_async(self._send_and_receive(question))

        events.request.fire(
            request_type="WebSocket",
            name="monitoring_turn",
            response_time=result["elapsed_ms"],
            response_length=result.get("response_length", 0),
            exception=None if result["success"] else Exception(result.get("error")),
        )
