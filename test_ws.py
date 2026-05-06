import asyncio
import websockets
import json
import uuid

async def test_websocket():
    session_id = str(uuid.uuid4())
    uri = f"ws://127.0.0.1:8000/ws/chat/{session_id}"
    
    # 1. First HTTP POST to init session if needed (it creates the session in dict)
    import httpx
    async with httpx.AsyncClient() as client:
        await client.post("http://127.0.0.1:8000/session/new")

    print("Connecting to WEBSOCKET")
    async with websockets.connect(uri) as websocket:
        print("Connected. Sending message...")
        await websocket.send("Hi")
        
        while True:
            try:
                msg = await websocket.recv()
                print(f"Received chunk: {repr(msg)}")
                if msg == "[END]":
                    break
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed unexpectedly")
                break

if __name__ == "__main__":
    asyncio.run(test_websocket())
