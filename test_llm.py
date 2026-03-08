import requests
import json
import time

URL = "http://localhost:8080/v1/chat/completions"
PAYLOAD = {
    "model": "qwen3-4b-q4_k_m.gguf",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! Are you there?"}
    ],
    "temperature": 0.7,
    "max_tokens": 50,
    "stream": False
}

print(f"Sending request to {URL}...")
start_time = time.time()
try:
    response = requests.post(URL, json=PAYLOAD, timeout=120)
    response.raise_for_status()
    result = response.json()
    elapsed = time.time() - start_time
    print(f"Success! Response received in {elapsed:.2f} seconds.")
    print("Content:")
    print(result["choices"][0]["message"]["content"])
except Exception as e:
    print(f"Error: {e}")
