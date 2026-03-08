@echo off
set MODEL_PATH="..\qwen3-4b-q4_k_m.gguf"
set SERVER_EXE="llama.cpp\build\bin\llama-server.exe"

echo Starting Llama.cpp Server with Qwen3 4B Model...
echo Model: %MODEL_PATH%
echo Executable: %SERVER_EXE%

%SERVER_EXE% -m %MODEL_PATH% -c 4096 -ngl 33 --port 8080
pause
