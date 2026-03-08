$ModelPath = "llama.cpp\models\qwen3-4b-q4_k_m.gguf"
$ServerExe = "llama.cpp\build\bin\llama-server.exe"

Write-Host "Starting Llama.cpp Server with Qwen3 4B Model..." -ForegroundColor Green
Write-Host "Model: $ModelPath" -ForegroundColor Cyan
Write-Host "Executable: $ServerExe" -ForegroundColor Cyan

# Optimized for extreme CPU limitations
& $ServerExe -m $ModelPath -c 512 -t 4 -b 128 --threads-batch 4 --timeout 600 --port 8080
