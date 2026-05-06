#  Use a lightweight Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1  
ENV PYTHONUNBUFFERED=1

# Default LLM configurations (point to host machine from within Docker)
ENV LM_STUDIO_BASE_URL="http://host.docker.internal:1234/v1"
ENV LM_STUDIO_MODEL="qwen/qwen3-4b"

# Set working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

# Copy the rest of the application
COPY . .

# Run the model download script during build to bake models into the image
RUN python scripts/download_models.py

# Expose port for Uvicorn
EXPOSE 8000

# Command to run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]