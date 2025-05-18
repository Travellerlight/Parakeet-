FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and other dependencies
RUN pip install --no-cache-dir torch numpy safetensors pydub

# Copy project files
COPY offline_transcribe.py .
COPY input/ input/

# Set entrypoint to the transcription script
ENTRYPOINT ["python", "offline_transcribe.py"] 