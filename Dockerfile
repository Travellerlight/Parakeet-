FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and other dependencies
RUN pip install --no-cache-dir torch numpy safetensors pydub flask

# Copy project files
COPY offline_transcribe.py app.py ./
COPY templates templates/
COPY static static/
COPY input/ input/
COPY uploads/ uploads/

# Expose port for web interface
EXPOSE 3000

# Set entrypoint to the Flask app
CMD ["python", "app.py"] 