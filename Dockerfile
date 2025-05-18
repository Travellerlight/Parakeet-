FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Mac-specific dependencies when on Mac
ARG TARGETPLATFORM
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ] || [ "$TARGETPLATFORM" = "darwin" ] || [ -z "$TARGETPLATFORM" -a "$(uname -m)" = "arm64" ]; then \
    pip install --no-cache-dir mlx || echo "MLX installation skipped (only works on Apple Silicon)"; \
    fi

# Copy project files
COPY offline_transcribe.py app.py simple_transcribe.py ./
COPY templates templates/
COPY static static/
COPY tools tools/

# Create necessary directories
RUN mkdir -p input uploads output

# Expose port for web interface
EXPOSE 3000

# Set entrypoint to the Flask app
CMD ["python", "app.py"] 