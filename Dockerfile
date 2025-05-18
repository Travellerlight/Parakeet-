FROM nvcr.io/nvidia/nemo:23.06

WORKDIR /app

# Install additional dependencies
RUN pip install --no-cache-dir nemo_toolkit[asr]

# Copy project files
COPY . .

# Set entrypoint to the transcription script
ENTRYPOINT ["python", "offline_transcribe.py"] 