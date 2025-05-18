# Parakeet Offline Transcription

This repository provides a simple interface for offline speech transcription using a model.safetensors format. The model files are not committed to git and must be downloaded separately.

## Requirements

- Python 3.8 or later
- [PyTorch](https://pytorch.org/)
- SafeTensors
- NumPy
- pydub (for audio processing)
- ffmpeg (for audio file handling)
- Flask (for web interface)
- MLX (for Apple Silicon Macs)

Install dependencies with:

```bash
pip install -r requirements.txt
```

For Apple Silicon Mac users, also install MLX:

```bash
pip install mlx
```

You'll also need to install ffmpeg on your system.

## Model Setup

Place the following files in the `parakeet-tdt-0.6b-v2/` directory:
- `model.safetensors` - The model weights
- `config.json` - Model configuration
- `vocab.txt` - Vocabulary file
- `tokenizer.model` - Tokenizer model (if needed)

## Usage

### Command Line Interface

Run the transcription script:

```bash
python offline_transcribe.py <audio files>
```

Use `--timestamps` to include timestamp information. Example:

```bash
python offline_transcribe.py --timestamps example.wav
```

The script prints the transcription and, when requested, segment timestamps for each provided audio file.

### Web Interface

Run the Flask web application:

```bash
python app.py
```

Then access the web interface at: http://localhost:3000

The web interface allows you to:
- Upload audio files through the browser
- View transcription results
- See timestamps (optional)
- Play back the audio with timestamps that are clickable to jump to specific positions

## Docker Usage

The project includes Docker support for easy deployment on both Intel/AMD and Apple Silicon Macs:

1. Make sure the model files are in the correct location (`parakeet-tdt-0.6b-v2/` directory).
2. Build and run using Docker Compose:

```bash
docker-compose build
docker-compose up
```

This will start both the command-line tool and the web interface.

Access the web interface at: http://localhost:3000

To run the command-line transcription instead:

```bash
docker-compose run parakeet python offline_transcribe.py input/<audio-file>
```

To include timestamps:

```bash
docker-compose run parakeet python offline_transcribe.py --timestamps input/<audio-file>
```

Requirements for Docker:
- Docker and Docker Compose

### Apple Silicon Mac Notes

The application now detects Apple Silicon and uses MLX when available. This allows faster transcription on M1/M2/M3 Macs without requiring CUDA.
