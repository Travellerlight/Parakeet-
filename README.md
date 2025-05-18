# Parakeet Offline Transcription

This repository provides a simple command line interface for offline speech transcription using a model.safetensors format. The model files are not committed to git and must be downloaded separately.

## Requirements

- Python 3.8 or later
- [PyTorch](https://pytorch.org/)
- SafeTensors
- pydub (for audio processing)
- ffmpeg (for audio file handling)

Install dependencies with:

```bash
pip install torch safetensors pydub
```

You'll also need to install ffmpeg on your system.

## Model Setup

Place the following files in the `parakeet-tdt-0.6b-v2/` directory:
- `model.safetensors` - The model weights
- `config.json` - Model configuration
- `vocab.txt` - Vocabulary file
- `tokenizer.model` - Tokenizer model (if needed)

## Usage

Run the transcription script:

```bash
python offline_transcribe.py <audio files>
```

Use `--timestamps` to include timestamp information. Example:

```bash
python offline_transcribe.py --timestamps example.wav
```

The script prints the transcription and, when requested, segment timestamps for each provided audio file.

## Docker Usage

The project includes Docker support for easy deployment:

1. Place your audio files in the `input` directory.
2. Make sure the model files are in the correct location (`parakeet-tdt-0.6b-v2/` directory).
3. Build and run using Docker Compose:

```bash
docker-compose build
docker-compose run parakeet input/<audio-file>
```

To include timestamps:

```bash
docker-compose run parakeet --timestamps input/<audio-file>
```

Requirements for Docker:
- Docker and Docker Compose
