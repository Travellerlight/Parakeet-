# Parakeet Offline Transcription

This repository provides a simple interface for offline speech transcription using the NVIDIA Parakeet model in safetensors format. The model files are not committed to git and must be downloaded separately.

## Requirements

- Python 3.8 or later
- [PyTorch](https://pytorch.org/)
- SafeTensors
- NumPy
- pydub (for audio processing)
- ffmpeg (for audio file handling)
- Flask (for web interface)
- MLX (for Apple Silicon Macs - optional)

Install dependencies with:

```bash
pip install -r requirements.txt
```

For Apple Silicon Mac users, also install MLX for better performance:

```bash
pip install mlx
```

Install ffmpeg (required for audio processing):
- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH

## Model Setup

You need to place the NVIDIA Parakeet model files in the `parakeet-tdt-0.6b-v2/` directory:

1. Create the directory: `mkdir -p parakeet-tdt-0.6b-v2`
2. Download the following files and place them in this directory:
   - `model.safetensors` - The model weights
   - `config.json` - Model configuration
   - `vocab.txt` - Vocabulary file

**Note**: If these files are not present, the application will run in demonstration mode with simulated transcription output.

### Getting the Model Files

The original NVIDIA Parakeet model is available in NeMo format. To convert to safetensors format:

1. Download the NVIDIA Parakeet model from [NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/parakeet_tdt_0.6b)
2. Use the conversion script in the tools directory: `python tools/convert_nemo_to_safetensors.py`

Alternatively, you can:
- Access the model weights from the [Hugging Face Hub](https://huggingface.co/nvidia/parakeet-tdt-0.6b)
- Use GPU-accelerated transcription services like the NeMo API or NVIDIA's Riva SDK

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

## Status Indicators

The application will indicate its status:
- **Normal mode**: The model is loaded and ready for transcription
- **Demo mode**: Running with simulated transcription (placeholder text) because:
  - Model files are missing
  - Required dependencies are not installed
  - Model loading failed for some reason

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

The application automatically detects Apple Silicon and uses MLX when available. This allows faster transcription on M1/M2/M3 Macs without requiring CUDA.

## Limitations and Future Improvements

Currently, the application:
- Uses simulated transcription if model files are not available
- Supports common audio formats via pydub and ffmpeg
- Provides timestamps for the transcription
- Automatically cleans up uploaded files after 24 hours

Future improvements could include:
- Implementation of actual model inference using PyTorch
- Support for more languages
- Stream processing for long audio files
- User authentication for multi-user environments
