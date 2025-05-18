# Parakeet Offline Transcription

This repository provides a simple command line interface for offline speech transcription using NVIDIA's **parakeet-tdt-0.6b-v2** model. The large model file (`.nemo`) is not committed to git and must be downloaded separately.

## Requirements

- Python 3.8 or later
- [PyTorch](https://pytorch.org/) with GPU support
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) toolkit

Install NeMo (after PyTorch) with:

```bash
pip install -U nemo_toolkit[asr]
```

## Usage

Place `parakeet-tdt-0.6b-v2.nemo` inside the `parakeet-tdt-0.6b-v2/` directory and run:

```bash
python offline_transcribe.py <audio files>
```

Use `--timestamps` to include timestamp information and `--model` to specify a different path to the model file. Example:

```bash
python offline_transcribe.py --timestamps example.wav
```

The script prints the transcription and, when requested, segment timestamps for each provided audio file.

## Web Interface

You can also launch a minimal web UI using [Gradio](https://gradio.app/). Install
Gradio and start the app:

```bash
pip install gradio
python app.py
```

Upload a 16&nbsp;kHz WAV or FLAC file to transcribe it offline using the same Parakeet model.
