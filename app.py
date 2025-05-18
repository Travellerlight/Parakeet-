import argparse
import gradio as gr
import nemo.collections.asr as nemo_asr


def load_model(model_path: str):
    """Load the ASR model from the given .nemo file."""
    return nemo_asr.models.ASRModel.restore_from(model_path)


def transcribe(audio_path: str, model) -> str:
    """Transcribe a single audio file path using the model."""
    if not audio_path:
        return ""
    result = model.transcribe([audio_path])[0]
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Launch a simple UI for offline transcription"
    )
    parser.add_argument(
        "--model",
        default="parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo",
        help="Path to the .nemo model file",
    )
    args = parser.parse_args()

    model = load_model(args.model)

    interface = gr.Interface(
        fn=lambda audio: transcribe(audio, model),
        inputs=gr.Audio(source="upload", type="filepath"),
        outputs="textbox",
        title="Parakeet Offline Transcription",
        description="Upload a 16kHz wav/flac file to transcribe",
    )
    interface.launch()


if __name__ == "__main__":
    main()
