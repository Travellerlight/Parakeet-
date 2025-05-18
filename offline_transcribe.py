import argparse
import nemo.collections.asr as nemo_asr


def main():
    parser = argparse.ArgumentParser(
        description="Offline transcription using NVIDIA parakeet-tdt-0.6b-v2"
    )
    parser.add_argument(
        "audio", nargs="+", help="Path(s) to input audio files (16kHz wav/flac)"
    )
    parser.add_argument(
        "--model",
        default="parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo",
        help="Path to the .nemo model file",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Print word and segment timestamps",
    )

    args = parser.parse_args()

    asr_model = nemo_asr.models.ASRModel.restore_from(args.model)

    results = asr_model.transcribe(args.audio, timestamps=args.timestamps)

    for path, res in zip(args.audio, results):
        print(f"--- {path} ---")
        print(res.text)
        if args.timestamps:
            segments = res.timestamp.get("segment", [])
            for seg in segments:
                start = seg.get("start")
                end = seg.get("end")
                text = seg.get("segment")
                print(f"{start:.2f}s - {end:.2f}s : {text}")
        print()


if __name__ == "__main__":
    main()
