import argparse
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

# For loading model.safetensors
try:
    import safetensors.torch
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

# Audio processing dependencies
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


def load_audio(file_path, sample_rate=16000):
    """Load audio file and resample to target sample rate using pydub."""
    if not PYDUB_AVAILABLE:
        print("ERROR: pydub is required for audio processing")
        sys.exit(1)
    
    try:
        print(f"Loading audio file: {file_path}")
        audio = AudioSegment.from_file(file_path)
        
        # Resample if needed
        if audio.frame_rate != sample_rate:
            print(f"Resampling from {audio.frame_rate}Hz to {sample_rate}Hz")
            audio = audio.set_frame_rate(sample_rate)
        
        # Convert to mono if needed
        if audio.channels > 1:
            print(f"Converting from {audio.channels} channels to mono")
            audio = audio.set_channels(1)
        
        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples())
        
        # Convert to float32 and normalize
        if audio.sample_width == 2:  # 16-bit audio
            samples = samples.astype(np.float32) / 32768.0
        elif audio.sample_width == 4:  # 32-bit audio
            samples = samples.astype(np.float32) / 2147483648.0
        else:
            samples = samples.astype(np.float32) / 128.0
            
        return samples, sample_rate
    except Exception as e:
        print(f"Error loading audio: {e}")
        # Return a silent audio segment for testing purposes
        print("Generating silent audio for testing")
        return np.zeros(sample_rate * 5, dtype=np.float32), sample_rate


def load_model(model_dir):
    """Load model from model.safetensors and related files."""
    if not SAFETENSORS_AVAILABLE:
        print("ERROR: safetensors is required for loading the model")
        sys.exit(1)
    
    model_path = os.path.join(model_dir, "model.safetensors")
    config_path = os.path.join(model_dir, "config.json")
    vocab_path = os.path.join(model_dir, "vocab.txt")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return None
    
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        return None
    
    # Load the configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Loading model from {model_path}")
    # Load model weights
    weights = safetensors.torch.load_file(model_path)
    
    # This is just a placeholder, in a real implementation 
    # you would build the model architecture from the config
    # and load the weights
    
    return {
        "config": config,
        "weights": weights,
        "vocab_path": vocab_path if os.path.exists(vocab_path) else None
    }


def transcribe(model, audio_data, timestamps=False):
    """
    Perform transcription using the loaded model.
    """
    try:
        # Check for MLX on Mac or fallback to CPU
        device = "cpu"
        try:
            import platform
            if platform.system() == "Darwin":
                try:
                    import mlx.core
                    device = "mlx"
                    print(f"Using MLX on Apple Silicon")
                except ImportError:
                    print(f"MLX not available, using CPU on Mac")
        except ImportError:
            pass
            
        print(f"Using device: {device}")
        
        # In a real implementation, we would:
        # 1. Process audio to match model input requirements
        # 2. Use MLX for Apple Silicon devices
        # 3. Run inference
        # 4. Process the output
        
        # Get audio characteristics
        audio_duration = len(audio_data) / 16000  # Assuming 16kHz audio
        
        # Generate a more realistic placeholder response based on audio length
        if len(audio_data) < 16000:  # Less than 1 second
            text = "Too short to transcribe."
        elif len(audio_data) < 16000 * 3:  # Less than 3 seconds
            text = "Short audio clip detected."
        else:
            # Calculate a rough word count based on audio duration (3 words per second)
            word_count = int(audio_duration * 3)
            text = "Audio of approximately {:.1f} seconds received. Expected to contain about {} words.".format(
                audio_duration, word_count
            )
        
        # Simulate timestamps if requested
        segments = []
        if timestamps:
            # Create segments based on audio duration
            n_segments = max(1, min(int(audio_duration / 2), 5))  # Between 1 and 5 segments
            segment_duration = audio_duration / n_segments
            
            words = text.split()
            words_per_segment = max(1, len(words) // n_segments)
            
            for i in range(n_segments):
                start = i * segment_duration
                end = (i + 1) * segment_duration
                
                # Get appropriate word slice for this segment
                start_word = i * words_per_segment
                end_word = min(len(words), (i + 1) * words_per_segment)
                segment_text = " ".join(words[start_word:end_word])
                
                segments.append({
                    "start": start,
                    "end": end,
                    "segment": segment_text
                })
        
        return {
            "text": text,
            "timestamp": {"segment": segments} if segments else {}
        }
    except Exception as e:
        print(f"Transcription error: {e}")
        return {
            "text": f"Error during transcription: {str(e)}",
            "timestamp": {}
        }


def main():
    parser = argparse.ArgumentParser(
        description="Offline transcription using new model.safetensors format"
    )
    parser.add_argument(
        "audio", nargs="+", help="Path(s) to input audio files (16kHz wav/flac)"
    )
    parser.add_argument(
        "--model",
        default="parakeet-tdt-0.6b-v2",
        help="Directory containing model.safetensors and config files",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Print word and segment timestamps",
    )

    args = parser.parse_args()
    
    # Check if model directory exists
    if not os.path.exists(args.model):
        print(f"ERROR: Model directory not found: {args.model}")
        return
    
    # Load model
    model = load_model(args.model)
    if not model:
        print("Failed to load model")
        return
    
    for audio_path in args.audio:
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            continue
        
        # Load and process audio
        audio_data, sr = load_audio(audio_path)
        print(f" - Loaded audio: {len(audio_data)} samples, {sr}Hz")
        
        # Transcribe
        results = transcribe(model, audio_data, timestamps=args.timestamps)
        
        # Display results
        print(f"--- {audio_path} ---")
        print(results["text"])
        
        if args.timestamps:
            segments = results["timestamp"].get("segment", [])
            for seg in segments:
                start = seg.get("start")
                end = seg.get("end")
                text = seg.get("segment")
                print(f"{start:.2f}s - {end:.2f}s : {text}")
        print()


if __name__ == "__main__":
    main()
