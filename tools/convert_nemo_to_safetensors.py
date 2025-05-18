#!/usr/bin/env python3
"""
Conversion utility to convert NVIDIA Parakeet model from NeMo format to safetensors format.

Usage:
  python convert_nemo_to_safetensors.py --nemo-path /path/to/parakeet-tdt-0.6b-v2.nemo --output-dir ./parakeet-tdt-0.6b-v2/

Requirements:
  - nemo_toolkit (for loading .nemo files)
  - safetensors (for saving tensors)
  - torch (for tensor operations)
"""

import os
import sys
import json
import argparse
import tarfile
import tempfile
import shutil
from pathlib import Path

try:
    import torch
    import safetensors.torch
except ImportError:
    print("ERROR: This tool requires PyTorch and safetensors. Install with:")
    print("pip install torch safetensors")
    sys.exit(1)

try:
    import nemo
    import nemo.collections.asr as nemo_asr
except ImportError:
    print("ERROR: NeMo toolkit is required to convert .nemo files. Install with:")
    print("pip install nemo_toolkit[asr]")
    sys.exit(1)

def extract_nemo_files(nemo_path, temp_dir):
    """Extract files from the .nemo archive to a temporary directory."""
    print(f"Extracting NEMO file: {nemo_path}")
    
    try:
        with tarfile.open(nemo_path, 'r') as tar:
            tar.extractall(path=temp_dir)
        print(f"  Extracted to: {temp_dir}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to extract NEMO file: {e}")
        return False

def convert_model(nemo_path, output_dir):
    """Convert NVIDIA Parakeet model from NeMo format to safetensors format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        # Step 1: Extract the NEMO file
        if not extract_nemo_files(nemo_path, temp_dir):
            return False
        
        # Step 2: Find model weights
        model_dir = Path(temp_dir)
        model_weights = list(model_dir.glob("*.ckpt"))
        
        if not model_weights:
            print("ERROR: No model checkpoint found in the NEMO file")
            return False
        
        weight_file = model_weights[0]
        print(f"Found model weights: {weight_file.name}")
        
        # Step 3: Find model config
        model_config = list(model_dir.glob("model_config.yaml"))
        
        if not model_config:
            print("ERROR: No model configuration found in the NEMO file")
            return False
        
        config_file = model_config[0]
        print(f"Found model config: {config_file.name}")
        
        # Step 4: Load the model using NeMo
        try:
            print("Loading ASR model using NeMo toolkit...")
            asr_model = nemo_asr.models.EncDecCTCModel.restore_from(nemo_path)
            print("Model loaded successfully")
            
            # Step 5: Convert weights to a dictionary
            print("Converting model to PyTorch state dict...")
            model_state_dict = asr_model.state_dict()
            
            # Step 6: Convert state dict to a format compatible with safetensors
            filtered_state_dict = {}
            for key, tensor in model_state_dict.items():
                if isinstance(tensor, torch.Tensor):
                    filtered_state_dict[key] = tensor
            
            # Step 7: Save the state dict using safetensors
            output_safetensors = output_dir / "model.safetensors"
            print(f"Saving model weights to {output_safetensors}")
            safetensors.torch.save_file(filtered_state_dict, output_safetensors)
            
            # Step 8: Extract and save the vocab file
            vocab_file = model_dir / "vocab.txt"
            if vocab_file.exists():
                shutil.copy(vocab_file, output_dir / "vocab.txt")
                print(f"Copied vocab file to {output_dir / 'vocab.txt'}")
            else:
                print("WARNING: No vocab.txt found in the NEMO file")
            
            # Step 9: Convert the model config to JSON
            try:
                config = asr_model.cfg.to_dict()
                config_json = output_dir / "config.json"
                with open(config_json, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"Saved model configuration to {config_json}")
            except Exception as e:
                print(f"WARNING: Failed to save config.json: {e}")
            
            print("\nConversion complete!")
            print(f"Files saved to: {output_dir}")
            print("You can now use these files with the offline_transcribe.py script")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to load or convert the model: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(
        description="Convert NVIDIA Parakeet model from NeMo format to safetensors format"
    )
    parser.add_argument(
        "--nemo-path", 
        required=True,
        help="Path to the .nemo model file"
    )
    parser.add_argument(
        "--output-dir",
        default="parakeet-tdt-0.6b-v2",
        help="Directory to save the converted model files (default: parakeet-tdt-0.6b-v2)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.nemo_path):
        print(f"ERROR: Input file not found: {args.nemo_path}")
        return 1
    
    if not args.nemo_path.endswith('.nemo'):
        print(f"WARNING: Input file does not have .nemo extension: {args.nemo_path}")
    
    # Convert the model
    success = convert_model(args.nemo_path, args.output_dir)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 