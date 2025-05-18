import os
import json
import uuid
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import datetime

# Import functions from offline_transcribe.py
from offline_transcribe import load_model, transcribe, load_audio
# Get availability flags from the module
from offline_transcribe import SAFETENSORS_AVAILABLE, PYDUB_AVAILABLE, TORCH_AVAILABLE, MLX_AVAILABLE

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['UPLOAD_MAX_AGE'] = 24 * 60 * 60  # 24 hours in seconds

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model on startup
MODEL_DIR = 'parakeet-tdt-0.6b-v2'
model = None

def initialize_model():
    """Load model and return status information"""
    global model
    
    # Check for model directory
    if not os.path.exists(MODEL_DIR):
        print(f"WARNING: Model directory not found: {MODEL_DIR}")
        model_status = "missing_directory"
    else:
        # Try to load model
        model = load_model(MODEL_DIR)
        
        if model.get("dummy_model", True):
            if not SAFETENSORS_AVAILABLE:
                model_status = "missing_dependencies"
            else:
                model_status = "missing_files"
        else:
            model_status = "loaded"
    
    # If model is None or failed to load, use a dummy model
    if model is None:
        model = {
            "config": {"dummy": True},
            "weights": None,
            "vocab_path": None,
            "dummy_model": True
        }
    
    # Print model status
    if model.get("dummy_model", True):
        print("NOTICE: Using dummy transcription model (no actual speech recognition)")
        if model_status == "missing_directory":
            print(f"  - Model directory '{MODEL_DIR}' not found")
        elif model_status == "missing_dependencies":
            print("  - Required dependencies (safetensors) not available")
        elif model_status == "missing_files":
            print("  - Model files incomplete or corrupt")
    else:
        print("Model loaded successfully. Ready for transcription.")
    
    return model_status

model_status = initialize_model()

@app.route('/')
def index():
    return render_template('index.html', 
                         model_status=model_status,
                         dependencies={
                             "safetensors": SAFETENSORS_AVAILABLE,
                             "pydub": PYDUB_AVAILABLE,
                             "torch": TORCH_AVAILABLE,
                             "mlx": MLX_AVAILABLE
                         })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/status')
def status():
    return jsonify({
        "model_status": model_status,
        "dependencies": {
            "safetensors": SAFETENSORS_AVAILABLE,
            "pydub": PYDUB_AVAILABLE,
            "torch": TORCH_AVAILABLE,
            "mlx": MLX_AVAILABLE
        }
    })

@app.route('/transcribe', methods=['POST'])
def transcribe_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # If user does not select file, browser submits an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Check for valid file
    if file:
        # Generate unique filename
        extension = secure_filename(file.filename).rsplit('.', 1)[1].lower() if '.' in file.filename else 'wav'
        filename = str(uuid.uuid4()) + '.' + extension
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the file
        file.save(file_path)
        
        # Process with timestamps
        include_timestamps = request.form.get('timestamps', 'false').lower() == 'true'
        
        try:
            # Load and process audio
            audio_data, sr = load_audio(file_path)
            
            # Transcribe
            results = transcribe(model, audio_data, timestamps=include_timestamps)
            
            # Get the transcription text and timestamps
            text = results["text"]
            segments = []
            
            if include_timestamps:
                segments = results["timestamp"].get("segment", [])
                segments = [
                    {
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "text": seg.get("segment")
                    }
                    for seg in segments
                ]
            
            # Return results with model status
            return jsonify({
                'filename': filename,
                'text': text,
                'segments': segments,
                'success': True,
                'dummy_model': results.get("dummy_model", True),
                'model_status': model_status
            })
            
        except Exception as e:
            return jsonify({
                'error': str(e),
                'model_status': model_status,
                'dummy_model': True
            }), 500
    
    return jsonify({'error': 'Unknown error'}), 500

def cleanup_old_uploads():
    """Remove uploaded files older than the configured maximum age"""
    try:
        now = datetime.datetime.now()
        upload_dir = app.config['UPLOAD_FOLDER']
        max_age = app.config['UPLOAD_MAX_AGE']
        
        count = 0
        for filename in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, filename)
            if os.path.isfile(file_path):
                file_age = now - datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_age.total_seconds() > max_age:
                    os.remove(file_path)
                    count += 1
        
        if count > 0:
            print(f"Cleaned up {count} old files from uploads directory")
    except Exception as e:
        print(f"Error during cleanup: {e}")

@app.before_request
def before_request():
    # Run cleanup occasionally (approx once per 100 requests)
    import random
    if random.random() < 0.01:
        cleanup_old_uploads()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000) 