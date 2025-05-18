import os
import json
import uuid
import torch
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from pydub import AudioSegment

# Import functions from offline_transcribe.py
from offline_transcribe import load_model, transcribe, load_audio

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model on startup
MODEL_DIR = 'parakeet-tdt-0.6b-v2'
model = load_model(MODEL_DIR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

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
        filename = str(uuid.uuid4()) + '.' + secure_filename(file.filename).rsplit('.', 1)[1].lower()
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
            
            # Return results
            return jsonify({
                'filename': filename,
                'text': text,
                'segments': segments,
                'success': True
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Unknown error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000) 