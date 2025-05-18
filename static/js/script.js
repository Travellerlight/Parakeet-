document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const fileSelect = document.getElementById('file-select');
    const fileInfo = document.getElementById('file-info');
    const fileName = document.getElementById('file-name');
    const removeFileBtn = document.getElementById('remove-file');
    const transcribeBtn = document.getElementById('transcribe-btn');
    const uploadForm = document.getElementById('upload-form');
    const transcriptionContainer = document.getElementById('transcription-container');
    const loadingIndicator = document.getElementById('loading-indicator');
    const results = document.getElementById('results');
    const transcriptionText = document.getElementById('transcription-text');
    const timestampsContainer = document.getElementById('timestamps-container');
    const timestampsList = document.getElementById('timestamps-list');
    const audioElement = document.getElementById('audio-element');
    const errorMessage = document.getElementById('error-message');
    const errorText = document.getElementById('error-text');
    const tryAgainBtn = document.getElementById('try-again');
    const newTranscriptionBtn = document.getElementById('new-transcription');
    const modelStatusIndicator = document.getElementById('model-status-indicator');

    // File handling
    let selectedFile = null;

    // Event listeners for drag and drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        dropArea.classList.add('dragover');
    }

    function unhighlight() {
        dropArea.classList.remove('dragover');
    }

    // Handle dropped files
    dropArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length) {
            handleFiles(files);
        }
    }

    // Handle file selection
    fileSelect.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            handleFiles(fileInput.files);
        }
    });

    function handleFiles(files) {
        selectedFile = files[0];
        updateFileInfo();
    }

    // Update file info UI
    function updateFileInfo() {
        if (selectedFile) {
            fileName.textContent = selectedFile.name;
            fileInfo.classList.remove('hidden');
            transcribeBtn.disabled = false;
        } else {
            fileInfo.classList.add('hidden');
            transcribeBtn.disabled = true;
        }
    }

    // Remove selected file
    removeFileBtn.addEventListener('click', () => {
        selectedFile = null;
        fileInput.value = '';
        updateFileInfo();
    });

    // Form submission
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (!selectedFile) {
            return;
        }

        // Show loading state
        transcriptionContainer.classList.remove('hidden');
        loadingIndicator.classList.remove('hidden');
        results.classList.add('hidden');
        errorMessage.classList.add('hidden');

        // Create form data
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        // Add timestamps checkbox value
        const includeTimestamps = document.getElementById('include-timestamps');
        formData.append('timestamps', includeTimestamps.checked);

        try {
            // Send request
            const response = await fetch('/transcribe', {
                method: 'POST',
                body: formData
            });

            // Check for errors
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Server responded with an error');
            }

            // Parse response
            const data = await response.json();
            
            // Display results
            displayResults(data);
        } catch (error) {
            showError(error.message);
        }
    });

    // Display transcription results
    function displayResults(data) {
        // Hide loading indicator
        loadingIndicator.classList.add('hidden');
        
        // Show results
        results.classList.remove('hidden');
        
        // Update transcription text
        transcriptionText.textContent = data.text;
        
        // Show model status if using dummy model
        if (data.dummy_model) {
            modelStatusIndicator.classList.remove('hidden');
        } else {
            modelStatusIndicator.classList.add('hidden');
        }
        
        // Update audio player
        audioElement.src = `/uploads/${data.filename}`;
        
        // Handle timestamps if present
        if (data.segments && data.segments.length) {
            timestampsContainer.classList.remove('hidden');
            timestampsList.innerHTML = '';
            
            // Create timestamp items
            data.segments.forEach(segment => {
                const timestampItem = document.createElement('div');
                timestampItem.classList.add('timestamp-item');
                
                const timestampTime = document.createElement('div');
                timestampTime.classList.add('timestamp-time');
                timestampTime.textContent = `${segment.start.toFixed(2)}s - ${segment.end.toFixed(2)}s`;
                
                const timestampText = document.createElement('div');
                timestampText.classList.add('timestamp-text');
                timestampText.textContent = segment.text;
                
                timestampItem.appendChild(timestampTime);
                timestampItem.appendChild(timestampText);
                timestampsList.appendChild(timestampItem);
                
                // Add click event to seek to timestamp
                timestampItem.addEventListener('click', () => {
                    audioElement.currentTime = segment.start;
                    audioElement.play();
                });
            });
        } else {
            timestampsContainer.classList.add('hidden');
        }
    }

    // Show error message
    function showError(message) {
        loadingIndicator.classList.add('hidden');
        results.classList.add('hidden');
        errorMessage.classList.remove('hidden');
        errorText.textContent = message || 'Something went wrong. Please try again.';
    }

    // Try again button
    tryAgainBtn.addEventListener('click', () => {
        transcriptionContainer.classList.add('hidden');
    });

    // New transcription button
    newTranscriptionBtn.addEventListener('click', () => {
        // Reset form
        selectedFile = null;
        fileInput.value = '';
        updateFileInfo();
        
        // Hide transcription container
        transcriptionContainer.classList.add('hidden');
    });
    
    // Add click handlers to any audio timestamps
    document.addEventListener('click', (e) => {
        if (e.target.closest('.timestamp-item')) {
            const timestampItem = e.target.closest('.timestamp-item');
            const timeText = timestampItem.querySelector('.timestamp-time').textContent;
            const startTime = parseFloat(timeText.split('-')[0]);
            
            if (!isNaN(startTime) && audioElement) {
                audioElement.currentTime = startTime;
                audioElement.play();
            }
        }
    });
}); 