document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    const tabUpload = document.getElementById('tabUpload');
    const tabRecord = document.getElementById('tabRecord');
    const contentUpload = document.getElementById('contentUpload');
    const contentRecord = document.getElementById('contentRecord');
    
    const dropZone = document.getElementById('dropZone');
    const audioFileInput = document.getElementById('audioFileInput');
    const filePreview = document.getElementById('filePreview');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const removeFileBtn = document.getElementById('removeFileBtn');
    const audioPlayer = document.getElementById('audioPlayer');

    const micRecordBtn = document.getElementById('micRecordBtn');
    const recordTimer = document.getElementById('recordTimer');
    const recordStatus = document.getElementById('recordStatus');
    const recordedPreview = document.getElementById('recordedPreview');
    const recordedAudioPlayer = document.getElementById('recordedAudioPlayer');

    const beamSizeInput = document.getElementById('beamSizeInput');
    const vadFilterInput = document.getElementById('vadFilterInput');
    const wordTimestampsInput = document.getElementById('wordTimestampsInput');
    const transcribeBtn = document.getElementById('transcribeBtn');

    const emptyState = document.getElementById('emptyState');
    const loadingState = document.getElementById('loadingState');
    const resultContainer = document.getElementById('resultContainer');
    const elapsedTimer = document.getElementById('elapsedTimer');

    const statDuration = document.getElementById('statDuration');
    const statProcTime = document.getElementById('statProcTime');
    const fullTranscriptText = document.getElementById('fullTranscriptText');
    const segmentCount = document.getElementById('segmentCount');
    const segmentsList = document.getElementById('segmentsList');

    const copyTextBtn = document.getElementById('copyTextBtn');
    const downloadTxtBtn = document.getElementById('downloadTxtBtn');
    const downloadJsonBtn = document.getElementById('downloadJsonBtn');

    // State Variables
    let selectedFile = null;
    let recordedBlob = null;
    let currentMode = 'upload'; // 'upload' | 'record'
    let mediaRecorder = null;
    let audioChunks = [];
    let recordInterval = null;
    let recordSeconds = 0;
    let lastResult = null;
    let elapsedInterval = null;

    // Check Server Health
    async function checkHealth() {
        try {
            const res = await fetch('/health');
            if (res.ok) {
                const data = await res.json();
                statusDot.className = 'status-dot online';
                statusText.textContent = `Online (${data.device.toUpperCase()})`;
            } else {
                throw new Error('Health check failed');
            }
        } catch (e) {
            statusDot.className = 'status-dot pulsing';
            statusText.textContent = 'Alustetaan mallia...';
        }
    }
    checkHealth();
    setInterval(checkHealth, 10000);

    // Tab Switching
    tabUpload.addEventListener('click', () => {
        currentMode = 'upload';
        tabUpload.classList.add('active');
        tabRecord.classList.remove('active');
        contentUpload.classList.add('active');
        contentRecord.classList.remove('active');
        updateSubmitButtonState();
    });

    tabRecord.addEventListener('click', () => {
        currentMode = 'record';
        tabRecord.classList.add('active');
        tabUpload.classList.remove('active');
        contentRecord.classList.add('active');
        contentUpload.classList.remove('active');
        updateSubmitButtonState();
    });

    // File Upload Handlers
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleSelectedFile(files[0]);
        }
    });

    audioFileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    function handleSelectedFile(file) {
        if (!file.type.startsWith('audio/') && !file.name.match(/\.(wav|mp3|m4a|ogg|webm|flac|aac)$/i)) {
            alert('Valitse kelvollinen äänitiedosto!');
            return;
        }
        selectedFile = file;
        fileName.textContent = file.name;
        fileSize.textContent = formatBytes(file.size);
        
        const url = URL.createObjectURL(file);
        audioPlayer.src = url;
        
        dropZone.classList.add('hidden');
        filePreview.classList.remove('hidden');
        updateSubmitButtonState();
    }

    removeFileBtn.addEventListener('click', () => {
        selectedFile = null;
        audioFileInput.value = '';
        audioPlayer.src = '';
        filePreview.classList.add('hidden');
        dropZone.classList.remove('hidden');
        updateSubmitButtonState();
    });

    // Audio Recorder Handlers
    micRecordBtn.addEventListener('click', async () => {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            stopRecording();
        } else {
            await startRecording();
        }
    });

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioChunks = [];
            mediaRecorder = new MediaRecorder(stream);

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    audioChunks.push(e.data);
                }
            };

            mediaRecorder.onstop = () => {
                recordedBlob = new Blob(audioChunks, { type: 'audio/webm' });
                const audioUrl = URL.createObjectURL(recordedBlob);
                recordedAudioPlayer.src = audioUrl;
                recordedPreview.classList.remove('hidden');
                recordStatus.textContent = 'Nauhoitus valmis. Voit nyt tunnistaa puheen.';
                updateSubmitButtonState();
            };

            mediaRecorder.start();
            micRecordBtn.classList.add('recording');
            recordStatus.textContent = 'Nauhoitetaan puhetta... Napsauta painiketta lopettaaksesi.';
            recordSeconds = 0;
            updateRecordTimer();
            recordInterval = setInterval(() => {
                recordSeconds++;
                updateRecordTimer();
            }, 1000);
        } catch (err) {
            console.error('Mikrofonin käyttö estetty tai epäonnistui:', err);
            alert('Mikrofonin käyttö estetty tai ei saatavilla.');
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
            micRecordBtn.classList.remove('recording');
            clearInterval(recordInterval);
        }
    }

    function updateRecordTimer() {
        const mins = Math.floor(recordSeconds / 60).toString().padStart(2, '0');
        const secs = (recordSeconds % 60).toString().padStart(2, '0');
        recordTimer.textContent = `${mins}:${secs}`;
    }

    function updateSubmitButtonState() {
        if (currentMode === 'upload' && selectedFile) {
            transcribeBtn.disabled = false;
        } else if (currentMode === 'record' && recordedBlob) {
            transcribeBtn.disabled = false;
        } else {
            transcribeBtn.disabled = true;
        }
    }

    // Submit Transcription
    transcribeBtn.addEventListener('click', async () => {
        let fileToUpload = null;
        let filename = 'audio.wav';

        if (currentMode === 'upload' && selectedFile) {
            fileToUpload = selectedFile;
            filename = selectedFile.name;
        } else if (currentMode === 'record' && recordedBlob) {
            fileToUpload = recordedBlob;
            filename = `nauhoitus_${Date.now()}.webm`;
        }

        if (!fileToUpload) return;

        // UI state -> Loading
        emptyState.classList.add('hidden');
        resultContainer.classList.add('hidden');
        loadingState.classList.remove('hidden');
        transcribeBtn.disabled = true;

        let startTime = Date.now();
        elapsedTimer.textContent = '0.0 s';
        elapsedInterval = setInterval(() => {
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
            elapsedTimer.textContent = `${elapsed} s`;
        }, 100);

        const formData = new FormData();
        formData.append('file', fileToUpload, filename);
        formData.append('beam_size', beamSizeInput.value || '5');
        formData.append('vad_filter', vadFilterInput.checked ? 'true' : 'false');
        formData.append('word_timestamps', wordTimestampsInput.checked ? 'true' : 'false');

        try {
            const response = await fetch('/api/v1/transcribe', {
                method: 'POST',
                body: formData
            });

            clearInterval(elapsedInterval);

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Virhe puheentunnistuksessa');
            }

            const data = await response.json();
            lastResult = data;
            displayResults(data);

        } catch (error) {
            clearInterval(elapsedInterval);
            alert(`Tunnistus epäonnistui: ${error.message}`);
            emptyState.classList.remove('hidden');
            loadingState.classList.add('hidden');
        } finally {
            transcribeBtn.disabled = false;
        }
    });

    function displayResults(data) {
        loadingState.classList.add('hidden');
        resultContainer.classList.remove('hidden');

        statDuration.textContent = data.duration ? `${data.duration}s` : 'N/A';
        statProcTime.textContent = `${data.processing_time}s`;
        fullTranscriptText.textContent = data.text || '(Ei tunnistettua puhetta)';

        segmentCount.textContent = data.segments ? data.segments.length : 0;
        segmentsList.innerHTML = '';

        if (data.segments && data.segments.length > 0) {
            data.segments.forEach(seg => {
                const div = document.createElement('div');
                div.className = 'segment-item';
                div.innerHTML = `
                    <span class="segment-time">[${formatTime(seg.start)} &rarr; ${formatTime(seg.end)}]</span>
                    <span class="segment-text">${escapeHtml(seg.text)}</span>
                `;
                segmentsList.appendChild(div);
            });
        } else {
            segmentsList.innerHTML = '<p class="sub-text">Ei segmenttejä.</p>';
        }
    }

    // Utility Functions
    function formatBytes(bytes, decimals = 1) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }

    function formatTime(seconds) {
        const m = Math.floor(seconds / 60).toString().padStart(2, '0');
        const s = (seconds % 60).toFixed(2).padStart(5, '0');
        return `${m}:${s}`;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Export & Copy Actions
    copyTextBtn.addEventListener('click', () => {
        if (lastResult && lastResult.text) {
            navigator.clipboard.writeText(lastResult.text);
            const originalText = copyTextBtn.innerHTML;
            copyTextBtn.innerHTML = 'Kopioitu!';
            setTimeout(() => copyTextBtn.innerHTML = originalText, 2000);
        }
    });

    downloadTxtBtn.addEventListener('click', () => {
        if (lastResult && lastResult.text) {
            downloadFile(`${lastResult.filename || 'tunnistus'}.txt`, lastResult.text, 'text/plain');
        }
    });

    downloadJsonBtn.addEventListener('click', () => {
        if (lastResult) {
            downloadFile(`${lastResult.filename || 'tunnistus'}.json`, JSON.stringify(lastResult, null, 2), 'application/json');
        }
    });

    function downloadFile(filename, content, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
});
