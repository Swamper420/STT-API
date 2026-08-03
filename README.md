# Finnish STT REST API Server

A high-performance REST Speech-to-Text (STT) API server serving [RASMUS/whisper-large-v3-turbo-finnish-ct2](https://huggingface.co/RASMUS/whisper-large-v3-turbo-finnish-ct2) via `faster-whisper` (CTranslate2) on CUDA.

The server is **locked to Finnish language** (`language="fi"`), eliminating language detection overhead and optimizing Finnish speech recognition accuracy.

---

## Features

- **Model**: `RASMUS/whisper-large-v3-turbo-finnish-ct2` (CTranslate2 FP16 on CUDA).
- **Language Lock**: Restricted to Finnish (`language="fi"`).
- **REST API**: Simple multipart audio transcription endpoint.
- **Web UI**: Modern responsive test web interface built-in (browser mic recorder, file drag & drop, waveform/audio player, segment breakdown, export TXT/JSON).
- **Default Port**: `8001` (configurable via `STT_PORT`).

---

## Installation & Running

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the API Server
```bash
python3 run.py
```
Or using Uvicorn directly:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

The server will automatically preload `RASMUS/whisper-large-v3-turbo-finnish-ct2` on GPU CUDA on startup.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `STT_MODEL_ID` | `RASMUS/whisper-large-v3-turbo-finnish-ct2` | Hugging Face model repository ID |
| `STT_DEVICE` | `cuda` | PyTorch / CTranslate2 device (`cuda` or `cpu`) |
| `STT_COMPUTE_TYPE` | `float16` | Precision mode (`float16`, `int8_float16`, etc.) |
| `STT_HOST` | `0.0.0.0` | Server bind host |
| `STT_PORT` | `8001` | Server bind port |
| `STT_BEAM_SIZE` | `5` | Decoding beam search size |
| `STT_VAD_FILTER` | `true` | Enable Voice Activity Detection filtering |

---

## REST API Specification

### 1. Transcribe Audio
`POST /api/v1/transcribe`

**Content-Type**: `multipart/form-data`

#### Request Parameters
- `file` (*file*, required): Audio file (WAV, MP3, M4A, OGG, FLAC, WEBM, etc.).
- `beam_size` (*integer*, optional, default: 5): Beam size for search decoding.
- `vad_filter` (*boolean*, optional, default: true): Enable VAD silence filtering.
- `word_timestamps` (*boolean*, optional, default: false): Include word-level timestamps.
- `initial_prompt` (*string*, optional): Context or style prompt.

#### cURL Example
```bash
curl -X POST "http://localhost:8001/api/v1/transcribe" \
  -F "file=@/path/to/puhe.wav" \
  -F "vad_filter=true"
```

#### Example Response JSON
```json
{
  "status": "success",
  "filename": "puhe.wav",
  "text": "Tämä on esimerkki puheentunnistuksesta suomeksi.",
  "language": "fi",
  "duration": 3.45,
  "duration_after_vad": 3.12,
  "processing_time": 0.14,
  "segments": [
    {
      "id": 1,
      "start": 0.0,
      "end": 3.45,
      "text": "Tämä on esimerkki puheentunnistuksesta suomeksi.",
      "avg_logprob": -0.15,
      "no_speech_prob": 0.01
    }
  ]
}
```

---

### 2. Health Check
`GET /health`

#### Example Response
```json
{
  "status": "healthy",
  "model": "RASMUS/whisper-large-v3-turbo-finnish-ct2",
  "language": "fi",
  "device": "cuda",
  "compute_type": "float16"
}
```

---

## Web Frontend

Open your browser and navigate to:
```
http://localhost:8001/
```
Features include:
- Audio file drag & drop or device browsing.
- Live microphone recording right in the browser.
- Real-time processing elapsed timer.
- Detailed segment list with timestamps.
- Copy transcript to clipboard or download as TXT / JSON.
