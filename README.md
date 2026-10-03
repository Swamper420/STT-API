# STT REST API Server

A high-performance REST Speech-to-Text (STT) API server serving Parakeet TDT 0.6B v3 via `onnx-asr` (INT8 quantized ONNX, CUDA by default with CPU fallback).

The model auto-detects the spoken language (25 European languages, incl. Finnish) — no language selection needed.

---

## Features

- **Model**: `nemo-parakeet-tdt-0.6b-v3` ([istupakov/parakeet-tdt-0.6b-v3-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx), INT8 quantized ONNX).
- **Language**: auto-detected (25 European languages, incl. Finnish).
- **REST API**: Simple multipart audio transcription endpoint.
- **Web UI**: Modern responsive test web interface built-in (browser mic recorder, file drag & drop, waveform/audio player, segment breakdown, export TXT/JSON).
- **Default Port**: `8001` (configurable via `STT_PORT`).

---

## Installation & Running

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
Needs CUDA/cuDNN for GPU; otherwise set `STT_DEVICE=cpu`.

### 2. Start the API Server
```bash
python3 run.py
```
Or using Uvicorn directly:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

The server will automatically download `nemo-parakeet-tdt-0.6b-v3` (INT8 quantized ONNX) from Hugging Face and preload it on startup.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `STT_MODEL_ID` | `nemo-parakeet-tdt-0.6b-v3` | onnx-asr model name (or Hugging Face repo ID) |
| `STT_QUANTIZATION` | `int8` | Quantized weights (`int8`, empty/`none` = full precision) |
| `STT_DEVICE` | `cuda` | ONNX Runtime device (`cuda` default, `cpu` fallback/override) |
| `STT_LANGUAGE` | `auto` | Response language tag (model auto-detects speech language) |
| `STT_HOST` | `0.0.0.0` | Server bind host |
| `STT_PORT` | `8001` | Server bind port |
| `STT_BEAM_SIZE` | `5` | Accepted for compatibility; ignored (greedy TDT decoding) |
| `STT_VAD_FILTER` | `true` | Enable VAD chunking for long audio |

---

## REST API Specification

### 1. Transcribe Audio
`POST /api/v1/transcribe`

**Content-Type**: `multipart/form-data`

#### Request Parameters
- `file` (*file*, required): Audio file (WAV, MP3, M4A, OGG, FLAC, WEBM, etc.).
- `beam_size` (*integer*, optional): Accepted for compatibility; ignored.
- `vad_filter` (*boolean*, optional, default: true): Enable VAD chunking for long audio.
- `word_timestamps` (*boolean*, optional, default: false): Include word-level timestamps.
- `initial_prompt` (*string*, optional): Accepted for compatibility; ignored.

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
  "language": "auto",
  "duration": 3.45,
  "duration_after_vad": 3.12,
  "processing_time": 0.14,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.45,
      "text": "Tämä on esimerkki puheentunnistuksesta suomeksi."
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
  "model": "nemo-parakeet-tdt-0.6b-v3",
  "quantization": "int8",
  "language": "auto",
  "device": "cuda"
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
