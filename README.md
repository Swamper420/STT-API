# STT REST API Server

A high-performance REST Speech-to-Text (STT) API server serving Parakeet Ultra 0.6B GGUF via `transcribe-cpp` (Q4_K_M 4-bit by default, selectable).

The model auto-detects the spoken language (25 European languages, incl. Finnish) — no language selection needed.

---

## Features

- **Model**: Parakeet Ultra 0.6B GGUF ([Nairod785/parakeet-ultra-gguf](https://huggingface.co/Nairod785/parakeet-ultra-gguf), Q4_K_M 4-bit by default).
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
Needs a C++ toolchain only if building `transcribe.cpp` from source; wheels bundle CPU+Vulkan. For CUDA: `pip install "transcribe-cpp[cu12]"`.

### 2. Start the API Server
```bash
python3 run.py
```
Or using Uvicorn directly:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

The server will automatically download `parakeet-ultra-0.6b-Q4_K_M.gguf` (4-bit, 463 MB) from Hugging Face and preload it on startup. Set `STT_GGUF_FILE` to any other file below to switch versions.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `STT_MODEL_ID` | `Nairod785/parakeet-ultra-gguf` | Hugging Face repo holding the GGUF files |
| `STT_GGUF_FILE` | `parakeet-ultra-0.6b-Q4_K_M.gguf` | GGUF file: full filename (`parakeet-ultra-0.6b-Q8_0.gguf`), short quant alias (`F16`, `Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`), or local `.gguf` path |
| `STT_DEVICE` | `auto` | transcribe.cpp backend (`auto`, `cpu`, `cuda`, `vulkan`, `metal`, `rocm`); `gpu` is an alias for `cuda` |
| `STT_LANGUAGE` | `auto` | Response language tag (model auto-detects speech language) |
| `STT_HOST` | `0.0.0.0` | Server bind host |
| `STT_PORT` | `8001` | Server bind port |
| `STT_BEAM_SIZE` | `5` | Accepted for compatibility; ignored (greedy TDT decoding) |
| `STT_VAD_FILTER` | `true` | Accepted for compatibility; ignored (no VAD in GGUF runtime) |

---

## REST API Specification

### Available GGUF versions (`STT_GGUF_FILE`)

| `STT_GGUF_FILE` | Size | Notes |
|---|---|---|
| `parakeet-ultra-0.6b-F16.gguf` (or `F16`) | 1.20 GB | Exact copy of parent F16 weights |
| `parakeet-ultra-0.6b-Q8_0.gguf` (or `Q8_0`) | 705 MB | Upstream recommended default, ~F16 accuracy |
| `parakeet-ultra-0.6b-Q6_K.gguf` (or `Q6_K`) | 582 MB |  |
| `parakeet-ultra-0.6b-Q5_K_M.gguf` (or `Q5_K_M`) | 523 MB |  |
| `parakeet-ultra-0.6b-Q4_K_M.gguf` (or `Q4_K_M`) | 463 MB | Default here; smallest, for memory-constrained devices |

```bash
STT_GGUF_FILE=Q8_0 python3 run.py
```

---

### 1. Transcribe Audio
`POST /api/v1/transcribe`

**Content-Type**: `multipart/form-data`

#### Request Parameters
- `file` (*file*, required): Audio file (WAV, MP3, M4A, OGG, FLAC, WEBM, etc.).
- `beam_size` (*integer*, optional): Accepted for compatibility; ignored.
- `vad_filter` (*boolean*, optional, default: true): Accepted for compatibility; ignored (no VAD in GGUF runtime).
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
  "model": "Nairod785/parakeet-ultra-gguf",
  "gguf_file": "parakeet-ultra-0.6b-Q4_K_M.gguf",
  "quantization": "Q4_K_M",
  "language": "auto",
  "device": "auto"
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
