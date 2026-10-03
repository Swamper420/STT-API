import logging
import os
import subprocess
import threading
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.config import settings

logger = logging.getLogger("stt_api.model")

TARGET_SAMPLE_RATE = 16000
SUPPORTED_SAMPLE_RATES = (8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000)


def _decode_with_ffmpeg(path: str) -> Tuple[np.ndarray, int]:
    """Decodes any ffmpeg-readable audio to mono float32 16kHz."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", path,
        "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE),
        "-f", "f32le", "-acodec", "pcm_f32le", "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return np.frombuffer(proc.stdout, dtype=np.float32).copy(), TARGET_SAMPLE_RATE


def _decode_with_soundfile(path: str) -> Tuple[np.ndarray, int]:
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = np.mean(data, axis=-1, dtype=np.float32)
    return np.asarray(data, dtype=np.float32), int(sr)


def _decode_with_wave(path: str) -> Tuple[np.ndarray, int]:
    """Stdlib fallback for PCM WAV files."""
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if sampwidth == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth}")
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32), sr


def load_waveform_mono(path: Any) -> Tuple[np.ndarray, int]:
    """Returns (mono float32 waveform, sample_rate). Caller resamples to 16k."""
    if not isinstance(path, (str, Path)):
        return np.asarray(path, dtype=np.float32), TARGET_SAMPLE_RATE
    str_path = str(path)
    errors = []
    try:
        return _decode_with_ffmpeg(str_path)
    except FileNotFoundError:
        errors.append("ffmpeg binary not found")
    except Exception as e:
        errors.append(f"ffmpeg: {e}")
    try:
        waveform, sr = _decode_with_soundfile(str_path)
    except Exception as e:
        errors.append(f"soundfile: {e}")
        try:
            waveform, sr = _decode_with_wave(str_path)
        except Exception as e2:
            errors.append(f"wave: {e2}")
            raise RuntimeError(f"Could not decode audio '{str_path}': {'; '.join(errors)}")
    if sr not in SUPPORTED_SAMPLE_RATES:
        # ponytail: linear resample, exact resamplers if quality matters
        src_len = len(waveform)
        dst_len = max(1, int(round(src_len * TARGET_SAMPLE_RATE / float(sr))))
        waveform = np.interp(
            np.linspace(0.0, float(src_len), dst_len, endpoint=False),
            np.arange(src_len, dtype=np.float64),
            waveform.astype(np.float64),
        ).astype(np.float32)
        sr = TARGET_SAMPLE_RATE
    return np.ascontiguousarray(waveform, dtype=np.float32), sr


def _to_16k(waveform: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SAMPLE_RATE:
        return np.ascontiguousarray(waveform, dtype=np.float32)
    # ponytail: linear resample, exact resamplers if quality matters
    src_len = len(waveform)
    dst_len = max(1, int(round(src_len * TARGET_SAMPLE_RATE / float(sr))))
    out = np.interp(
        np.linspace(0.0, float(src_len), dst_len, endpoint=False),
        np.arange(src_len, dtype=np.float64),
        np.asarray(waveform, dtype=np.float64),
    ).astype(np.float32)
    return np.ascontiguousarray(out, dtype=np.float32)


def _backend() -> str:
    d = (settings.device or "auto").strip().lower()
    if d in ("cuda", "gpu"):
        return "cuda"
    return d


class STTModelWrapper:
    """Wrapper around Parakeet Ultra 0.6B GGUF via transcribe.cpp (ggml)."""

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()

    def _resolve_model_path(self) -> str:
        ref = (settings.gguf_file or "").strip()
        if ref and os.path.exists(ref):
            return ref
        from huggingface_hub import hf_hub_download

        filename = os.path.basename(ref) if ref.endswith(".gguf") else ref
        logger.info(f"Downloading '{filename}' from '{settings.model_id}'...")
        return hf_hub_download(repo_id=settings.model_id, filename=filename)

    def load_model(self) -> None:
        """Loads the GGUF model (downloads from Hugging Face on first run)."""
        import transcribe_cpp

        with self._lock:
            if self._model is not None:
                return
            path = self._resolve_model_path()
            backend = _backend()
            logger.info(
                f"Loading STT model '{settings.model_id}/{os.path.basename(path)}' "
                f"backend='{backend}'..."
            )
            self._model = transcribe_cpp.Model(path, backend=backend)
            logger.info("STT model loaded successfully!")

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def transcribe(
        self,
        audio_path_or_file: Any,
        beam_size: Optional[int] = None,  # ignored: greedy TDT, kept for API compat
        vad_filter: Optional[bool] = None,  # ignored: no VAD in this runtime
        word_timestamps: bool = False,
        initial_prompt: Optional[str] = None,  # ignored: Parakeet takes no prompt
    ) -> Dict[str, Any]:
        """Transcribes audio. Language auto-detected by Parakeet Ultra."""
        if self._model is None:
            self.load_model()

        start_time = time.time()

        waveform, sample_rate = load_waveform_mono(audio_path_or_file)
        pcm = _to_16k(waveform, sample_rate)
        duration = round(float(len(pcm)) / float(TARGET_SAMPLE_RATE), 3)

        if len(pcm) == 0:
            return {
                "text": "",
                "language": settings.language,
                "duration": 0.0,
                "duration_after_vad": 0.0,
                "processing_time": round(time.time() - start_time, 3),
                "segments": [],
            }

        language = None if settings.language.strip().lower() == "auto" else settings.language
        timestamps = "word" if word_timestamps else "segment"

        # One run at a time per Model: sessions share the compute backend.
        with self._lock:
            with self._model.session() as session:
                result = session.run(pcm, language=language, timestamps=timestamps)

        text = (result.text or "").strip()
        lang = result.language or settings.language
        segments: List[Dict[str, Any]] = []
        for i, seg in enumerate(result.segments or ()):
            segments.append({
                "id": i,
                "start": round(seg.t0_ms / 1000.0, 3),
                "end": round(seg.t1_ms / 1000.0, 3),
                "text": (seg.text or "").strip(),
            })
        if word_timestamps and getattr(result, "words", None):
            by_seg: Dict[int, List[Dict[str, Any]]] = {}
            for w in result.words:
                by_seg.setdefault(w.seg_index, []).append({
                    "word": w.text,
                    "start": round(w.t0_ms / 1000.0, 3),
                    "end": round(w.t1_ms / 1000.0, 3),
                })
            for i, item in enumerate(segments):
                if i in by_seg:
                    item["words"] = by_seg[i]
        if not segments and text:
            segments = [{"id": 0, "start": 0.0, "end": duration, "text": text}]

        return {
            "text": text,
            "language": lang,
            "duration": duration,
            "duration_after_vad": duration,
            "processing_time": round(time.time() - start_time, 3),
            "segments": segments,
        }


stt_engine = STTModelWrapper()
