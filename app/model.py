import logging
import math
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
# Must mirror onnx-asr SampleRates; anything else is resampled below.
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
    """Returns (mono float32 waveform, sample_rate). Resampling handled by onnx-asr."""
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


def _group_tokens_to_words(
    tokens: List[str],
    timestamps: Optional[List[float]],
    logprobs: Optional[List[float]],
    seg_start: float,
    seg_end: float,
) -> List[Dict[str, Any]]:
    """Groups SentencePiece tokens (leading space = word start) into word dicts."""
    words: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    n = len(tokens)
    if not timestamps or len(timestamps) != n:
        timestamps = None
    if not logprobs or len(logprobs) != n:
        logprobs = None
    seg_dur = max(0.0, seg_end - seg_start)
    for i, tok in enumerate(tokens):
        start = round(seg_start + (timestamps[i] if timestamps else 0.0), 3)
        rel_end = timestamps[i + 1] if timestamps and i + 1 < n else seg_dur
        end = round(seg_start + rel_end, 3)
        prob = round(float(math.exp(logprobs[i])), 4) if logprobs else None
        if tok.startswith(" ") or current is None:
            if current is not None:
                current["end"] = start
                words.append(current)
            current = {"word": tok.strip(), "start": start, "end": end}
            if prob is not None:
                current["probability"] = prob
        else:
            current["word"] += tok.strip()
            current["end"] = end
    if current is not None:
        words.append(current)
    return [w for w in words if w["word"]]


class STTModelWrapper:
    """Wrapper around onnx-asr Parakeet TDT 0.6B v3 (INT8 quantized ONNX)."""

    def __init__(self):
        self._model = None
        self._vad = None
        self._lock = threading.Lock()

    def _providers(self) -> List[str]:
        if settings.device.lower() in ("cuda", "gpu"):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def load_model(self) -> None:
        """Loads the quantized ONNX model (downloads from Hugging Face on first run)."""
        import onnx_asr

        with self._lock:
            if self._model is not None:
                return
            logger.info(
                f"Loading STT model '{settings.model_id}' "
                f"quantization='{settings.quantization}' providers={self._providers()}..."
            )
            try:
                self._model = onnx_asr.load_model(
                    settings.model_id,
                    quantization=settings.quantization,
                    providers=self._providers(),
                )
            except Exception as e:
                if self._providers() != ["CPUExecutionProvider"]:
                    logger.warning(f"GPU load failed ({e}), retrying on CPU...")
                    self._model = onnx_asr.load_model(
                        settings.model_id,
                        quantization=settings.quantization,
                        providers=["CPUExecutionProvider"],
                    )
                else:
                    raise
            logger.info("STT model loaded successfully!")

    def _get_vad(self):
        """Lazy-loads Silero VAD for long-form (>30s) chunking."""
        import onnx_asr

        with self._lock:
            if self._vad is None:
                logger.info("Loading Silero VAD model for long-form transcription...")
                self._vad = onnx_asr.load_vad("silero", providers=["CPUExecutionProvider"])
            return self._vad

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def transcribe(
        self,
        audio_path_or_file: Any,
        beam_size: Optional[int] = None,  # ignored: greedy TDT, kept for API compat
        vad_filter: Optional[bool] = None,
        word_timestamps: bool = False,
        initial_prompt: Optional[str] = None,  # ignored: Parakeet takes no prompt
    ) -> Dict[str, Any]:
        """Transcribes audio. Language auto-detected by Parakeet v3."""
        if self._model is None:
            self.load_model()

        eff_vad_filter = vad_filter if vad_filter is not None else settings.vad_filter
        start_time = time.time()

        waveform, sample_rate = load_waveform_mono(audio_path_or_file)
        duration = round(float(len(waveform)) / float(sample_rate), 3)

        if len(waveform) == 0:
            return {
                "text": "",
                "language": settings.language,
                "duration": 0.0,
                "duration_after_vad": 0.0,
                "processing_time": round(time.time() - start_time, 3),
                "segments": [],
            }

        adapter = self._model
        if eff_vad_filter:
            # with_vad first: with_timestamps() on the VAD adapter keeps both.
            adapter = adapter.with_vad(self._get_vad())
        if word_timestamps:
            adapter = adapter.with_timestamps()

        if eff_vad_filter:
            seg_results = list(adapter.recognize(waveform, sample_rate=sample_rate))
            segments: List[Dict[str, Any]] = []
            texts: List[str] = []
            spoken = 0.0
            for i, seg in enumerate(seg_results):
                s = round(float(seg.start), 3)
                e = round(float(seg.end), 3)
                text = seg.text.strip()
                item: Dict[str, Any] = {"id": i, "start": s, "end": e, "text": text}
                if word_timestamps and getattr(seg, "tokens", None):
                    item["words"] = _group_tokens_to_words(
                        seg.tokens, seg.timestamps, getattr(seg, "logprobs", None), s, e
                    )
                segments.append(item)
                if text:
                    texts.append(text)
                spoken += max(0.0, e - s)
            full_text = " ".join(texts)
            duration_after_vad = round(spoken, 3)
        else:
            result = adapter.recognize(waveform, sample_rate=sample_rate)
            if isinstance(result, str):
                full_text = result.strip()
                segments = [{"id": 0, "start": 0.0, "end": duration, "text": full_text}]
            else:  # TimestampedResult
                full_text = result.text.strip()
                item = {"id": 0, "start": 0.0, "end": duration, "text": full_text}
                if getattr(result, "tokens", None):
                    item["words"] = _group_tokens_to_words(
                        result.tokens, result.timestamps, result.logprobs, 0.0, duration
                    )
                segments = [item]
            duration_after_vad = duration

        return {
            "text": full_text,
            "language": settings.language,
            "duration": duration,
            "duration_after_vad": duration_after_vad,
            "processing_time": round(time.time() - start_time, 3),
            "segments": segments,
        }


stt_engine = STTModelWrapper()
