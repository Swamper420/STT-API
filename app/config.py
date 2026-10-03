import os
from typing import Optional
from pydantic import BaseModel


def _device_default() -> str:
    return os.getenv("STT_DEVICE", "cuda")


def _quantization_default(device: str) -> Optional[str]:
    raw = os.getenv("STT_QUANTIZATION")
    if raw is None:
        # INT8 has no CUDA kernels: full precision on GPU, INT8 on CPU.
        return "int8" if device.strip().lower() == "cpu" else None
    raw = raw.strip().lower()
    if raw in ("", "none", "null", "fp32", "float32", "full"):
        return None
    return raw


class Settings(BaseModel):
    # Model configuration (Parakeet TDT 0.6B v3 via onnx-asr, GPU-exclusive)
    model_id: str = os.getenv("STT_MODEL_ID", "nemo-parakeet-tdt-0.6b-v3")
    device: str = _device_default()
    quantization: Optional[str] = _quantization_default(_device_default())

    # Parakeet v3 auto-detects language (25 European languages, incl. Finnish).
    language: str = os.getenv("STT_LANGUAGE", "auto")

    # Kept for API compatibility. Parakeet uses greedy TDT decoding:
    # beam_size / initial_prompt are accepted but ignored.
    beam_size: int = int(os.getenv("STT_BEAM_SIZE", "5"))
    vad_filter: bool = os.getenv("STT_VAD_FILTER", "true").lower() in ("true", "1", "yes")
    
    # Server network settings
    host: str = os.getenv("STT_HOST", "0.0.0.0")
    port: int = int(os.getenv("STT_PORT", "8001"))
    use_ssl: bool = os.getenv("STT_USE_SSL", "false").lower() in ("true", "1", "yes")
    ssl_certfile: str = os.getenv("STT_SSL_CERT", "cert.pem")
    ssl_keyfile: str = os.getenv("STT_SSL_KEY", "key.pem")

settings = Settings()
