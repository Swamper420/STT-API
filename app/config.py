import os
from typing import Optional
from pydantic import BaseModel

GGUF_REPO_DEFAULT = "Nairod785/parakeet-ultra-gguf"
GGUF_FILE_DEFAULT = "parakeet-ultra-0.6b-Q4_K_M.gguf"

# Every file in Nairod785/parakeet-ultra-gguf (alternatives, not parts).
GGUF_QUANTS = {
    "F16": "parakeet-ultra-0.6b-F16.gguf",
    "Q8_0": "parakeet-ultra-0.6b-Q8_0.gguf",
    "Q6_K": "parakeet-ultra-0.6b-Q6_K.gguf",
    "Q5_K_M": "parakeet-ultra-0.6b-Q5_K_M.gguf",
    "Q4_K_M": "parakeet-ultra-0.6b-Q4_K_M.gguf",
}


def _resolve_gguf_file(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return GGUF_FILE_DEFAULT
    if raw.endswith(".gguf") or "/" in raw or os.path.exists(raw):
        return raw
    hit = GGUF_QUANTS.get(raw.upper())
    return hit if hit is not None else raw


def _gguf_default() -> str:
    return _resolve_gguf_file(os.getenv("STT_GGUF_FILE", GGUF_FILE_DEFAULT))


class Settings(BaseModel):
    # Model configuration (Parakeet Ultra 0.6B GGUF via transcribe.cpp).
    # STT_GGUF_FILE accepts a full filename (parakeet-ultra-0.6b-Q8_0.gguf),
    # a short quant alias (Q8_0, Q4_K_M, F16, ...), or a local .gguf path.
    model_id: str = os.getenv("STT_MODEL_ID", GGUF_REPO_DEFAULT)
    gguf_file: str = _gguf_default()
    device: str = os.getenv("STT_DEVICE", "auto")

    # Parakeet Ultra auto-detects language (25 European languages, incl. Finnish).
    language: str = os.getenv("STT_LANGUAGE", "auto")

    # Kept for API compatibility. Greedy TDT decoding, no VAD in this runtime:
    # beam_size / vad_filter / initial_prompt are accepted but ignored.
    beam_size: int = int(os.getenv("STT_BEAM_SIZE", "5"))
    vad_filter: bool = os.getenv("STT_VAD_FILTER", "true").lower() in ("true", "1", "yes")

    # Long audio: single model pass up to this many seconds, then windows of
    # this size cut at quiet points. ponytail: one knob, rest fixed in model.py.
    chunk_seconds: float = float(os.getenv("STT_CHUNK_SECONDS", "30"))

    # Server network settings
    host: str = os.getenv("STT_HOST", "0.0.0.0")
    port: int = int(os.getenv("STT_PORT", "8001"))
    use_ssl: bool = os.getenv("STT_USE_SSL", "false").lower() in ("true", "1", "yes")
    ssl_certfile: str = os.getenv("STT_SSL_CERT", "cert.pem")
    ssl_keyfile: str = os.getenv("STT_SSL_KEY", "key.pem")

    @property
    def quantization(self) -> Optional[str]:
        base = os.path.basename(self.gguf_file)
        if base.endswith(".gguf"):
            stem = base[: -len(".gguf")]
            if "-" in stem:
                return stem.rsplit("-", 1)[-1].upper() or None
            return stem.upper() or None
        return self.gguf_file.upper() or None


settings = Settings()
