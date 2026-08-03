import os
from pydantic import BaseModel

class Settings(BaseModel):
    # Model configuration
    model_id: str = os.getenv("STT_MODEL_ID", "RASMUS/whisper-large-v3-turbo-finnish-ct2")
    device: str = os.getenv("STT_DEVICE", "cuda")
    compute_type: str = os.getenv("STT_COMPUTE_TYPE", "float16")
    
    # Transcription configuration - STRICTLY locked to Finnish
    language: str = "fi"
    
    # Default inference parameters
    beam_size: int = int(os.getenv("STT_BEAM_SIZE", "5"))
    vad_filter: bool = os.getenv("STT_VAD_FILTER", "true").lower() in ("true", "1", "yes")
    
    # Server network settings
    host: str = os.getenv("STT_HOST", "0.0.0.0")
    port: int = int(os.getenv("STT_PORT", "8001"))
    use_ssl: bool = os.getenv("STT_USE_SSL", "false").lower() in ("true", "1", "yes")
    ssl_certfile: str = os.getenv("STT_SSL_CERT", "cert.pem")
    ssl_keyfile: str = os.getenv("STT_SSL_KEY", "key.pem")

settings = Settings()
