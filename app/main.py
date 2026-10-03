import asyncio
import os
import tempfile
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from app.config import settings
from app.model import stt_engine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("stt_api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for warm-loading the model on server startup."""
    logger.info("Initializing STT REST API Server...")
    try:
        stt_engine.load_model()
    except Exception as e:
        logger.warning(f"Could not preload model on startup (will lazy-load on first request): {e}")
    yield
    logger.info("Shutting down STT REST API Server.")

app = FastAPI(
    title="Parakeet Ultra STT REST API",
    description=f"REST API server serving {settings.model_id}/{settings.gguf_file} (GGUF {settings.quantization}) with automatic language detection.",
    version="2.2.0",
    lifespan=lifespan
)

# Enable CORS for cross-origin testing / frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static directory for web testing interface
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", include_in_schema=False)
async def serve_index():
    """Serves the test web frontend."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"message": "Parakeet STT REST API Server is running."})

@app.get("/health", summary="Health check endpoint")
async def health_check():
    """Returns server and model loading status."""
    return {
        "status": "healthy" if stt_engine.is_ready else "initializing",
        "model": settings.model_id,
        "gguf_file": settings.gguf_file,
        "quantization": settings.quantization,
        "language": settings.language,
        "device": settings.device
    }

@app.post("/api/v1/transcribe", summary="Transcribe audio file to text")
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file (wav, mp3, ogg, m4a, flac, webm, etc.)"),
    beam_size: int = Form(default=settings.beam_size, description="Accepted for compatibility; ignored (greedy TDT decoding)"),
    vad_filter: bool = Form(default=settings.vad_filter, description="Accepted for compatibility; ignored (no VAD in GGUF runtime)"),
    word_timestamps: bool = Form(default=False, description="Include word-level timestamps"),
    initial_prompt: str = Form(default=None, description="Accepted for compatibility; ignored (Parakeet takes no prompt)")
):
    """
    Transcribes an audio file using Parakeet Ultra 0.6B GGUF.
    The spoken language is auto-detected (25 European languages, incl. Finnish).
    """
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file provided")

    suffix = os.path.splitext(file.filename)[1] or ".tmp"
    
    # Save uploaded bytes to temporary file for ffmpeg decoding
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty audio file provided")
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Execute transcription off the event loop (minutes-long on long audio)
        result = await asyncio.to_thread(
            stt_engine.transcribe,
            temp_file_path,
            beam_size,
            vad_filter,
            word_timestamps,
            initial_prompt,
        )

        return {
            "status": "success",
            "filename": file.filename,
            **result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during transcription of {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {str(e)}"
        )
    finally:
        # Cleanup temp file
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {temp_file_path}: {e}")
