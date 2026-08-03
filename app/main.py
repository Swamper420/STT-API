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
    title="Finnish STT REST API",
    description=f"REST API server serving {settings.model_id} strictly for Finnish language speech-to-text.",
    version="1.0.0",
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
    return JSONResponse({"message": "Finnish STT REST API Server is running."})

@app.get("/health", summary="Health check endpoint")
async def health_check():
    """Returns server and model loading status."""
    return {
        "status": "healthy" if stt_engine.is_ready else "initializing",
        "model": settings.model_id,
        "language": settings.language,
        "device": settings.device,
        "compute_type": settings.compute_type
    }

@app.post("/api/v1/transcribe", summary="Transcribe audio file into Finnish text")
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file (wav, mp3, ogg, m4a, flac, webm, etc.)"),
    beam_size: int = Form(default=settings.beam_size, description="Beam size for decoding search"),
    vad_filter: bool = Form(default=settings.vad_filter, description="Enable Voice Activity Detection filter"),
    word_timestamps: bool = Form(default=False, description="Include word-level timestamps"),
    initial_prompt: str = Form(default=None, description="Optional prompt to guide transcription style/context")
):
    """
    Transcribes an audio file into Finnish text using RASMUS/whisper-large-v3-turbo-finnish-ct2.
    The transcription language is hardcoded/restricted strictly to Finnish (`fi`).
    """
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file provided")

    suffix = os.path.splitext(file.filename)[1] or ".tmp"
    
    # Save uploaded bytes to temporary file for CTranslate2 / ffmpeg decoding
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty audio file provided")
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Execute transcription
        result = stt_engine.transcribe(
            audio_path_or_file=temp_file_path,
            beam_size=beam_size,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt
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
