import os
import sys
import site
import ctypes
import time
import logging
import threading
from typing import Dict, Any, List, Optional
from app.config import settings

logger = logging.getLogger("stt_api.model")

def fix_cuda_libraries():
    """Dynamically preloads libcublas and libcudnn shared libraries from PyTorch / NVIDIA site-packages."""
    search_dirs = []
    # Add site packages
    try:
        search_dirs.extend(site.getsitepackages())
    except Exception:
        pass
    if hasattr(site, 'USER_SITE') and site.USER_SITE:
        search_dirs.append(site.USER_SITE)
    
    # Add virtual environment site-packages
    for path in sys.path:
        if "site-packages" in path and path not in search_dirs:
            search_dirs.append(path)

    lib_paths = []
    for sp in search_dirs:
        nvidia_dir = os.path.join(sp, "nvidia")
        if os.path.isdir(nvidia_dir):
            for root, dirs, _ in os.walk(nvidia_dir):
                if os.path.basename(root) == "lib":
                    lib_paths.append(root)

    # Preload shared libraries
    loaded_count = 0
    for lib_dir in lib_paths:
        if lib_dir not in os.environ.get("LD_LIBRARY_PATH", ""):
            os.environ["LD_LIBRARY_PATH"] = lib_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        
        for file in os.listdir(lib_dir):
            if file.startswith("libcublas") or file.startswith("libcudnn") or file.startswith("libcublasLt"):
                full_path = os.path.join(lib_dir, file)
                try:
                    ctypes.CDLL(full_path, mode=ctypes.RTLD_GLOBAL)
                    loaded_count += 1
                except Exception:
                    pass
    if loaded_count > 0:
        logger.info(f"Preloaded {loaded_count} CUDA shared libraries (cuBLAS / cuDNN) for CTranslate2.")

# Execute library preloading on module import
fix_cuda_libraries()

from faster_whisper import WhisperModel

class STTModelWrapper:
    """Wrapper around faster-whisper WhisperModel enforcing Finnish language transcription."""
    
    def __init__(self):
        self._model: Optional[WhisperModel] = None
        self._lock = threading.Lock()
        self._is_loading = False

    def load_model(self) -> None:
        """Loads the CTranslate2 model onto CUDA."""
        with self._lock:
            if self._model is not None:
                return
            
            self._is_loading = True
            logger.info(
                f"Loading STT model '{settings.model_id}' on device='{settings.device}' with compute_type='{settings.compute_type}'..."
            )
            try:
                self._model = WhisperModel(
                    settings.model_id,
                    device=settings.device,
                    compute_type=settings.compute_type
                )
                logger.info("STT model loaded successfully!")
            except Exception as e:
                logger.error(f"Failed to load STT model: {e}")
                # Fallback check if float16 failed on specific CUDA GPU types
                if settings.compute_type == "float16" and "float16" in str(e).lower():
                    logger.info("Attempting fallback to compute_type='int8_float16'...")
                    self._model = WhisperModel(
                        settings.model_id,
                        device=settings.device,
                        compute_type="int8_float16"
                    )
                    logger.info("STT model loaded with fallback compute_type 'int8_float16'.")
                else:
                    raise e
            finally:
                self._is_loading = False

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def transcribe(
        self,
        audio_path_or_file: Any,
        beam_size: Optional[int] = None,
        vad_filter: Optional[bool] = None,
        word_timestamps: bool = False,
        initial_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribes the given audio file.
        Strictly enforces language='fi' (Finnish).
        """
        if not self._model:
            self.load_model()

        eff_beam_size = beam_size if beam_size is not None else settings.beam_size
        eff_vad_filter = vad_filter if vad_filter is not None else settings.vad_filter

        start_time = time.time()
        
        # Enforce language="fi" regardless of user input
        segments_iter, info = self._model.transcribe(
            audio_path_or_file,
            language=settings.language,
            beam_size=eff_beam_size,
            vad_filter=eff_vad_filter,
            word_timestamps=word_timestamps,
            initial_prompt=initial_prompt
        )

        segments_list: List[Dict[str, Any]] = []
        full_text_parts: List[str] = []

        for seg in segments_iter:
            segment_dict = {
                "id": seg.id,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
                "avg_logprob": round(seg.avg_logprob, 4),
                "no_speech_prob": round(seg.no_speech_prob, 4)
            }
            if word_timestamps and getattr(seg, "words", None):
                segment_dict["words"] = [
                    {
                        "word": w.word,
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "probability": round(w.probability, 4)
                    }
                    for w in seg.words
                ]
            
            segments_list.append(segment_dict)
            if seg.text.strip():
                full_text_parts.append(seg.text.strip())

        processing_time = round(time.time() - start_time, 3)
        full_text = " ".join(full_text_parts)

        return {
            "text": full_text,
            "language": settings.language,
            "duration": round(info.duration, 3) if hasattr(info, "duration") else None,
            "duration_after_vad": round(info.duration_after_vad, 3) if hasattr(info, "duration_after_vad") else None,
            "processing_time": processing_time,
            "segments": segments_list
        }

stt_engine = STTModelWrapper()
