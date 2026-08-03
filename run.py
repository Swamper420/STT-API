#!/usr/bin/env python3
"""
Entry point to run the STT API server.
Usage:
    python3 run.py
"""

import uvicorn
from app.config import settings

if __name__ == "__main__":
    print(f"Starting STT REST API server on http://{settings.host}:{settings.port}")
    print(f"Model: {settings.model_id}")
    print(f"Language: {settings.language} (Finnish only)")
    print(f"Device: {settings.device}")

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False
    )
