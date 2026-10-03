#!/usr/bin/env python3
"""
Entry point to run the STT API server.
Usage:
    python3 run.py [--ssl]
"""

import sys
import os
import subprocess
import uvicorn
from app.config import settings

def ensure_self_signed_cert(cert_path: str, key_path: str):
    """Generates self-signed SSL cert/key using openssl if they do not exist."""
    if not os.path.exists(cert_path) or not os.path.exists(key_path):
        print("Generating self-signed SSL certificate for local network HTTPS...")
        cmd = [
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", key_path, "-out", cert_path,
            "-sha256", "-days", "365", "-nodes",
            "-subj", "/CN=stt-api-local"
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"Created SSL certificate: {cert_path}, key: {key_path}")
        except Exception as e:
            print(f"Warning: Could not auto-generate SSL cert: {e}")

if __name__ == "__main__":
    use_ssl = settings.use_ssl or "--ssl" in sys.argv

    if use_ssl:
        ensure_self_signed_cert(settings.ssl_certfile, settings.ssl_keyfile)
        proto = "https"
    else:
        proto = "http"

    print(f"Starting STT REST API server on {proto}://{settings.host}:{settings.port}")
    print(f"Model: {settings.model_id} (quantization={settings.quantization})")
    print(f"Language: {settings.language} (auto-detected, 25 European languages)")
    print(f"Device: {settings.device}")
    if use_ssl:
        print(f"SSL Enabled: {settings.ssl_certfile}, {settings.ssl_keyfile}")

    kwargs = {
        "app": "app.main:app",
        "host": settings.host,
        "port": settings.port,
        "reload": False
    }

    if use_ssl and os.path.exists(settings.ssl_certfile) and os.path.exists(settings.ssl_keyfile):
        kwargs["ssl_certfile"] = settings.ssl_certfile
        kwargs["ssl_keyfile"] = settings.ssl_keyfile

    uvicorn.run(**kwargs)
