"""
whisperclient package.
Exports transcription functions via lazy loading to avoid importing
heavy dependencies (torch, faster_whisper) at package import time.
"""

import os

# Light config variables — safe to load immediately
api_key = os.getenv("API_KEY", "bad-key")
model = os.getenv("MODEL", "large-v3")
server_url = os.getenv(
    "WHISPER_URL",
    "https://whisper.bezrabotnyi.com/transcribe"
)


def __getattr__(name: str):
    """
    Lazy loader for heavy functions.
    Called only when attribute is accessed, not at import.
    """
    if name in (
        "transcribe_sync",
        "transcribe_stream_sync",
        "transcribe_with_fallback",
        "transcribe_stream_with_fallback",
        "voice_to_text",
    ):
        from .transcriber import (
            transcribe_sync,
            transcribe_stream_sync,
            transcribe_with_fallback,
            transcribe_stream_with_fallback,
            voice_to_text,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Explicitly declare exports for type checkers / IDEs
__all__ = [
    "api_key",
    "model",
    "server_url",
    "transcribe_sync",
    "transcribe_stream_sync",
    "transcribe_with_fallback",
    "transcribe_stream_with_fallback",
    "voice_to_text",
]