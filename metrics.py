"""JSONL transcription telemetry and public model metadata helpers."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

import torch


logger = logging.getLogger(__name__)
_metrics_path = ""


def configure_metrics(path: str) -> None:
    """Configure the process-wide JSONL metrics destination."""
    global _metrics_path
    _metrics_path = path


def audio_duration_seconds(audio_path: str) -> float | None:
    """Read duration with ffprobe when the ASR backend does not expose it."""
    if shutil.which("ffprobe") is None:
        return None
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            return None
        duration = float(result.stdout.strip())
        return duration if duration >= 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def reset_cuda_peak_memory() -> None:
    """Reset per-job CUDA peak accounting when CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_vram_mb() -> int:
    """Return peak reserved CUDA memory for the current worker process."""
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_reserved() / (1024 * 1024))


def append_metric(record: dict) -> None:
    """Append one JSONL job metric without making transcription fail."""
    if not _metrics_path:
        return
    try:
        metrics_path = Path(_metrics_path).expanduser()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        logger.exception("Could not append transcription metrics to %s", _metrics_path)


def model_metadata(result: dict) -> dict:
    """Return model-resolution fields suitable for public API responses."""
    fields = (
        "requested_model",
        "served_model",
        "model_substituted",
        "substitution_reason",
    )
    return {field: result[field] for field in fields if field in result}


def transcription_response_payload(result: dict) -> dict:
    """Build an OpenAI-compatible response with truthful model metadata."""
    payload = {"text": result.get("text", "")}
    payload.update(model_metadata(result))
    return payload


def append_cache_metric(result: dict, requested_model: str) -> None:
    """Record a successful cache hit as a zero-inference job."""
    append_metric({
        "requested_model": requested_model,
        "served_model": result.get("served_model", requested_model),
        "model_substituted": result.get("model_substituted", False),
        "substitution_reason": result.get("substitution_reason"),
        "device": "cache",
        "cold_load": False,
        "load_ms": 0.0,
        "queue_ms": 0.0,
        "audio_seconds": None,
        "inference_ms": 0.0,
        "rtf": None,
        "peak_vram_mb": 0,
        "success": True,
        "cache_hit": True,
    })
