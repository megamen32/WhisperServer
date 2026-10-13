"""Local Community-1 diarization and timestamp-based speaker attribution."""

from __future__ import annotations

import gc
import os
import subprocess

MODEL_ID = "pyannote/speaker-diarization-community-1"
MAX_AUDIO_SECONDS = 1200
ACCESS_ERROR = 'Community-1 недоступна: примите условия модели в Hugging Face и настройте HF_TOKEN на сервере'


def _load_pipeline(pipeline_type):
    """Translate gated/invalid credentials into an actionable server-side error."""
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    try:
        pipeline = pipeline_type.from_pretrained(
            MODEL_ID, token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
        )
    except HfHubHTTPError as exc:
        if isinstance(exc, GatedRepoError) or getattr(exc.response, 'status_code', None) in {401, 403}:
            raise RuntimeError(ACCESS_ERROR) from exc
        raise
    if pipeline is None:
        raise RuntimeError(ACCESS_ERROR)
    return pipeline


def diarize_audio(audio_path: str) -> list[dict]:
    """Run on CPU inside the existing sequential worker; release after each job.

    Decode through ffmpeg to avoid torchcodec and to bound decoded audio memory.
    The extra second detects oversized files rather than silently truncating them.
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError('Разделение говорящих требует установки whisper-server[diarization]') from exc
    import numpy as np
    import torch

    decoded = subprocess.run(
        ["ffmpeg", "-v", "error", "-threads", "1", "-i", audio_path, "-t", str(MAX_AUDIO_SECONDS + 1),
         "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"],
        capture_output=True, timeout=60, check=True,
    )
    samples = np.frombuffer(decoded.stdout, dtype="<f4").copy()
    if len(samples) > MAX_AUDIO_SECONDS * 16000:
        raise ValueError('Разделение говорящих поддерживает записи до 20 минут')
    if not len(samples):
        return []
    pipeline = None
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(min(previous_threads, 2))
        pipeline = _load_pipeline(Pipeline)
        pipeline.to(torch.device("cpu"))
        output = pipeline({"waveform": torch.from_numpy(samples).unsqueeze(0), "sample_rate": 16000})
        labels = {}
        turns = []
        for turn, label in output.exclusive_speaker_diarization:
            speaker = labels.setdefault(label, f"SPEAKER_{len(labels):02d}")
            turns.append({"start": float(turn.start), "end": float(turn.end), "speaker": speaker})
        return turns
    finally:
        pipeline = None
        gc.collect()
        torch.set_num_threads(previous_threads)


def speaker_for_interval(start: float, end: float, turns: list[dict]) -> str | None:
    """Choose the greatest overlap; retain unknown when timestamps do not match."""
    best, overlap = None, 0.0
    for turn in turns:
        shared = max(0.0, min(end, turn["end"]) - max(start, turn["start"]))
        if shared > overlap or (end == start and turn["start"] <= start < turn["end"]):
            best, overlap = turn["speaker"], shared
    return best


def attribute_segment(segment: dict, words: list[dict], turns: list[dict]) -> list[dict]:
    """Split ASR segments at word-level speaker changes, preserving native timing."""
    if not words:
        return [{**segment, "speaker": speaker_for_interval(segment["start"], segment["end"], turns)}]
    groups = []
    for word in words:
        word["speaker"] = speaker_for_interval(word["start"], word["end"], turns)
        if not groups or groups[-1]["speaker"] != word["speaker"]:
            groups.append({**segment, "start": word["start"], "end": word["end"],
                           "text": word["word"], "speaker": word["speaker"]})
        else:
            groups[-1]["end"] = word["end"]
            token = word["word"]
            # Parakeet tokens omit leading whitespace; Whisper usually keeps it.
            separator = " " if token[:1].isalnum() and not ('\u2e80' <= token[0] <= '\u9fff') else ""
            groups[-1]["text"] += separator + token
    if len(groups) == 1:
        groups[0]["text"] = segment["text"]
    for group in groups:
        group["text"] = group["text"].strip()
    return groups


def speaker_text(segments: list[dict]) -> str:
    """Readable transcript for UI copy and compact API clients."""
    lines = []
    previous = object()
    for segment in segments:
        speaker = segment.get("speaker")
        label = f"Спикер {int(speaker.rsplit('_', 1)[1]) + 1}" if speaker else "Неизвестный спикер"
        if lines and previous == speaker:
            lines[-1] += " " + segment["text"]
        else:
            lines.append(f"{label}: {segment['text']}")
        previous = speaker
    return "\n".join(lines)
