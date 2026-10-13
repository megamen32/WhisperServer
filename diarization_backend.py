"""Local Community-1 diarization and timestamp-based speaker attribution."""

from __future__ import annotations

import gc
import traceback
from contextlib import contextmanager
import os
import subprocess
import tempfile
import shutil
from pathlib import Path

MODEL_ID = "pyannote/speaker-diarization-community-1"
MAX_PCM_BYTES = 2 * 1024 ** 3
SAMPLE_RATE = 16000
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


class Community1Model:
    """A pipeline owned by the same broker lifecycle as ASR models."""

    def __init__(self, pipeline, device):
        self.pipeline = pipeline
        self.device = device

    @classmethod
    def from_pretrained(cls, device="cpu"):
        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise RuntimeError('Разделение говорящих требует установки whisper-server[diarization]') from exc
        import torch
        pipeline = _load_pipeline(Pipeline)
        pipeline.segmentation_batch_size = 1
        pipeline.embedding_batch_size = 1
        pipeline.to(torch.device(device))
        return cls(pipeline, device)

    def unload(self):
        """Free CUDA tensors, including WeSpeaker's class-level frame caches."""
        import torch
        pipeline = self.pipeline
        if pipeline is None:
            return
        pipeline.to(torch.device("cpu"))
        embedding = getattr(getattr(pipeline, "_embedding", None), "model_", None)
        models = list(getattr(pipeline, "_models", {}).values())
        if embedding is not None:
            models.extend([embedding, getattr(embedding, "resnet", None)])
        for model in models:
            cached = getattr(model, "num_frames", None)
            if hasattr(cached, "cache_clear"):
                cached.cache_clear()
        self.pipeline = None
        self.device = "unloaded"

    def diarize(self, samples, hook=None, num_speakers=None):
        import torch
        output = self.pipeline({"waveform": torch.from_numpy(samples).unsqueeze(0), "sample_rate": SAMPLE_RATE}, hook=hook, num_speakers=num_speakers)
        labels, turns = {}, []
        for turn, label in output.exclusive_speaker_diarization:
            speaker = labels.setdefault(label, f"SPEAKER_{len(labels):02d}")
            turns.append({"start": float(turn.start), "end": float(turn.end), "speaker": speaker})
        return turns


@contextmanager
def _cpu_model():
    model = Community1Model.from_pretrained("cpu")
    try:
        yield model
    finally:
        model.unload()
        model = None


def _compact_speech(samples, path):
    """Bound VAD working memory; preserve one global speaker clustering pass."""
    import numpy as np
    from faster_whisper.vad import get_speech_timestamps, VadOptions
    regions = []
    block = 300 * SAMPLE_RATE
    for offset in range(0, len(samples), block):
        low, high = max(0, offset - SAMPLE_RATE), min(len(samples), offset + block + SAMPLE_RATE)
        for region in get_speech_timestamps(samples[low:high], VadOptions(speech_pad_ms=500)):
            start, end = low + region["start"], low + region["end"]
            if regions and start <= regions[-1][1]:
                regions[-1] = (regions[-1][0], max(end, regions[-1][1]))
            else:
                regions.append((start, end))
    mapping, cursor = [], 0
    with open(path, "wb") as target:
        for start, end in regions:
            samples[start:end].tofile(target)
            mapping.append((cursor / SAMPLE_RATE, (cursor + end - start) / SAMPLE_RATE, start / SAMPLE_RATE))
            cursor += end - start
    return (np.memmap(path, dtype="<f4", mode="c") if cursor else None), mapping


def _restore_turns(turns, mapping):
    restored = []
    for turn in turns:
        for low, high, original in mapping:
            start, end = max(turn["start"], low), min(turn["end"], high)
            if end > start:
                restored.append({**turn, "start": original + start - low, "end": original + end - low})
    return restored


def diarize_audio(audio_path: str, vad_filter: bool = True, model_provider=None) -> list[dict]:
    """Decode/VAD first, then reserve GPU only for the global model pass."""
    import numpy as np
    import torch
    model = None
    previous_threads = torch.get_num_threads()
    try:
        with tempfile.TemporaryDirectory(prefix="whisper-diarization-") as directory:
            if shutil.disk_usage(directory).free < 20 * 1024 ** 3 + 2 * MAX_PCM_BYTES:
                raise RuntimeError("Недостаточно временного дискового пространства для диаризации")
            pcm = str(Path(directory) / "audio.pcm")
            subprocess.run(
                ["prlimit", f"--fsize={MAX_PCM_BYTES}", "--", "ffmpeg", "-nostdin", "-v", "error",
                 "-threads", "1", "-i", audio_path, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
                 "-f", "f32le", pcm], capture_output=True, timeout=600, check=True,
            )
            if not Path(pcm).stat().st_size:
                return []
            samples = np.memmap(pcm, dtype="<f4", mode="c")
            mapping = None
            if vad_filter and len(samples) > 300 * SAMPLE_RATE:
                samples, mapping = _compact_speech(samples, str(Path(directory) / "speech.pcm"))
                if samples is None:
                    return []
            torch.set_num_threads(min(previous_threads, 2))
            with (model_provider or _cpu_model)() as model:
                try:
                    turns = model.diarize(samples)
                except Exception as exc:
                    traceback.clear_frames(exc.__traceback__)
                    raise
                finally:
                    # Drop the live pipeline reference before the provider releases VRAM.
                    model = None
            return _restore_turns(turns, mapping) if mapping is not None else turns
    finally:
        model = None
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
