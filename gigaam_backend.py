"""GigaAM-v3 adapter with explicit safe audio-chunk boundaries."""
from __future__ import annotations

from typing import Any, Iterable
import tempfile
import os
import shutil
import wave
from pathlib import Path

import numpy as np
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps

from transcription_types import ParakeetInfo, ParakeetSegment

GIGAAM_V3_HF_ID = "ai-sage/GigaAM-v3"
# Immutable revision already installed in production; checkpoints must not refer
# to a branch that can change underneath otherwise identical requests.
GIGAAM_V3_REVISION = "7655ad717f8122257385bb4b2f373db3697e8680"
GIGAAM_ADAPTER_VERSION = "bounded-turns-v2"
MIN_DECODER_SAMPLES = 3200
SAMPLE_RATE = 16000
MAX_CHUNK_SAMPLES = 20 * SAMPLE_RATE
MAX_PCM_BYTES = 2 * 1024 ** 3
MAX_ACTIVE_PCM_BYTES = 8 * 1024 ** 3


def cleanup_stale_chunk_plans(root):
    """Only remove private plan directories whose worker no longer exists."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    for directory in root.glob("gigaam-plan-*"):
        if directory.is_symlink() or not directory.is_dir():
            continue
        try:
            owner_pid = int(directory.name.split("-")[2])
            os.kill(owner_pid, 0)
        except ProcessLookupError:
            shutil.rmtree(directory)
        except (ValueError, PermissionError, OSError):
            continue



class GigaAMChunkPlan:
    """Original boundaries with optional reusable source PCM and context plans."""
    def __init__(self, audio_path, *, vad_filter, vad_parameters=None, speaker_turns=None,
                 spool_dir=None, expected_seconds=None, max_active_bytes=MAX_ACTIVE_PCM_BYTES,
                 source_hash=None, cancel_check=None, preview_seconds=None, max_chunks=None):
        if expected_seconds and expected_seconds * SAMPLE_RATE * 4 > MAX_PCM_BYTES:
            raise ValueError("Audio exceeds the decoded-audio storage limit")
        if spool_dir is not None:
            spool_dir = Path(spool_dir)
            spool_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory = tempfile.TemporaryDirectory(prefix=f"gigaam-plan-{os.getpid()}-", dir=spool_dir)
        self.samples = None
        self._prepared = None
        self.durable = False
        self.pcm_reused = self.plan_reused = False
        check = cancel_check or (lambda: None)
        try:
            check()
            if source_hash is not None and spool_dir is not None:
                from importlib.metadata import version, PackageNotFoundError
                from prepared_audio import PreparedAudioCache
                versions = {}
                for package in ("faster-whisper", "av"):
                    try:
                        versions[package] = version(package)
                    except PackageNotFoundError:
                        versions[package] = "unknown"
                identity = {"format": "gigaam-f32-16000-v1", **versions}
                cache = PreparedAudioCache(
                    spool_dir / "cache-v1", max_bytes=max_active_bytes,
                    ttl_seconds=float(os.getenv("WHISPER_PCM_CACHE_TTL_SECONDS", "86400")))
                self._prepared = cache.borrow(
                    source_hash, identity, lambda: decode_audio(audio_path, sampling_rate=SAMPLE_RATE),
                    cancel_check=check)
                self.total_samples = self._prepared.metadata["samples"]
                if self.total_samples:
                    self.samples = np.memmap(self._prepared.path, dtype="<f4", mode="r")
                options = {"adapter": GIGAAM_ADAPTER_VERSION, "bounds": "gigaam-context-v1",
                           "vad_filter": bool(vad_filter), "vad_parameters": vad_parameters,
                           "speaker_turns": speaker_turns,
                           "preview_seconds": preview_seconds, "max_chunks": max_chunks}
                self.chunks, self.plan_reused = self._prepared.context(
                    options, lambda: self._chunks(
                        self.samples, self.total_samples, vad_filter, vad_parameters,
                        speaker_turns, preview_seconds, max_chunks), check)
                self.durable = True
                self.pcm_reused = self._prepared.reused
                check()
                return
            audio = decode_audio(audio_path, sampling_rate=SAMPLE_RATE)
            check()
            self.total_samples = len(audio)
            size = self.total_samples * 4
            if size > MAX_PCM_BYTES:
                raise ValueError("Audio exceeds the decoded-audio storage limit")
            if spool_dir is not None:
                used = sum(p.stat().st_size for p in spool_dir.glob("gigaam-plan-*/audio.pcm") if p.is_file())
                if used + size > max_active_bytes:
                    raise ValueError("Concurrent decoded-audio storage limit reached; retry after another job finishes")
            if shutil.disk_usage(self.directory.name).free < size + 1024 ** 3:
                raise ValueError("Not enough temporary storage for audio decoding")
            self.chunks = self._chunks(audio, len(audio), vad_filter, vad_parameters,
                                       speaker_turns, preview_seconds, max_chunks)
            check()
            pcm_path = Path(self.directory.name) / "audio.pcm"
            np.asarray(audio, dtype="<f4").tofile(pcm_path)
            del audio
            if self.total_samples:
                self.samples = np.memmap(pcm_path, dtype="<f4", mode="r")
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _chunks(audio, total_samples, vad_filter, vad_parameters, speaker_turns,
                preview_seconds, max_chunks):
        limit = min(total_samples, int(preview_seconds * SAMPLE_RATE)) if preview_seconds else total_samples
        if not limit:
            return []
        if speaker_turns is not None:
            regions = [{"start": max(0, int(turn["start"] * SAMPLE_RATE)),
                        "end": min(limit, int(turn["end"] * SAMPLE_RATE))}
                       for turn in speaker_turns]
        elif vad_filter:
            options = dict(vad_parameters or {})
            options["max_speech_duration_s"] = 20
            regions = get_speech_timestamps(audio[:limit], VadOptions(**options))
        else:
            regions = [{"start": 0, "end": limit}]
        chunks = [(start, min(start + MAX_CHUNK_SAMPLES, region["end"]))
                  for region in regions
                  for start in range(region["start"], region["end"], MAX_CHUNK_SAMPLES)]
        return chunks[:max_chunks] if max_chunks is not None else chunks

    @property
    def duration(self):
        return self.total_samples / SAMPLE_RATE

    def write_chunk(self, index):
        start, end = self.chunks[index]
        pcm = (np.clip(self.samples[start:end], -1, 1) * 32767).astype("<i2")
        if len(pcm) < MIN_DECODER_SAMPLES:
            pcm = np.pad(pcm, (0, MIN_DECODER_SAMPLES - len(pcm)))
        path = str(Path(self.directory.name) / "chunk.wav")
        with wave.open(path, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(SAMPLE_RATE)
            output.writeframes(pcm.tobytes())
        return path

    def close(self):
        samples, self.samples = self.samples, None
        try:
            if samples is not None:
                samples._mmap.close()
        finally:
            try:
                self.directory.cleanup()
            finally:
                prepared, self._prepared = self._prepared, None
                if prepared is not None:
                    prepared.close()


class GigaAMModel:
    """Expose GigaAM through both the streaming and resumable-worker contracts."""
    supports_audio_chunks = True

    def __init__(self, model: Any):
        self._model = model

    @classmethod
    def from_pretrained(cls, device: str) -> "GigaAMModel":
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError("GigaAM-v3 requires transformers and its runtime dependencies") from exc
        model = AutoModel.from_pretrained(GIGAAM_V3_HF_ID, revision=GIGAAM_V3_REVISION, trust_remote_code=True)
        model = model.to(device)
        model.eval()
        return cls(model)

    def to_device(self, device: str) -> "GigaAMModel":
        if self._model is None:
            raise RuntimeError("GigaAM model is unloaded")
        self._model = self._model.to(device)
        return self

    def unload(self) -> None:
        """Drop an unpublished CPU preparation or a failed GPU transfer."""
        self._model = None

    @property
    def checkpoint_revision(self):
        return str(getattr(getattr(self._model, "config", None), "_commit_hash", None) or GIGAAM_V3_REVISION)

    @staticmethod
    def prepare_chunks(audio_path, *, vad_filter, vad_parameters=None, speaker_turns=None):
        return GigaAMChunkPlan(audio_path, vad_filter=vad_filter, vad_parameters=vad_parameters,
                              speaker_turns=speaker_turns)

    def decode_chunk(self, plan, index, *, language=None, **_):
        """Decode one complete unit; the caller may release its model afterwards."""
        path = plan.write_chunk(index)
        start, end = plan.chunks[index]
        text = str(self._model.transcribe(path) or "").strip()
        segments = [ParakeetSegment(start=start / SAMPLE_RATE, end=end / SAMPLE_RATE, text=text)] if text else []
        return segments, ParakeetInfo(language=language or "ru", language_probability=1.0 if language else 0.0)

    def transcribe(self, audio_path: str, *, language: str | None = None,
                   vad_filter: bool = False, vad_parameters: dict | None = None,
                   speaker_turns: list[dict] | None = None,
                   **_: Any) -> tuple[Iterable[ParakeetSegment], ParakeetInfo]:
        info = ParakeetInfo(language=language or "ru", language_probability=1.0 if language else 0.0)

        def decode_segments():
            if not vad_filter and speaker_turns is None:
                try:
                    text = str(self._model.transcribe(audio_path) or "").strip()
                    if text:
                        yield ParakeetSegment(start=0.0, end=0.0, text=text)
                    return
                except ValueError as exc:
                    if "Too long wav file" not in str(exc):
                        raise
            plan = self.prepare_chunks(audio_path, vad_filter=vad_filter,
                                       vad_parameters=vad_parameters, speaker_turns=speaker_turns)
            try:
                for index in range(len(plan.chunks)):
                    segments, _ = self.decode_chunk(plan, index, language=language)
                    yield from segments
            finally:
                plan.close()

        return decode_segments(), info
