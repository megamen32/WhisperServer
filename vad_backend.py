"""Shared Silero VAD frontend for model backends without built-in VAD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

try:
    import torchaudio
    from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
except ImportError as exc:  # The error is raised explicitly when this backend is used.
    torchaudio = None
    get_speech_timestamps = None
    load_silero_vad = None
    read_audio = None
    _VAD_IMPORT_ERROR = exc
else:
    _VAD_IMPORT_ERROR = None


SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class SpeechChunk:
    """A speech-only waveform chunk with offsets in the original audio."""

    samples: torch.Tensor
    start_seconds: float
    end_seconds: float


class SileroVAD:
    """Extract speech chunks with the official Silero VAD package."""

    def __init__(self) -> None:
        """Load the small VAD model on CPU."""
        if _VAD_IMPORT_ERROR is not None:
            raise RuntimeError(
                "Silero VAD requires silero-vad and torchaudio; install project requirements"
            ) from _VAD_IMPORT_ERROR
        self._model = load_silero_vad()
        self._model.eval()

    def speech_chunks(
        self,
        audio_path: str,
        parameters: Mapping[str, Any],
    ) -> list[SpeechChunk]:
        """Return speech-only chunks while preserving original time offsets."""
        waveform = read_audio(audio_path, sampling_rate=SAMPLE_RATE)
        timestamps = get_speech_timestamps(
            waveform,
            self._model,
            sampling_rate=SAMPLE_RATE,
            threshold=float(parameters.get("threshold", 0.50)),
            min_speech_duration_ms=int(parameters.get("min_speech_duration_ms", 250)),
            min_silence_duration_ms=int(parameters.get("min_silence_duration_ms", 700)),
            speech_pad_ms=int(parameters.get("speech_pad_ms", 200)),
            return_seconds=False,
        )
        return [
            SpeechChunk(
                samples=waveform[int(timestamp["start"]):int(timestamp["end"])]
                .detach()
                .cpu()
                .contiguous(),
                start_seconds=int(timestamp["start"]) / SAMPLE_RATE,
                end_seconds=int(timestamp["end"]) / SAMPLE_RATE,
            )
            for timestamp in timestamps
        ]


def save_speech_chunk(path: str, chunk: SpeechChunk) -> None:
    """Write a speech chunk as a mono WAV file for a path-based ASR backend."""
    if torchaudio is None:
        raise RuntimeError("torchaudio is required to write Silero VAD chunks") from _VAD_IMPORT_ERROR
    samples = chunk.samples
    if samples.ndim == 1:
        samples = samples.unsqueeze(0)
    torchaudio.save(path, samples, SAMPLE_RATE)
