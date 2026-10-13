"""GigaAM-v3 adapter for the server's transcription worker contract."""

from __future__ import annotations

from typing import Any, Iterable
import tempfile
import wave
from pathlib import Path

import numpy as np
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps

try:
    from transformers import AutoModel
except ImportError as exc:
    AutoModel = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None

from parakeet_backend import ParakeetInfo, ParakeetSegment

GIGAAM_V3_HF_ID = "ai-sage/GigaAM-v3"
GIGAAM_V3_REVISION = "e2e_rnnt"
# Valid feature frames even for tiny turns/tails; original timestamps stay unchanged.
MIN_DECODER_SAMPLES = 3200


class GigaAMModel:
    """Expose GigaAM-v3 e2e_rnnt through the worker's common contract."""

    def __init__(self, model: Any):
        self._model = model

    @classmethod
    def from_pretrained(cls, device: str) -> "GigaAMModel":
        if AutoModel is None:
            raise RuntimeError("GigaAM-v3 requires transformers and its runtime dependencies") from _TRANSFORMERS_IMPORT_ERROR
        model = AutoModel.from_pretrained(GIGAAM_V3_HF_ID, revision=GIGAAM_V3_REVISION, trust_remote_code=True)
        model = model.to(device)
        model.eval()
        return cls(model)

    def transcribe(self, audio_path: str, *, language: str | None = None,
                   vad_filter: bool = False, vad_parameters: dict | None = None,
                   speaker_turns: list[dict] | None = None,
                   **_: Any) -> tuple[Iterable[ParakeetSegment], ParakeetInfo]:
        """Decode one audio file and normalize GigaAM's text-only result."""
        info = ParakeetInfo(language=language or "ru", language_probability=1.0 if language else 0.0)
        if not vad_filter and speaker_turns is None:
            try:
                text = str(self._model.transcribe(audio_path) or "").strip()
                return ([ParakeetSegment(start=0.0, end=0.0, text=text)] if text else []), info
            except ValueError as exc:
                if "Too long wav file" not in str(exc):
                    raise

        # GigaAM's short decoder is limited to 25 seconds. Use the existing
        # local Silero frontend instead of requiring gated pyannote weights.
        rate = 16000
        limit = 20 * rate
        audio = decode_audio(audio_path, sampling_rate=rate)
        if speaker_turns is not None:
            # Decode each exclusive speaker turn separately: GigaAM has no word
            # timestamps, so tagging a whole VAD region would mix speakers.
            regions = [{'start': max(0, int(turn['start'] * rate)),
                        'end': min(len(audio), int(turn['end'] * rate))}
                       for turn in speaker_turns]
        elif vad_filter:
            options = dict(vad_parameters or {})
            options['max_speech_duration_s'] = 20
            regions = get_speech_timestamps(audio, VadOptions(**options))
        else:
            regions = [{'start': 0, 'end': len(audio)}]
        segments = []
        with tempfile.TemporaryDirectory(prefix='gigaam-chunks-') as directory:
            path = str(Path(directory) / 'chunk.wav')
            for region in regions:
                # Bound even padded VAD regions and uninterrupted speech.
                for start in range(region['start'], region['end'], limit):
                    end = min(start + limit, region['end'])
                    pcm = (np.clip(audio[start:end], -1, 1) * 32767).astype('<i2')
                    if len(pcm) < MIN_DECODER_SAMPLES:
                        pcm = np.pad(pcm, (0, MIN_DECODER_SAMPLES - len(pcm)))
                    with wave.open(path, 'wb') as out:
                        out.setnchannels(1)
                        out.setsampwidth(2)
                        out.setframerate(rate)
                        out.writeframes(pcm.tobytes())
                    text = str(self._model.transcribe(path) or '').strip()
                    if text:
                        segments.append(ParakeetSegment(start=start/rate, end=end/rate, text=text))
        return segments, info
