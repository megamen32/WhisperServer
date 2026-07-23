"""NeMo backend for NVIDIA Parakeet TDT v3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Mapping

try:
    import nemo.collections.asr as nemo_asr
except ImportError as exc:  # Keep Whisper-only deployments startable.
    nemo_asr = None
    _NEMO_IMPORT_ERROR = exc
else:
    _NEMO_IMPORT_ERROR = None

from vad_backend import SileroVAD, save_speech_chunk


PARAKEET_V3_HF_ID = "nvidia/parakeet-tdt-0.6b-v3"


@dataclass(frozen=True)
class ParakeetWord:
    """A word timestamp in the format expected by the server worker."""

    start: float
    end: float
    word: str


@dataclass(frozen=True)
class ParakeetSegment:
    """A Parakeet segment normalized to the faster-whisper segment contract."""

    start: float
    end: float
    text: str
    words: tuple[ParakeetWord, ...] = ()


@dataclass(frozen=True)
class ParakeetInfo:
    """Minimal transcription metadata consumed by the server worker."""

    language: str
    language_probability: float


DEFAULT_VAD_PARAMETERS = {
    "threshold": 0.50,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 700,
    "speech_pad_ms": 200,
}


def _timestamp_number(timestamp: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    """Read a numeric timestamp field from a NeMo timestamp mapping."""
    value = timestamp.get(key, default)
    return float(value) if value is not None else default


def _timestamp_text(timestamp: Mapping[str, Any], *keys: str) -> str:
    """Read the first available text field from a NeMo timestamp mapping."""
    for key in keys:
        value = timestamp.get(key)
        if value:
            return str(value).strip()
    return ""


class ParakeetModel:
    """Adapter exposing NeMo Parakeet through the server's model interface."""

    def __init__(self, model: Any):
        """Store an already loaded NeMo model."""
        self._model = model
        self._vad: SileroVAD | None = None

    @classmethod
    def from_pretrained(cls, device: str) -> "ParakeetModel":
        """Load Parakeet v3 from Hugging Face onto the requested device.

        Raises:
            RuntimeError: If the NeMo ASR dependency is not installed.
        """
        if nemo_asr is None:
            raise RuntimeError(
                "Parakeet v3 requires nemo_toolkit[asr]; install the project requirements"
            ) from _NEMO_IMPORT_ERROR

        model = nemo_asr.models.ASRModel.from_pretrained(model_name=PARAKEET_V3_HF_ID)
        model = model.to(device)
        model.eval()
        return cls(model)

    def transcribe(
        self,
        audio_path: str,
        *,
        language: str | None = None,
        word_timestamps: bool = False,
        vad_filter: bool = True,
        vad_parameters: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> tuple[Iterable[ParakeetSegment], ParakeetInfo]:
        """Transcribe one file and normalize NeMo's timestamped hypothesis.

        NeMo performs the full decode before returning. The server therefore
        emits normalized segments after decoding, while preserving timestamps
        and the existing streaming response format.
        """
        parameters = {**DEFAULT_VAD_PARAMETERS, **(vad_parameters or {})}
        if not vad_filter:
            return self._transcribe_file(audio_path, language, word_timestamps, 0.0)

        if self._vad is None:
            self._vad = SileroVAD()
        chunks = self._vad.speech_chunks(audio_path, parameters)
        if not chunks:
            return [], ParakeetInfo(language=language or "unknown", language_probability=0.0)

        all_segments: list[ParakeetSegment] = []
        detected_language = language or "unknown"
        for chunk in chunks:
            with NamedTemporaryFile(suffix=".wav", delete=False) as chunk_file:
                chunk_path = chunk_file.name
            try:
                save_speech_chunk(chunk_path, chunk)
                segments, info = self._transcribe_file(
                    chunk_path,
                    language,
                    word_timestamps,
                    chunk.start_seconds,
                )
            finally:
                Path(chunk_path).unlink(missing_ok=True)
            all_segments.extend(segments)
            if info.language != "unknown":
                detected_language = info.language
        return all_segments, ParakeetInfo(language=detected_language, language_probability=0.0)

    def _transcribe_file(
        self,
        audio_path: str,
        language: str | None,
        include_words: bool,
        offset_seconds: float,
    ) -> tuple[list[ParakeetSegment], ParakeetInfo]:
        """Decode one path and apply an offset from the original audio."""
        outputs = self._model.transcribe([audio_path], timestamps=True)
        hypotheses = outputs[0] if isinstance(outputs, tuple) else outputs
        hypothesis = hypotheses[0]

        timestamp_data = getattr(hypothesis, "timestamp", {}) or {}
        segment_timestamps = timestamp_data.get("segment", [])
        normalized_words = self._normalize_words(timestamp_data.get("word", []))
        segments = [
            self._normalize_segment(timestamp, normalized_words, include_words, offset_seconds)
            for timestamp in segment_timestamps
            if isinstance(timestamp, Mapping)
        ]
        if not segments:
            text = str(getattr(hypothesis, "text", hypothesis) or "").strip()
            if text:
                segments = [ParakeetSegment(start=offset_seconds, end=offset_seconds, text=text)]

        detected_language = language or str(getattr(hypothesis, "language", "unknown") or "unknown")
        return segments, ParakeetInfo(language=detected_language, language_probability=0.0)

    @staticmethod
    def _normalize_words(timestamps: Iterable[Any]) -> tuple[ParakeetWord, ...]:
        """Convert NeMo word timestamp mappings to server word objects."""
        words = []
        for timestamp in timestamps:
            if not isinstance(timestamp, Mapping):
                continue
            word = _timestamp_text(timestamp, "word", "text")
            if word:
                words.append(
                    ParakeetWord(
                        start=_timestamp_number(timestamp, "start"),
                        end=_timestamp_number(timestamp, "end"),
                        word=word,
                    )
                )
        return tuple(words)

    @staticmethod
    def _normalize_segment(
        timestamp: Mapping[str, Any],
        words: tuple[ParakeetWord, ...],
        include_words: bool,
        offset_seconds: float = 0.0,
    ) -> ParakeetSegment:
        """Convert one NeMo segment timestamp into the worker contract."""
        start = _timestamp_number(timestamp, "start")
        end = _timestamp_number(timestamp, "end")
        segment_words = tuple(word for word in words if word.start >= start and word.end <= end)
        return ParakeetSegment(
            start=start + offset_seconds,
            end=end + offset_seconds,
            text=_timestamp_text(timestamp, "segment", "text"),
            words=(
                tuple(
                    ParakeetWord(
                        start=word.start + offset_seconds,
                        end=word.end + offset_seconds,
                        word=word.word,
                    )
                    for word in segment_words
                )
                if include_words
                else ()
            ),
        )
