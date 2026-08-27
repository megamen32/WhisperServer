"""GigaAM-v3 adapter for the server's transcription worker contract."""

from __future__ import annotations

from typing import Any, Iterable

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

    def transcribe(self, audio_path: str, *, language: str | None = None, **_: Any) -> tuple[Iterable[ParakeetSegment], ParakeetInfo]:
        """Decode one audio file and normalize GigaAM's text-only result."""
        text = str(self._model.transcribe(audio_path) or "").strip()
        segments = [ParakeetSegment(start=0.0, end=0.0, text=text)] if text else []
        return segments, ParakeetInfo(language=language or "ru", language_probability=1.0 if language else 0.0)
