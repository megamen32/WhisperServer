"""Tests for the Parakeet-to-server model adapter."""

from parakeet_backend import ParakeetModel


class _Hypothesis:
    text = "Привет мир"
    timestamp = {
        "segment": [{"start": 0.1, "end": 1.2, "segment": "Привет мир"}],
        "word": [
            {"start": 0.1, "end": 0.5, "word": "Привет"},
            {"start": 0.6, "end": 1.2, "word": "мир"},
        ],
    }


class _FakeNeMoModel:
    def transcribe(self, paths, timestamps):
        assert paths == ["sample.wav"]
        assert timestamps is True
        return [_Hypothesis()]


class _EmptyVAD:
    def speech_chunks(self, audio_path, parameters):
        assert audio_path == "sample.wav"
        assert parameters["threshold"] == 0.5
        return []


def test_parakeet_transcription_matches_worker_contract():
    """Normalize NeMo hypotheses into segments, words, and info metadata."""
    model = ParakeetModel(_FakeNeMoModel())

    segments, info = model.transcribe(
        "sample.wav",
        language="ru",
        word_timestamps=True,
        vad_filter=False,
    )
    segment = list(segments)[0]

    assert segment.text == "Привет мир"
    assert segment.start == 0.1
    assert segment.end == 1.2
    assert [word.word for word in segment.words] == ["Привет", "мир"]
    assert info.language == "ru"
    assert info.language_probability == 0.0


def test_parakeet_vad_drops_audio_without_speech():
    """Avoid invoking Parakeet when the shared VAD finds no speech."""
    model = ParakeetModel(_FakeNeMoModel())
    model._vad = _EmptyVAD()

    segments, info = model.transcribe("sample.wav", vad_filter=True)

    assert list(segments) == []
    assert info.language == "unknown"
