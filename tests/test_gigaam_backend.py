"""Tests for the GigaAM-v3 to worker adapter."""

from gigaam_backend import GigaAMModel


class _FakeGigaAM:
    def transcribe(self, audio_path):
        assert audio_path == "sample.wav"
        return "Привет, мир!"


def test_gigaam_transcription_matches_worker_contract():
    segments, info = GigaAMModel(_FakeGigaAM()).transcribe("sample.wav", language="ru")
    segment = list(segments)[0]
    assert segment.text == "Привет, мир!"
    assert segment.start == 0.0
    assert segment.end == 0.0
    assert info.language == "ru"
