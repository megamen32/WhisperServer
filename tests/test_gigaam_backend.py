"""Tests for the GigaAM-v3 to worker adapter."""

from gigaam_backend import GigaAMModel
import gigaam_backend
import wave
import numpy as np


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


def test_long_audio_is_split_under_gigaam_limit(tmp_path):
    audio = tmp_path / 'long.wav'
    with wave.open(str(audio), 'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(np.zeros(42 * 16000, dtype=np.int16).tobytes())
    calls = []
    class ShortOnly:
        def transcribe(self, path):
            with wave.open(path) as inp:
                duration = inp.getnframes() / inp.getframerate()
            if duration > 25:
                raise ValueError("Too long wav file, use 'transcribe_longform' method.")
            calls.append(duration)
            return 'текст'
    segments, _ = GigaAMModel(ShortOnly()).transcribe(str(audio), vad_filter=False)
    segments = list(segments)
    assert len(calls) == 3
    assert max(calls) <= 20
    assert segments[0].start == 0
    assert segments[-1].end == 42


def test_vad_silence_does_not_call_decoder(monkeypatch):
    monkeypatch.setattr(gigaam_backend, 'decode_audio', lambda *a, **k: np.zeros(16000))
    monkeypatch.setattr(gigaam_backend, 'get_speech_timestamps', lambda *a: [])
    class NoSpeech:
        def transcribe(self, path):
            raise AssertionError('silence must not reach ASR')
    segments, _ = GigaAMModel(NoSpeech()).transcribe('silence.wav', vad_filter=True)
    assert list(segments) == []


def test_vad_preserves_original_offsets_and_bounds_padded_regions(monkeypatch):
    monkeypatch.setattr(gigaam_backend, 'decode_audio', lambda *a, **k: np.zeros(50 * 16000))
    def vad(audio, options):
        assert options.max_speech_duration_s == 20
        return [{'start': 16000, 'end': 43 * 16000}]
    monkeypatch.setattr(gigaam_backend, 'get_speech_timestamps', vad)
    class Decoder:
        def transcribe(self, path):
            with wave.open(path) as inp:
                assert inp.getnframes() <= 20 * 16000
            return 'речь'
    segments, _ = GigaAMModel(Decoder()).transcribe('voice.wav', vad_filter=True)
    assert [(s.start, s.end) for s in segments] == [(1, 21), (21, 41), (41, 43)]
