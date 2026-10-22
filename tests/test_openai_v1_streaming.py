"""Tests for OpenAI-compatible streaming transcription behavior."""

import io
import json
from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

import main
from main import app, cache, pending_streams


class FakeStreamingQueue:
    """Satisfy app.state.request_queue.put without a real model worker."""

    def put(self, request):
        assert request["stream"] is True
        # The alias resolves to the configured backend before it is queued, so
        # the expectation follows the deployment instead of a fixed model.
        assert request["model"] == main.OPENAI_MODEL_MAP.get("whisper-1", "whisper-1")
        queue = pending_streams[request["request_id"]]
        queue.put_nowait({"segment": {"id": 0, "start": 0.0, "end": 1.0, "text": "hello"}})
        queue.put_nowait({"result": {"text": "hello", "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}})
        queue.put_nowait(None)


def _wav_bytes():
    return io.BytesIO(
        b"RIFF\x24\x00\x00\x00WAVEfmt "
        b"\x10\x00\x00\x00\x01\x00\x01\x00"
        b"\x40\x1f\x00\x00\x80\x3e\x00\x00"
        b"\x02\x00\x10\x00data\x00\x00\x00\x00"
    )


def _openai_cache_key(audio_hash, model="whisper-1", language=None, vad_filter=True):
    """Build the cache key the OpenAI endpoint uses for a public model name.

    The endpoint resolves the public model name to its backend first, so the
    key depends on the configured default. Mirroring that here keeps the test
    independent of the deployment model.
    """
    actual_model = main.OPENAI_MODEL_MAP.get(model, model)
    cache_key_model = model if actual_model == main.OPENAI_WHISPER_INTERNAL_MODEL else actual_model
    return main.transcription_cache_key("openai", cache_key_model, language, audio_hash, vad_filter=vad_filter)


def _sse_events(body: str):
    events = []
    for block in body.strip().split("\n\n"):
        if not block:
            continue
        assert block.startswith("data: ")
        events.append(json.loads(block[len("data: ") :]))
    return events


def test_openai_v1_transcriptions_stream_true_returns_sse(monkeypatch):
    monkeypatch.setenv("API_KEY", "WHISPER_TEST_API_KEY")
    cache.clear()
    pending_streams.clear()
    app.state.request_queue = FakeStreamingQueue()

    client = TestClient(app)
    response = client.post(
        "/v1/audio/transcriptions",
        headers={"Authorization": "Bearer WHISPER_TEST_API_KEY"},
        files={"file": ("voice.wav", _wav_bytes(), "audio/wav")},
        data={"model": "whisper-1", "stream": "true"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-accel-buffering"] == "no"

    events = _sse_events(response.text)
    assert events == [
        {"type": "transcript.text.delta", "delta": "hello", "segment_id": "0"},
        {"type": "transcript.text.done", "text": "hello"},
    ]
    assert pending_streams == {}


def test_openai_v1_streaming_requires_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "WHISPER_TEST_API_KEY")
    cache.clear()
    app.state.request_queue = FakeStreamingQueue()

    client = TestClient(app)
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("voice.wav", _wav_bytes(), "audio/wav")},
        data={"model": "whisper-1", "stream": "true"},
    )

    assert response.status_code == 401
    assert response.json()["detail"]["error"]["code"] == "invalid_api_key"


def test_openai_v1_streaming_cache_hit_uses_sse(monkeypatch):
    monkeypatch.setenv("API_KEY", "WHISPER_TEST_API_KEY")
    cache.clear()
    audio = _wav_bytes().getvalue()
    audio_hash = main.hash_bytes(audio)
    cache_key = _openai_cache_key(audio_hash)
    cache[cache_key] = {"text": "cached hello"}

    client = TestClient(app)
    response = client.post(
        "/v1/audio/transcriptions",
        headers={"Authorization": "Bearer WHISPER_TEST_API_KEY"},
        files={"file": ("voice.wav", io.BytesIO(audio), "audio/wav")},
        data={"model": "whisper-1", "stream": "true"},
    )

    assert response.status_code == 200
    events = _sse_events(response.text)
    assert events == [
        {"type": "transcript.text.delta", "delta": "cached hello"},
        {"type": "transcript.text.done", "text": "cached hello"},
    ]


def test_openai_v1_transcriptions_cache_hit_returns_json(monkeypatch):
    """Exercise the normal OpenAI POST cache-hit path without a worker call."""
    monkeypatch.setenv("API_KEY", "WHISPER_TEST_API_KEY")
    cache.clear()
    audio = _wav_bytes().getvalue()
    cache_key = _openai_cache_key(main.hash_bytes(audio))
    cache[cache_key] = {"text": "cached hello"}

    client = TestClient(app)
    response = client.post(
        "/v1/audio/transcriptions",
        headers={"Authorization": "Bearer WHISPER_TEST_API_KEY"},
        files={"file": ("voice.wav", io.BytesIO(audio), "audio/wav")},
        data={"model": "whisper-1"},
    )

    assert response.status_code == 200
    assert response.json() == {"text": "cached hello"}
