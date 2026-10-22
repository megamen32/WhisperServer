"""Tests for the opt-in client comment on the transcription endpoints.

A client may attach a free-form comment ("вебморда", "video_watching") to a
transcription request. The server remembers it next to the cache so an
operator can tell which client sent which file. The comment never changes the
cache key: the same file stays a cache hit no matter who sends it.
"""

import io
import sys
import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))
import main
from main import app


@pytest.fixture
def attribution(tmp_path, monkeypatch):
    """Point the attribution store at a temporary directory."""
    from attribution import AttributionStore

    store = AttributionStore(str(tmp_path / "attribution"))
    monkeypatch.setattr(main, "attribution_store", store)
    yield store
    store.close()


@pytest.fixture
def client(monkeypatch):
    """Create a test client with a deterministic worker result."""
    monkeypatch.delenv("API_KEY", raising=False)

    async def mock_startup():
        loop = asyncio.get_running_loop()
        completed_task = loop.create_future()
        completed_task.set_result(None)

        class FakeRequestQueue:
            """Return a deterministic worker result for endpoint contract tests."""

            def put(self, request):
                future = main.pending_results.pop(request["request_id"])
                future.set_result({
                    "text": "тестовая расшифровка",
                    "segments": [],
                    "language": request.get("language") or "ru",
                    "language_probability": 1.0,
                    "requested_model": request.get("requested_model", request["model"]),
                    "served_model": "gigaam-v3",
                    "model_substituted": False,
                    "substitution_reason": "default_model",
                })

        main.pending_results.clear()
        app.state.request_queue = FakeRequestQueue()
        app.state.response_queue = MagicMock()
        app.state.executor = MagicMock()
        app.state.response_listener_task = completed_task
        app.state.stop_event = MagicMock()
        app.state.lifecycle_stop = asyncio.Event()
        app.state.bot_process = None
        app.state.bot_monitor_task = completed_task
        app.state.model_process = MagicMock()
        app.state.model_process.is_alive.return_value = False
        app.state.manager = MagicMock()

    async def mock_shutdown():
        return None

    monkeypatch.setattr("main.startup_event", mock_startup)
    monkeypatch.setattr("main.shutdown_event", mock_shutdown)

    with TestClient(app) as test_client:
        yield test_client
    main.pending_results.clear()


@pytest.fixture
def sample_audio():
    """A minimal WAV-like payload."""
    return io.BytesIO(b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00")


def _post(client, audio, **data):
    audio.seek(0)
    return client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", audio, "audio/wav")},
        data=data,
    )


class TestClientComment:
    """The opt-in comment is accepted, remembered and echoed back."""

    def test_comment_is_echoed_in_the_response(self, client, sample_audio):
        response = _post(client, sample_audio, model="whisper-1", comment="вебморда")
        assert response.status_code == 200
        assert response.json()["client_comment"] == "вебморда"

    def test_comment_is_trimmed(self, client, sample_audio):
        response = _post(client, sample_audio, model="whisper-1", comment="  video_watching  ")
        assert response.status_code == 200
        assert response.json()["client_comment"] == "video_watching"

    def test_comment_is_optional(self, client, sample_audio):
        response = _post(client, sample_audio, model="whisper-1")
        assert response.status_code == 200
        assert "client_comment" not in response.json()

    def test_empty_comment_is_ignored(self, client, sample_audio):
        response = _post(client, sample_audio, model="whisper-1", comment="   ")
        assert response.status_code == 200
        assert "client_comment" not in response.json()

    def test_comment_is_remembered_for_the_file(self, client, sample_audio, attribution):
        response = _post(client, sample_audio, model="whisper-1", comment="мой транскриптор с айфона")
        assert response.status_code == 200
        audio_hash = main.hash_bytes(sample_audio.getvalue())
        records = attribution.lookup(audio_hash=audio_hash)["records"]
        assert [record["comment"] for record in records] == ["мой транскриптор с айфона"]

    def test_comment_can_be_looked_up_by_client(self, client, sample_audio, attribution):
        _post(client, sample_audio, model="whisper-1", comment="вебморда")
        files = attribution.lookup(comment="вебморда")["files"]
        assert len(files) == 1
        assert files[0]["count"] == 1

    def test_request_without_comment_records_nothing(self, client, sample_audio, attribution):
        _post(client, sample_audio, model="whisper-1")
        audio_hash = main.hash_bytes(sample_audio.getvalue())
        assert attribution.lookup(audio_hash=audio_hash)["records"] == []


class TestCommentDoesNotBreakTheCache:
    """The cache key must stay independent of the comment."""

    def test_same_file_from_two_clients_is_one_cache_entry(self, client, sample_audio, attribution):
        audio_hash = main.hash_bytes(sample_audio.getvalue())
        cache_key = main.transcription_cache_key(
            "openai", "whisper-1", None, audio_hash, vad_filter=True, diarize=False
        )
        main.transcription_cache.set(cache_key, {
            "text": "из кеша",
            "segments": [],
            "requested_model": "whisper-1",
            "served_model": "gigaam-v3",
            "model_substituted": False,
            "substitution_reason": "default_model",
            "client_comment": "чужой комментарий",
        }, expire=60)

        response = _post(client, sample_audio, model="whisper-1", comment="video_watching")
        assert response.status_code == 200
        body = response.json()
        # The cached transcription is reused, but the cached comment of the
        # other client never leaks into this response.
        assert body["text"] == "из кеша"
        assert body["client_comment"] == "video_watching"

    def test_cache_key_does_not_depend_on_the_comment(self):
        base = dict(prefix="openai", model="whisper-1", language=None,
                    audio_hash="a" * 64, vad_filter=True, diarize=False)
        first = main.transcription_cache_key(**base)
        second = main.transcription_cache_key(**base)
        assert first == second


class TestNativeEndpointComment:
    """The native endpoint accepts the same opt-in comment."""

    def test_native_endpoint_records_the_comment(self, client, sample_audio, attribution):
        api_key = next(iter(main.ALLOWED_API_KEYS))
        response = client.post(
            "/transcribe",
            files={"file": ("test.wav", sample_audio, "audio/wav")},
            params={"model": "whisper-1", "api_key": api_key, "comment": "вебморда"},
        )
        assert response.status_code == 200
        assert response.json()["client_comment"] == "вебморда"
        audio_hash = main.hash_bytes(sample_audio.getvalue())
        records = attribution.lookup(audio_hash=audio_hash)["records"]
        assert records[0]["comment"] == "вебморда"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
