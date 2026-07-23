"""Tests for OpenAI-compatible /v1/audio/transcriptions endpoint."""

import io
import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock, ANY
from fastapi.testclient import TestClient
import sys
import asyncio
from multiprocessing import Queue, Manager, Event

# Import main app
sys.path.insert(0, str(Path(__file__).parent.parent))
import main
from main import app


@pytest.fixture
def client(monkeypatch):
    """Create a test client with mocked message queue."""
    monkeypatch.delenv("API_KEY", raising=False)

    async def mock_startup():
        """Provide lifecycle state without starting a worker process."""
        loop = asyncio.get_running_loop()
        completed_task = loop.create_future()
        completed_task.set_result(None)

        class FakeRequestQueue:
            """Return a deterministic worker result for endpoint contract tests."""

            def put(self, request):
                future = main.pending_results.pop(request["request_id"])
                future.set_result({
                    "text": "",
                    "segments": [],
                    "language": request.get("language") or "unknown",
                    "language_probability": 0.0,
                    "requested_model": request.get("requested_model", request["model"]),
                    "served_model": "parakeet-v3",
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
        """Avoid touching multiprocessing resources in endpoint unit tests."""
        return None

    monkeypatch.setattr("main.startup_event", mock_startup)
    monkeypatch.setattr("main.shutdown_event", mock_shutdown)

    with TestClient(app) as test_client:
        yield test_client
    main.pending_results.clear()


@pytest.fixture
def sample_audio():
    """Create a mock audio file."""
    # Create a minimal WAV-like file (just headers, no actual audio data)
    wav_header = (
        b'RIFF'  # Chunk ID
        b'\x24\x00\x00\x00'  # Chunk size
        b'WAVE'  # Format
        b'fmt '  # Subchunk1 ID
        b'\x10\x00\x00\x00'  # Subchunk1 size (16 bytes)
        b'\x01\x00'  # Audio format (PCM)
        b'\x01\x00'  # Num channels
        b'\x44\xac\x00\x00'  # Sample rate (44100)
        b'\x10\xb1\x02\x00'  # Byte rate
        b'\x02\x00'  # Block align
        b'\x10\x00'  # Bits per sample
        b'data'  # Subchunk2 ID
        b'\x00\x00\x00\x00'  # Subchunk2 size
    )
    return io.BytesIO(wav_header)


class TestOpenAITranscriptionEndpoint:
    """Test OpenAI-compatible transcription endpoint."""

    def test_endpoint_exists(self, client):
        """Test that /v1/audio/transcriptions endpoint is available."""
        # This should return 422 (Unprocessable Entity) because file is required
        response = client.post("/v1/audio/transcriptions")
        assert response.status_code in [422, 400]

    def test_missing_file_parameter(self, client):
        """Test that missing file parameter returns error."""
        response = client.post(
            "/v1/audio/transcriptions",
            data={
                "model": "whisper-1",
            }
        )
        assert response.status_code == 422

    def test_empty_file_rejected(self, client):
        """Test that empty file is rejected."""
        empty_file = io.BytesIO(b"")
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", empty_file, "audio/wav")},
            data={"model": "whisper-1"}
        )
        assert response.status_code == 400
        assert "Empty audio file" in response.json()["error"]

    def test_unsupported_format_rejected(self, client):
        """Test that unsupported audio formats are rejected."""
        unsupported_file = io.BytesIO(b"invalid audio data")
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.txt", unsupported_file, "text/plain")},
            data={"model": "whisper-1"}
        )
        assert response.status_code == 400
        assert "Unsupported audio format" in response.json()["error"]

    def test_supported_formats_accepted(self, client, sample_audio):
        """Test that supported audio formats are accepted."""
        supported_formats = [
            ("test.mp3", "audio/mpeg"),
            ("test.wav", "audio/wav"),
            ("test.m4a", "audio/mp4"),
            ("test.ogg", "audio/ogg"),
            ("test.flac", "audio/flac"),
            ("test.mp4", "video/mp4"),
            ("test.webm", "audio/webm"),
        ]

        for filename, mime_type in supported_formats:
            sample_audio.seek(0)
            response = client.post(
                "/v1/audio/transcriptions",
                files={"file": (filename, sample_audio, mime_type)},
                data={"model": "whisper-1"}
            )
            # Should not fail due to format error
            if response.status_code == 400:
                assert "format" not in response.json().get("error", "").lower()

    def test_default_model_mapping(self, client, sample_audio):
        """Test that 'whisper-1' model is mapped correctly."""
        # Model name should be mapped via OPENAI_MODEL_MAP
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", sample_audio, "audio/wav")},
            data={"model": "whisper-1"}
        )
        # Should not fail due to unsupported model
        if response.status_code == 400:
            assert "Unsupported model" not in response.json().get("error", "")

    def test_custom_model_name(self, client, sample_audio):
        """Test that custom model names are handled."""
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", sample_audio, "audio/wav")},
            data={"model": "base"}  # Valid faster-whisper model
        )
        # Should process or queue the request (not reject the model)
        if response.status_code == 400:
            assert "Unsupported model" not in response.json().get("error", "")

    def test_optional_parameters(self, client, sample_audio):
        """Test that optional parameters are accepted."""
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", sample_audio, "audio/wav")},
            data={
                "model": "whisper-1",
                "language": "en",
                "temperature": "0.5",
            }
        )
        # Should accept the parameters (not reject due to validation)
        assert response.status_code != 422

    def test_response_format(self, client, sample_audio):
        """Test that response matches OpenAI spec: {'text': '...'}."""
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", sample_audio, "audio/wav")},
            data={"model": "whisper-1"}
        )

        # Check response is JSON with 'text' field when successful
        if response.status_code == 200:
            data = response.json()
            assert "text" in data, "Response must contain 'text' field per OpenAI spec"
            assert isinstance(data["text"], str), "'text' must be a string"


class TestOpenAISDKCompatibility:
    """Test compatibility with OpenAI Python SDK.

    This tests that the endpoint can be used with the official OpenAI SDK.
    """

    @pytest.mark.asyncio
    async def test_openai_sdk_can_connect(self, client):
        """Test that OpenAI SDK can be configured to use our endpoint."""
        try:
            from openai import OpenAI
        except ImportError:
            pytest.skip("openai SDK not installed")

        # This test just verifies the SDK can be imported and instantiated
        # Actual integration test would need a running server
        openai_client = OpenAI(
            api_key="test-key",
            base_url="http://localhost:7653/v1"
        )

        assert openai_client is not None
        assert openai_client.api_key == "test-key"

    def test_endpoint_response_format_matches_openai(self, client, sample_audio):
        """Verify response format matches OpenAI spec.

        Per OpenAI spec:
        - Response should have 'text' field with transcribed text
        - Can optionally have 'segments', 'language', 'language_probability'
        """
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", sample_audio, "audio/wav")},
            data={"model": "whisper-1"}
        )

        if response.status_code == 200:
            data = response.json()

            # Required per OpenAI spec
            assert "text" in data
            assert isinstance(data["text"], str)

            # Check for optional OpenAI-compatible fields
            # (our implementation may include these)
            if "language" in data:
                assert isinstance(data["language"], str)
            if "language_probability" in data:
                assert isinstance(data["language_probability"], (int, float))


class TestEndpointIntegration:
    """Integration tests for the endpoint."""

    def test_endpoint_accepts_multipart_formdata(self, client, sample_audio):
        """Test that endpoint accepts multipart/form-data format."""
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", sample_audio, "audio/wav")},
            data={"model": "whisper-1"}
        )

        # Should not fail due to content-type issues
        assert response.status_code != 415  # Unsupported Media Type

    def test_various_temperature_values(self, client, sample_audio):
        """Test that temperature parameter is accepted for various values."""
        temps = ["0.0", "0.5", "1.0"]

        for temp in temps:
            sample_audio.seek(0)
            response = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", sample_audio, "audio/wav")},
                data={
                    "model": "whisper-1",
                    "temperature": temp,
                }
            )
            # Should accept temperature parameter
            assert response.status_code != 422  # Validation error

    def test_various_languages(self, client, sample_audio):
        """Test that language parameter is accepted for various values."""
        langs = ["en", "es", "fr", "de", None]

        for lang in langs:
            sample_audio.seek(0)
            data = {"model": "whisper-1"}
            if lang:
                data["language"] = lang

            response = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", sample_audio, "audio/wav")},
                data=data
            )
            # Should accept language parameter
            assert response.status_code != 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
