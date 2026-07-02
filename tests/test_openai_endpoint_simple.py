"""Simple tests for OpenAI-compatible endpoint validation."""

import io
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
import sys
from unittest.mock import MagicMock

# Import main app
sys.path.insert(0, str(Path(__file__).parent.parent))
from main import app


TEST_API_KEY = "WHISPER_TEST_API_KEY"


def auth_headers():
    return {"Authorization": f"Bearer {TEST_API_KEY}"}


@pytest.fixture
def client(monkeypatch):
    """Create a test client with mocked startup/shutdown."""
    monkeypatch.setenv("API_KEY", TEST_API_KEY)
    # Mock the startup to avoid worker process
    async def mock_startup():
        app.state.request_queue = MagicMock()
        app.state.response_queue = MagicMock()
        app.state.executor = MagicMock()
        app.state.stop_event = MagicMock()
        app.state.model_usage = {}
        app.state.response_listener_task = MagicMock()
        app.state.lifecycle_stop = MagicMock()
        app.state.bot_process = None
        app.state.bot_monitor_task = MagicMock()

    async def mock_shutdown():
        pass

    # Patch startup and shutdown
    import main
    monkeypatch.setattr(main, "startup_event", mock_startup)
    monkeypatch.setattr(main, "shutdown_event", mock_shutdown)

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_audio():
    """Create a minimal WAV file."""
    wav_header = (
        b'RIFF'
        b'\x24\x00\x00\x00'
        b'WAVE'
        b'fmt '
        b'\x10\x00\x00\x00'
        b'\x01\x00'
        b'\x01\x00'
        b'\x44\xac\x00\x00'
        b'\x10\xb1\x02\x00'
        b'\x02\x00'
        b'\x10\x00'
        b'data'
        b'\x00\x00\x00\x00'
    )
    return io.BytesIO(wav_header)


class TestOpenAIEndpointValidation:
    """Test OpenAI endpoint validation and request handling."""

    def test_endpoint_exists(self, client):
        """Test that /v1/audio/transcriptions endpoint exists."""
        response = client.post("/v1/audio/transcriptions", headers=auth_headers())
        assert response.status_code in [422, 400]  # Missing required fields

    def test_missing_file_returns_422(self, client):
        """Test that missing file parameter returns 422 (validation error)."""
        response = client.post(
            "/v1/audio/transcriptions",
            headers=auth_headers(),
            data={"model": "whisper-1"}
        )
        assert response.status_code == 422

    def test_empty_file_returns_400(self, client):
        """Test that empty file is rejected."""
        empty_file = io.BytesIO(b"")
        response = client.post(
            "/v1/audio/transcriptions",
            headers=auth_headers(),
            files={"file": ("test.wav", empty_file, "audio/wav")},
            data={"model": "whisper-1"}
        )
        assert response.status_code == 400
        assert "Empty audio file" in response.json()["error"]

    def test_unsupported_format_returns_400(self, client):
        """Test that unsupported format is rejected."""
        unsupported = io.BytesIO(b"not audio data")
        response = client.post(
            "/v1/audio/transcriptions",
            headers=auth_headers(),
            files={"file": ("test.txt", unsupported, "text/plain")},
            data={"model": "whisper-1"}
        )
        assert response.status_code == 400
        assert "Unsupported audio format" in response.json()["error"]

    def test_supported_formats(self, client, sample_audio):
        """Test that all supported formats are accepted."""
        formats = [
            ("test.mp3", "audio/mpeg"),
            ("test.wav", "audio/wav"),
            ("test.m4a", "audio/mp4"),
            ("test.ogg", "audio/ogg"),
            ("test.flac", "audio/flac"),
            ("test.mp4", "video/mp4"),
            ("test.webm", "audio/webm"),
        ]

        for filename, mime_type in formats:
            sample_audio.seek(0)
            response = client.post(
                "/v1/audio/transcriptions",
                files={"file": (filename, sample_audio, mime_type)},
                data={"model": "whisper-1"}
            )
            # Should not reject due to format
            if response.status_code == 400:
                error = response.json().get("error", "")
                assert "format" not in error.lower(), f"Format {filename} rejected: {error}"

    def test_model_parameter_accepted(self, client, sample_audio):
        """Test that model parameter is accepted."""
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", sample_audio, "audio/wav")},
            data={"model": "whisper-1"}
        )
        # Should not fail validation
        assert response.status_code != 422

    def test_optional_language_parameter(self, client, sample_audio):
        """Test that language parameter is optional."""
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", sample_audio, "audio/wav")},
            data={
                "model": "whisper-1",
                "language": "en"
            }
        )
        # Should not fail validation
        assert response.status_code != 422

    def test_optional_temperature_parameter(self, client, sample_audio):
        """Test that temperature parameter is optional."""
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", sample_audio, "audio/wav")},
            data={
                "model": "whisper-1",
                "temperature": "0.5"
            }
        )
        # Should not fail validation
        assert response.status_code != 422

    def test_invalid_model_name(self, client, sample_audio):
        """Test that invalid model name is rejected."""
        response = client.post(
            "/v1/audio/transcriptions",
            headers=auth_headers(),
            files={"file": ("test.wav", sample_audio, "audio/wav")},
            data={"model": "invalid-model-xyz"}
        )
        # Should reject unknown model
        assert response.status_code == 400
        assert "Unsupported model" in response.json()["error"]

    def test_valid_whisper_1_model(self, client, sample_audio):
        """Test that 'whisper-1' model is mapped correctly."""
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", sample_audio, "audio/wav")},
            data={"model": "whisper-1"}
        )
        # Should not reject model
        if response.status_code == 400:
            error = response.json().get("error", "")
            assert "Unsupported model" not in error

    def test_valid_faster_whisper_models(self, client, sample_audio):
        """Test that faster-whisper model names work."""
        valid_models = ["base", "small", "medium", "large-v3"]

        for model in valid_models:
            sample_audio.seek(0)
            response = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("test.wav", sample_audio, "audio/wav")},
                data={"model": model}
            )
            # Should not reject model
            if response.status_code == 400:
                error = response.json().get("error", "")
                assert "Unsupported model" not in error, f"Model {model} rejected"


class TestOpenAISpecCompliance:
    """Test compliance with OpenAI API specification."""

    def test_multipart_formdata_accepted(self, client, sample_audio):
        """Test that endpoint accepts multipart/form-data."""
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", sample_audio, "audio/wav")},
            data={"model": "whisper-1"}
        )
        # Should not return 415 (Unsupported Media Type)
        assert response.status_code != 415

    def test_response_json_format(self, client, sample_audio):
        """Test that successful response is valid JSON."""
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", sample_audio, "audio/wav")},
            data={"model": "whisper-1"}
        )
        # Should be valid JSON
        if response.status_code < 500:
            data = response.json()
            assert isinstance(data, dict)

    def test_error_response_has_error_field(self, client):
        """Test that error responses have 'error' field."""
        response = client.post(
            "/v1/audio/transcriptions",
            headers=auth_headers(),
            files={"file": ("test.txt", io.BytesIO(b"not audio data"), "text/plain")},
            data={"model": "whisper-1"}
        )
        # Error response should have error field
        if response.status_code >= 400:
            data = response.json()
            assert "error" in data, "Error response should include 'error' field"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
