"""Fast unit/focused HTTP integration, expected8s/max30s; no model/GPU/network.

Detects dropped native timestamps on fresh and cached OpenAI responses while
preserving default JSON, header authentication and the existing cache result.
"""
import copy
import io
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
from metrics import transcription_response_payload

NATIVE = {
    "text": "Анна встретит Никиту завтра.",
    "segments": [{"start": 1.25, "end": 3.75, "text": "Анна встретит Никиту завтра.",
                  "avg_logprob": None, "no_speech_prob": None, "compression_ratio": None}],
    "language": "ru", "language_probability": 1.0,
    "requested_model": "whisper-1", "served_model": "gigaam-v3",
    "model_substituted": True, "substitution_reason": "configured_backend",
}

@pytest.fixture
def endpoint(monkeypatch):
    monkeypatch.setenv("API_KEY", "VERBOSE_TEST_ONLY")
    monkeypatch.setattr(main, "pending_results", {})
    monkeypatch.setattr(main, "pending_streams", {})
    monkeypatch.setattr(main, "append_cache_metric", lambda *a: None)
    calls = []
    def put(request):
        calls.append(request)
        assert not request["stream"] and not request["words"]
        main.pending_results.pop(request["request_id"]).set_result(copy.deepcopy(NATIVE))
    monkeypatch.setattr(main.app.state, "request_queue", SimpleNamespace(put=put), raising=False)
    return TestClient(main.app), calls

@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("response_format", [None, "verbose_json"])
def test_http_fresh_and_cache_preserve_format_without_changing_native_result(endpoint, monkeypatch, cached, response_format):
    client, calls = endpoint
    native = copy.deepcopy(NATIVE)
    monkeypatch.setattr(main, "_get_cached_result", lambda *a: native if cached else None)
    data = {"model": "whisper-1"}
    if response_format is not None: data["response_format"] = response_format
    response = client.post("/v1/audio/transcriptions",
        headers={"Authorization": "Bearer VERBOSE_TEST_ONLY"},
        files={"file": ("voice.wav", io.BytesIO(b"RIFF fixture bytes"), "audio/wav")}, data=data)
    assert response.status_code == 200
    result = response.json()
    assert result["text"] == NATIVE["text"]
    for field in ("requested_model", "served_model", "model_substituted", "substitution_reason"):
        assert result[field] == NATIVE[field]
    if response_format == "verbose_json":
        assert result["segments"] == NATIVE["segments"]
        assert result["language"] == "ru"
        assert result["language_probability"] == 1.0
        assert "words" not in result and "duration" not in result
    else:
        assert set(result) == {"text", "requested_model", "served_model", "model_substituted", "substitution_reason"}
    assert len(calls) == (0 if cached else 1)
    assert native == NATIVE

@pytest.mark.parametrize("header", [{}, {"X-API-Key": "VERBOSE_TEST_ONLY"}])
def test_verbose_requires_same_header_auth_not_query_credentials(endpoint, monkeypatch, header):
    client, calls = endpoint
    monkeypatch.setattr(main, "_get_cached_result", lambda *a: NATIVE)
    response = client.post("/v1/audio/transcriptions?api_key=VERBOSE_TEST_ONLY",headers=header,
        files={"file": ("voice.wav", io.BytesIO(b"RIFF fixture bytes"), "audio/wav")},
        data={"model": "whisper-1", "response_format": "verbose_json"})
    assert response.status_code == (200 if header else 401)
    assert not calls

@pytest.mark.parametrize("text", ["", "Legacy transcript without timing"])
def test_verbose_never_fabricates_timing_from_text_only_cache(text):
    source = {"text": text}
    result = transcription_response_payload(source, "verbose_json")
    assert result == {"text": text, "segments": []}
    assert source == {"text": text}
