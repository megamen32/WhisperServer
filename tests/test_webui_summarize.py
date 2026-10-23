"""Guards for the web UI summarisation tunnel.

The browser must never receive the model credential and must never send the
transcript text: it sends an opaque transcript id and the server forwards the
text it kept itself. These tests pin that contract down.
"""
import json

import pytest
from fastapi.testclient import TestClient

import main as server


@pytest.fixture()
def client():
    with TestClient(server.app) as test_client:
        yield test_client


def _session(client):
    response = client.post("/web/session", headers={"origin": "http://testserver"})
    assert response.status_code == 200
    return response.json()["token"]


def test_transcript_text_stays_on_the_server(client, monkeypatch):
    sid = "session-a"
    transcript_id = server._new_webui_transcript(sid)
    server._fill_webui_transcript(transcript_id, "Секретная расшифровка")

    calls = []

    def fake_open(request, timeout=None):
        calls.append(request)
        raise AssertionError("upstream must not be reached in this test")

    monkeypatch.setattr(server.urllib.request, "build_opener", lambda *a, **k: type(
        "Opener", (), {"open": staticmethod(fake_open)}
    )())

    token = _session(client)
    response = client.post(
        "/web/summarize",
        json={"transcript_id": transcript_id},
        headers={"origin": "http://testserver", "x-whisper-csrf": token},
    )
    assert response.status_code == 404
    assert "Unknown transcript id" in response.json()["detail"]


def test_transcript_belongs_to_its_session(client, monkeypatch):
    server._new_webui_transcript("someone-else")
    transcript_id = server._new_webui_transcript("session-a")
    server._fill_webui_transcript(transcript_id, "Текст")

    token = _session(client)
    response = client.post(
        "/web/summarize",
        json={"transcript_id": transcript_id},
        headers={"origin": "http://testserver", "x-whisper-csrf": token},
    )
    # The session of the browser does not own this transcript.
    assert response.status_code == 404


def test_summarize_requires_csrf_token(client):
    transcript_id = server._new_webui_transcript("session-a")
    server._fill_webui_transcript(transcript_id, "Текст")
    response = client.post(
        "/web/summarize",
        json={"transcript_id": transcript_id},
        headers={"origin": "http://testserver"},
    )
    assert response.status_code == 403


def test_summarize_sends_only_the_id_upstream(client, monkeypatch):
    sid = "session-a"
    transcript_id = server._new_webui_transcript(sid)
    server._fill_webui_transcript(transcript_id, "Текст расшифровки")

    seen = {}

    class FakeResponse:
        def read(self):
            return json.dumps(
                {
                    "model": "MiniMax-M3.1-Flash",
                    "choices": [{"message": {"content": "Готовая выжимка"}}],
                }
            ).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeOpener:
        def open(self, request, timeout=None):
            seen["body"] = json.loads(request.data.decode("utf-8"))
            seen["headers"] = request.headers
            seen["url"] = request.full_url
            return FakeResponse()

    monkeypatch.setattr(server.urllib.request, "build_opener", lambda *a, **k: FakeOpener())
    monkeypatch.setattr(server, "_omniroute_api_key", lambda: "sk-test")

    token = _session(client)
    # Bind the transcript to the session the client actually owns.
    server._webui_transcripts[transcript_id]["session_id"] = client.cookies.get(
        server.WEBUI_SESSION_COOKIE
    )

    response = client.post(
        "/web/summarize",
        json={"transcript_id": transcript_id},
        headers={"origin": "http://testserver", "x-whisper-csrf": token},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"] == "Готовая выжимка"

    # The upstream request carries the text and the key, the browser body does not.
    assert payload["source_chars"] == len("Текст расшифровки")
    assert seen["body"]["messages"][1]["content"] == "Текст расшифровки"
    assert seen["headers"]["Authorization"] == "Bearer sk-test"
    assert "sk-test" not in json.dumps(payload)


def test_summary_rate_limit_blocks_repeated_calls(monkeypatch):
    monkeypatch.setattr(server, "WEBUI_SUMMARY_MAX_PER_SESSION", 2)
    server._webui_summary_calls.clear()
    server._summary_rate_limit("session-a")
    server._summary_rate_limit("session-a")
    with pytest.raises(Exception) as error:
        server._summary_rate_limit("session-a")
    assert "Too many summaries" in str(error.value)
    server._webui_summary_calls.clear()


def test_page_never_contains_the_model_key(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.text
    assert "OMNIROUTE_API_KEY" not in body
    assert "Authorization" not in body
    assert "sk-" not in body
