"""Streaming regressions: show partial Web UI results before completion."""
import asyncio
import json
import queue

import httpx
import numpy as np
import pytest
from starlette.requests import Request

import main
import gigaam_backend
from webui_jobs import JobStore, InvalidTransition


@pytest.fixture(autouse=True)
def isolated_webui(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "WEBUI_JOB_DIR", tmp_path / "jobs")
    monkeypatch.setattr(main.app.state, "job_cancel_flags", {}, raising=False)
    monkeypatch.setattr(main.app.state, "worker_accepting", True, raising=False)
    monkeypatch.setattr(main.app.state, "model_process",
                        type("Alive", (), {"is_alive": lambda self: True})(), raising=False)
    monkeypatch.setattr(main, "_get_cached_result", lambda key: None)
    main._webui_sessions.clear()
    main._webui_transcripts.clear()


@pytest.mark.asyncio
async def test_job_exposes_segments_before_worker_finishes(monkeypatch):
    # Same /web/jobs upload/start/read path as the browser. Only the GPU source
    # is replaced; the actual queue, response listener and durable store run.
    requests, responses = queue.Queue(), queue.Queue()
    monkeypatch.setattr(main.app.state, "request_queue", requests, raising=False)
    listener = asyncio.create_task(main.response_listener(None, responses))
    worker_request = None
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app),
                                     base_url="http://testserver") as client:
            token = (await client.post("/web/session", headers={"origin": "http://testserver"})).json()["token"]
            async def mutate(method, path, **kwargs):
                nonlocal token
                response = await client.request(method, path, headers={
                    "origin": "http://testserver", "x-whisper-csrf": token}, **kwargs)
                token = response.headers.get("x-whisper-next-csrf", token)
                return response
            job = (await mutate("POST", "/web/jobs", json={
                "filename": "stream-check.wav", "size": 3, "model": "gigaam-v3",
                "diarize": True})).json()
            job_id = job["id"]
            await mutate("PUT", f"/web/jobs/{job_id}/upload?offset=0", content=b"abc")
            await mutate("POST", f"/web/jobs/{job_id}/start")
            for _ in range(100):
                if not requests.empty():
                    break
                await asyncio.sleep(.01)
            worker_request = requests.get_nowait()
            # A preliminary block stays in the same job until the final epoch begins.
            responses.put({"request_id": worker_request["request_id"], "progress": {
                "phase": "preliminary", "provisional": True, "total_seconds": 40.0}})
            responses.put({"request_id": worker_request["request_id"], "segment": {
                "start": 0.0, "end": 20.0, "text": "temporary beginning", "provisional": True}})
            for _ in range(100):
                preview = (await client.get(f"/web/jobs/{job_id}")).json()
                if (preview.get("result") or {}).get("segments"):
                    break
                await asyncio.sleep(.01)
            assert preview["progress"].get("provisional") is True
            assert preview["progress"]["phase"] == "preliminary"
            assert preview["progress"]["percent"] is None
            responses.put({"request_id": worker_request["request_id"], "progress": {
                "phase": "diarizing", "provisional": True, "step": "embeddings"}})
            for _ in range(100):
                preview = (await client.get(f"/web/jobs/{job_id}")).json()
                if preview.get("progress", {}).get("phase") == "diarizing":
                    break
                await asyncio.sleep(.01)
            assert preview["result"]["segments"][0]["text"] == "temporary beginning"
            assert preview["progress"]["provisional"] is True
            responses.put({"request_id": worker_request["request_id"], "progress": {
                "phase": "transcribing", "provisional": False, "reset": True,
                "total_seconds": 40.0, "processed_seconds": 0}})
            segment = {"start": 0.0, "end": 20.0, "text": "first fragment"}
            responses.put({"request_id": worker_request["request_id"], "segment": segment})
            for _ in range(100):
                view = (await client.get(f"/web/jobs/{job_id}")).json()
                if ((view.get("result") or {}).get("segments")
                        and view.get("progress", {}).get("phase") == "transcribing"
                        and not view.get("progress", {}).get("provisional")):
                    break
                await asyncio.sleep(.01)
            assert view["status"] == "processing"
            assert (view.get("result") or {}).get("segments") == [segment]
            assert view["result"]["text"] == "first fragment"
            assert view["progress"]["processed_seconds"] == 20.0
            assert view["progress"]["percent"] == 50.0
            sid = client.cookies.get(main.WEBUI_SESSION_COOKIE)
            owner = main._load_webui_session(sid)["owner_id"]
            persisted = JobStore(main.WEBUI_JOB_DIR / "jobs.sqlite").get(owner, job_id)
            assert persisted["result"]["segments"] == [segment]
            # Subscriber attaches after the first fragment and immediately
            # receives the authoritative snapshot, including on reconnect.
            async def receive():
                return {"type": "http.request", "body": b"", "more_body": False}
            request = Request({"type": "http", "method": "GET", "path": f"/web/jobs/{job_id}/events",
                               "headers": [(b"cookie", f"{main.WEBUI_SESSION_COOKIE}={sid}".encode())]}, receive)
            stream = await main.web_jobs_events(job_id, request)
            assert stream.headers["x-accel-buffering"] == "no"
            event = await anext(stream.body_iterator)
            assert json.loads(event.split("data: ", 1)[1])["segments"] == [segment]
            responses.put({"request_id": worker_request["request_id"], "result": {
                "text": "first fragment final", "segments": [segment]}, "final": True})
            worker_request = None
            await asyncio.wait_for(asyncio.gather(*list(main._job_tasks)), 2)
            event = await asyncio.wait_for(anext(stream.body_iterator), 2)
            done = json.loads(event.split("data: ", 1)[1])["job"]
            assert done["status"] == "done" and done["progress"]["percent"] == 100
            with pytest.raises(StopAsyncIteration):
                await anext(stream.body_iterator)
    finally:
        if worker_request:
            responses.put({"request_id": worker_request["request_id"], "result": {"text": "cleanup"}, "final": True})
            await asyncio.wait_for(asyncio.gather(*list(main._job_tasks)), 2)
        responses.put(None)
        await asyncio.wait_for(listener, 2)


def test_gigaam_yields_first_chunk_before_decoding_the_rest(monkeypatch):
    monkeypatch.setattr(gigaam_backend, "decode_audio", lambda *a, **k: np.zeros(42 * 16000))
    monkeypatch.setattr(gigaam_backend, "get_speech_timestamps",
                        lambda *a: [{"start": 0, "end": 42 * 16000}])
    decoded = []
    class Decoder:
        def transcribe(self, path):
            decoded.append(path)
            return f"fragment {len(decoded)}"
    segments, _ = gigaam_backend.GigaAMModel(Decoder()).transcribe("voice.wav", vad_filter=True)
    iterator = iter(segments)
    first = next(iterator)
    assert first.end == 20
    assert len(decoded) == 1, "Later chunks must not block the first streamed result"
    assert [s.end for s in iterator] == [40, 42]


@pytest.mark.asyncio
async def test_progress_event_stream_requires_job_owner():
    store = main._job_store()
    job = store.create("owner", {"filename": "x", "size": 1, "model": "small", "diarize": False})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app),
                                 base_url="http://testserver") as client:
        assert (await client.get(f"/web/jobs/{job['id']}/events")).status_code == 403
        await client.post("/web/session", headers={"origin": "http://testserver"})
        assert (await client.get(f"/web/jobs/{job['id']}/events")).status_code == 404

def test_progress_keeps_worker_oom_retry_and_resets_partial_decode(monkeypatch):
    from types import SimpleNamespace
    beams = []
    class Model:
        def transcribe(self, path, **options):
            beam = options["beam_size"]
            beams.append(beam)
            def decode():
                yield SimpleNamespace(start=0, end=5, text=f"beam {beam}", words=None)
                if beam == 5:
                    raise RuntimeError("CUDA out of memory")
            return decode(), SimpleNamespace(language="ru", language_probability=1)
    monkeypatch.setattr(main, "ManagedModel", None)
    monkeypatch.setattr(main, "_create_model_entry", lambda name: Model())
    monkeypatch.setattr(main, "audio_duration_seconds", lambda path: 10)
    monkeypatch.setattr(main, "append_metric", lambda metric: None)
    monkeypatch.setattr(main, "reset_cuda_peak_memory", lambda: None)
    monkeypatch.setattr(main, "peak_vram_mb", lambda: 0)
    monkeypatch.setattr(main, "release_cached_models", lambda *a, **k: None)
    requests, responses = queue.Queue(), queue.Queue()
    requests.put({"request_id": "retry", "model": "gigaam-v3", "audio_bytes": b"audio",
                  "beam_size": 5, "progress": True, "stream": False})
    checks = iter([False, True])
    main.model_worker(requests, responses, SimpleNamespace(is_set=lambda: next(checks)), {})
    emitted = []
    while not responses.empty():
        emitted.append(responses.get_nowait())
    assert beams == [5, main.RETRY_BEAM_SIZE]
    resets = [event for event in emitted if event.get("progress", {}).get("reset")]
    assert len(resets) == 2
    phases = [event["progress"]["phase"] for event in emitted if "progress" in event]
    assert phases[:3] == ["preparing", "loading", "transcribing"]
    assert emitted[-1]["final"] and emitted[-1]["result"]["text"] == f"beam {main.RETRY_BEAM_SIZE}"
    assert not any("error" in event for event in emitted)
