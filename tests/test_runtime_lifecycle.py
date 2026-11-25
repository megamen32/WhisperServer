"""Runtime regressions: cheap imports, accurate readiness and bounded reuse."""
import asyncio
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import httpx
import pytest

import main


def test_web_import_does_not_load_model_backends():
    code = """
import builtins
original = builtins.__import__
blocked = {'torch', 'torchaudio', 'nemo', 'transformers', 'faster_whisper'}
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise AssertionError('HTTP import loaded model backend: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import main
assert main.app is not None
"""
    env = dict(os.environ, MODEL="small")
    result = subprocess.run([sys.executable, "-c", code],
                            cwd=Path(__file__).resolve().parents[1],
                            env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr[-3000:]


@pytest.mark.asyncio
async def test_health_follows_worker_admission_and_handshake(monkeypatch):
    state = main.app.state
    monkeypatch.setattr(state, "model_process", SimpleNamespace(is_alive=lambda: True), raising=False)
    monkeypatch.setattr(state, "worker_accepting", False, raising=False)
    monkeypatch.setattr(state, "worker_ready", False, raising=False)
    monkeypatch.setattr(state, "worker_generation", 4, raising=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app),
                                 base_url="http://testserver") as client:
        response = await client.get("/health")
        assert response.status_code == 503
        assert response.json()["worker"]["ready"] is False
        assert response.json()["worker"]["pid"] is None
        state.worker_ready = True
        state.worker_accepting = True
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["worker"]["generation"] == 4
        state.worker_accepting = False
        assert (await client.get("/health")).status_code == 503


def test_successful_diarization_reuses_cpu_pipeline_and_discards_errors(monkeypatch):
    loaded, moved, unloaded = [], [], []
    class Model:
        device = "cpu"
        def __init__(self): self.pipeline = object()
        @classmethod
        def from_pretrained(cls, device="cpu"):
            model = cls(); model.device = device; loaded.append(model); return model
        def to_device(self, device): self.device = device; moved.append(device); return self
        def park_cpu(self): self.to_device("cpu")
        def cpu_size_bytes(self): return 1024
        def unload(self): unloaded.append(self); self.pipeline = None

    monkeypatch.setattr(main, "Community1Model", Model)
    monkeypatch.setattr(main, "DIARIZATION_DEVICE", "cpu")
    if hasattr(main, "_clear_idle_diarizer"): main._clear_idle_diarizer()
    try:
        with main._diarization_model_session() as first:
            assert first.device == "cpu"
        with main._diarization_model_session() as second:
            assert second is first
        assert len(loaded) == 1
        with pytest.raises(ValueError):
            with main._diarization_model_session():
                raise ValueError("inference failed")
        with main._diarization_model_session() as recovered:
            assert recovered is not first
        assert first in unloaded
    finally:
        if hasattr(main, "_clear_idle_diarizer"): main._clear_idle_diarizer()


@pytest.mark.asyncio
async def test_supervisor_can_recover_after_a_second_worker_failure(monkeypatch):
    import queue
    import threading
    from runtime_support import RestartBudget

    class Channel(queue.Queue):
        def cancel_join_thread(self): pass
        def close(self): pass
    class Process:
        def __init__(self, response=None): self.alive = False; self.response = response
        def is_alive(self): return self.alive
        def start(self):
            self.alive = True
            self.response.put({"__worker_ready__": True})
        def join(self, timeout=0): pass
    class Context:
        def __init__(self): self.processes = []
        def Queue(self): return Channel()
        def Process(self, **kwargs):
            process = Process(kwargs["args"][1])
            self.processes.append(process)
            return process
    state, ctx = main.app.state, Context()
    request_queue, response_queue = Channel(), Channel()
    values = {
        "model_process": Process(), "request_queue": request_queue, "response_queue": response_queue,
        "stop_event": threading.Event(), "lifecycle_stop": asyncio.Event(), "model_usage": {},
        "job_cancel_flags": {}, "worker_generation": 0, "worker_restarts": 0,
        "worker_accepting": False, "worker_ready": False,
        "worker_restart_policy": RestartBudget(limit=2, window=.2, base_delay=0, max_delay=0),
        "response_listener_retire": asyncio.Event(),
    }
    for key, value in values.items(): monkeypatch.setattr(state, key, value, raising=False)
    for key in ("pending_results", "pending_streams", "pending_progress"):
        monkeypatch.setattr(main, key, {})
    listener = asyncio.create_task(main.response_listener(None, response_queue, state.response_listener_retire))
    monkeypatch.setattr(state, "response_listener_task", listener, raising=False)
    supervisor = asyncio.create_task(main._worker_supervisor(ctx))
    async def wait_ready(count):
        for _ in range(400):
            if len(ctx.processes) == count and main._worker_accepts_requests():
                return
            await asyncio.sleep(.01)
        assert len(ctx.processes) == count, "Recovery stopped permanently after the first restart"
        assert main._worker_accepts_requests()
    try:
        await wait_ready(1)
        ctx.processes[0].alive = False
        await wait_ready(2)
        assert state.worker_restarts == 2
        assert state.worker_ready
    finally:
        state.lifecycle_stop.set()
        supervisor.cancel()
        await asyncio.gather(supervisor, return_exceptions=True)
        current = state.response_listener_task
        current.cancel()
        await asyncio.gather(current, return_exceptions=True)


def test_restart_budget_cools_down_and_becomes_available_again():
    from runtime_support import RestartBudget
    now = [0.0]
    budget = RestartBudget(limit=2, window=10, base_delay=1, max_delay=4, clock=lambda: now[0])
    assert budget.delay() == 1
    budget.record()
    now[0] = 1
    assert budget.delay() == 2
    budget.record()
    assert budget.delay() == 9
    now[0] = 10
    assert budget.count() == 1
    assert budget.delay() == 2
    now[0] = 11
    assert budget.count() == 0
    assert budget.delay() == 1


def test_idle_model_cache_enforces_ttl_bytes_and_cpu_only():
    from runtime_support import IdleModelCache
    now, discarded = [0.0], []
    class Model:
        def __init__(self, size, device="cpu"): self.size = size; self.device = device
        def cpu_size_bytes(self): return self.size
        def unload(self): discarded.append(self)
    cache = IdleModelCache(ttl=5, max_bytes=100, clock=lambda: now[0])
    accepted = Model(80)
    assert cache.put(accepted)
    now[0] = 5
    cache.prune()
    assert cache.take() is None and discarded == [accepted]
    assert not cache.put(Model(101))
    assert not cache.put(Model(10, "cuda"))
    assert not cache.put(Model(0))
    assert cache.model is None


@pytest.mark.parametrize("error", [None, ValueError, asyncio.CancelledError])
def test_diarizer_parks_before_lease_release_and_discards_failed_inference(monkeypatch, error):
    from contextlib import contextmanager
    events = []
    class Model:
        device = "cuda"
        def park_cpu(self): self.device = "cpu"; events.append("park")
        def cpu_size_bytes(self): return 64
        def unload(self): self.device = "unloaded"; events.append("unload")
    model = Model()
    class Managed:
        def acquire(self): events.append("acquire")
        @contextmanager
        def inference(self):
            try:
                yield model
            finally:
                events.append("idle")
        def release(self):
            assert model.device in ("cpu", "unloaded")
            events.append("release")
    monkeypatch.setattr(main, "ManagedModel", Managed)
    monkeypatch.setattr(main, "_create_diarization_entry", Managed)
    main._clear_idle_diarizer()
    try:
        if error:
            with pytest.raises(error):
                with main._diarization_model_session():
                    raise error()
            assert events == ["acquire", "unload", "idle", "release"]
            assert main._diarization_cpu_cache.model is None
        else:
            with main._diarization_model_session(): pass
            assert events == ["acquire", "park", "idle", "release"]
            assert main._diarization_cpu_cache.model is model
    finally:
        main._clear_idle_diarizer()


@pytest.mark.asyncio
async def test_recovered_jobs_wait_for_current_worker_handshake_once(monkeypatch):
    import queue
    state, seen = main.app.state, []
    recovered = {"id": "recovered-job", "owner": "owner", "attempt": 2, "status": "queued"}
    for key, value in {
        "worker_generation": 3, "worker_ready": False, "worker_accepting": False,
        "deferred_web_jobs": {}, "lifecycle_stop": asyncio.Event(),
        "model_process": SimpleNamespace(is_alive=lambda: True),
    }.items():
        monkeypatch.setattr(state, key, value, raising=False)
    monkeypatch.setattr(main, "_launch_job", lambda owner, job: seen.append((owner, job)))
    monkeypatch.setattr(main, "_job_store", lambda: SimpleNamespace(get=lambda owner, job_id: recovered))
    main._defer_or_launch_web_job("owner", recovered)
    old = queue.Queue()
    old.put({"__worker_ready__": True})
    old.put(None)
    await asyncio.wait_for(main.response_listener(None, old, generation=2), 2)
    assert not state.worker_ready and not seen
    current = queue.Queue()
    current.put({"__worker_ready__": True})
    current.put({"__worker_ready__": True})
    current.put(None)
    await asyncio.wait_for(main.response_listener(None, current, generation=3), 2)
    assert state.worker_ready and state.worker_accepting
    assert seen == [("owner", recovered)]
    assert state.deferred_web_jobs == {}
    state.worker_ready = state.worker_accepting = False
    state.lifecycle_stop.set()
    shutdown = queue.Queue()
    shutdown.put({"__worker_ready__": True})
    shutdown.put(None)
    await asyncio.wait_for(main.response_listener(None, shutdown, generation=3), 2)
    assert not state.worker_ready and not state.worker_accepting


@pytest.mark.asyncio
async def test_worker_retry_waits_for_ready_and_skips_cancelled_jobs(monkeypatch, tmp_path):
    import queue
    state, decoded = main.app.state, []
    monkeypatch.setattr(main, "WEBUI_JOB_DIR", tmp_path / "jobs")
    monkeypatch.setattr(main, "_job_tasks", set())
    monkeypatch.setattr(main, "_job_tasks_by_attempt", {}, raising=False)
    for key, value in {
        "worker_generation": 3, "worker_ready": False, "worker_accepting": False,
        "model_process": SimpleNamespace(is_alive=lambda: True),
        "deferred_web_jobs": {}, "job_cancel_flags": {}, "lifecycle_stop": asyncio.Event(),
    }.items():
        monkeypatch.setattr(state, key, value, raising=False)
    store = main._job_store()

    def queued_job():
        job = store.create("owner", {"filename": "recording.wav", "model": "small", "diarize": False})
        (main._job_spool().root / job["id"]).write_bytes(b"audio")
        return store.enqueue("owner", job["id"], job["attempt"])

    async def transcribe(*args, **kwargs):
        if not main._worker_accepts_requests():
            return main.JSONResponse({"error": "Model worker unavailable", "code": "worker_stopped"}, status_code=503)
        decoded.append(kwargs["web_job_key"])
        return {"text": "recovered", "segments": []}

    monkeypatch.setattr(main, "_transcribe_impl", transcribe)
    defer = getattr(main, "_defer_or_launch_web_job", main._launch_job)
    recovered = queued_job()
    defer("owner", recovered)
    defer("owner", recovered)
    await asyncio.sleep(.01)
    assert store.get("owner", recovered["id"])["status"] == "queued"
    assert not decoded

    cancelled = queued_job()
    defer("owner", cancelled)
    store.request_cancel("owner", cancelled["id"], cancelled["attempt"])
    old = queue.Queue()
    old.put({"__worker_ready__": True})
    old.put(None)
    await asyncio.wait_for(main.response_listener(None, old, generation=2), 2)
    assert not state.worker_ready and not decoded
    current = queue.Queue()
    current.put({"__worker_ready__": True})
    current.put({"__worker_ready__": True})
    current.put(None)
    await asyncio.wait_for(main.response_listener(None, current, generation=3), 2)
    await asyncio.wait_for(asyncio.gather(*list(main._job_tasks)), 2)
    assert decoded == [(recovered["id"], recovered["attempt"])]
    assert store.get("owner", recovered["id"])["status"] == "done"
    assert store.get("owner", cancelled["id"])["status"] == "cancelled"
    assert state.deferred_web_jobs == {}

    immediate = queued_job()
    defer("owner", immediate)
    defer("owner", immediate)
    await asyncio.wait_for(asyncio.gather(*list(main._job_tasks)), 2)
    assert decoded.count((immediate["id"], immediate["attempt"])) == 1
    assert store.get("owner", immediate["id"])["status"] == "done"
