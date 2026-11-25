"""Recovery regressions without GPU/model loading.

Hypotheses: repeated loss at one durable boundary currently retries forever;
an already-ready native worker can remain alive and hung forever.
"""
import asyncio
import threading
from types import SimpleNamespace

import pytest

import main
from webui_jobs import JobStore


@pytest.mark.asyncio
async def test_repeated_active_job_crash_is_quarantined_with_input_retained(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs.sqlite")
    settings = {"filename": "source.wav", "model": "gigaam-v3", "diarize": False}
    job = store.create("owner", settings)
    source = tmp_path / job["id"]
    source.write_bytes(b"saved audio")
    store.enqueue("owner", job["id"], 1)
    monkeypatch.setattr(main, "_job_store", lambda: store)
    monkeypatch.setattr(main, "_job_spool", lambda: SimpleNamespace(root=tmp_path))
    monkeypatch.setattr(main, "_notify_job", lambda job_id: None)
    monkeypatch.setattr(main, "_defer_or_launch_web_job", lambda owner, job: None)
    monkeypatch.setattr(main.app.state, "job_cancel_flags", {}, raising=False)
    monkeypatch.setenv("WHISPER_JOB_CRASH_LIMIT", "3")

    async def lost_worker(*args, **kwargs):
        job_key = kwargs["web_job_key"]
        return main.JSONResponse(status_code=503, content={
            "error": "Model worker stopped before responding", "code": "worker_stopped",
            "worker_failure": {"job_key": list(job_key), "phase": "inference",
                               "unit": "chunk:7", "checkpoint": 7, "reason": "worker_exited"},
        })
    monkeypatch.setattr(main, "_transcribe_impl", lost_worker)

    for _ in range(3):
        job = store.get("owner", job["id"])
        await main._execute_web_job("owner", job["id"], job["attempt"])
        # Reopening the store must not reset the failure budget.
        store = JobStore(tmp_path / "jobs.sqlite")
    job = store.get("owner", job["id"])
    assert job["status"] == "error", "same failing job must stop retrying after the bounded budget"
    assert source.read_bytes() == b"saved audio"
    assert job["recovery"]["quarantined"] is True
    assert job["recovery"]["failures"] == 3


@pytest.mark.asyncio
async def test_alive_worker_stuck_in_inference_is_terminated(monkeypatch):
    class Process:
        alive = True
        terminated = False
        def is_alive(self): return self.alive
        def terminate(self): self.terminated = True; self.alive = False
        def join(self, timeout=0): pass
    process = Process()
    state = main.app.state
    values = {
        "model_process": process, "worker_ready": True, "worker_accepting": True,
        "worker_generation": 0, "worker_restarts": 0, "worker_start_deadline": None,
        "worker_restart_policy": None, "lifecycle_stop": asyncio.Event(),
        "worker_execution_state": {"snapshot": {"generation": 0, "request_id": "running",
            "job_key": ["job", 1], "phase": "inference", "unit": "chunk:0",
            "progress": 0, "checkpoint": 0}},
        "response_queue": SimpleNamespace(), "response_listener_retire": asyncio.Event(),
    }
    for key, value in values.items(): monkeypatch.setattr(state, key, value, raising=False)
    monkeypatch.setenv("WHISPER_WATCHDOG_INFERENCE_SECONDS", "0.02")
    supervisor = asyncio.create_task(main._worker_supervisor(None))
    try:
        await asyncio.sleep(.65)
        assert process.terminated, "alive native worker exceeded its phase progress deadline"
        assert state.worker_accepting is False
    finally:
        state.lifecycle_stop.set()
        supervisor.cancel()
        await asyncio.gather(supervisor, return_exceptions=True)

def _queued_job(store, owner="owner"):
    settings = {"filename": "source.wav", "model": "gigaam-v3", "diarize": False}
    job = store.create(owner, settings)
    return store.enqueue(owner, job["id"], job["attempt"])


def _failure(job, checkpoint=0):
    return {"job_key": [job["id"], job["attempt"]], "checkpoint": checkpoint,
            "phase": "inference", "unit": f"chunk:{checkpoint}", "reason": "worker_exited"}


def test_only_new_durable_progress_resets_crash_budget(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    job = _queued_job(store)
    for checkpoint, expected_count in [(7, 1), (7, 2), (8, 1), (1, 2)]:
        store.start("owner", job["id"], job["attempt"])
        job = store.interrupt_for_retry("owner", job["id"], job["attempt"], failure=_failure(job, checkpoint))
        assert job["status"] == "queued"
        assert job["recovery"]["failures"] == expected_count
        assert job["recovery"]["checkpoint"] == (8 if checkpoint in (8, 1) else 7)
    store.start("owner", job["id"], job["attempt"])
    job = store.interrupt_for_retry("owner", job["id"], job["attempt"], failure=_failure(job, 8))
    assert job["status"] == "error"
    assert store.recover_interrupted() == []
    assert store.retry_if_diarization_changed("owner", job["id"], job["attempt"]) is None
    resumed = store.retry("owner", job["id"], job["diarization"])
    assert resumed["status"] == "queued"
    assert "recovery" not in resumed
    store.start("owner", resumed["id"], resumed["attempt"])
    retry = store.interrupt_for_retry("owner", resumed["id"], resumed["attempt"],
                                      failure=_failure(resumed, 8))
    assert retry["recovery"]["failures"] == 1


def test_pending_neighbour_retries_have_no_failure_penalty_and_cancel_wins(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    neighbour = _queued_job(store)
    culprit = _queued_job(store)
    for _ in range(5):
        # Even a wrongly addressed fault is not charged to this request.
        neighbour = store.interrupt_for_retry("owner", neighbour["id"], neighbour["attempt"],
                                              failure=_failure(culprit, 2))
    assert neighbour["status"] == "queued"
    assert "recovery" not in neighbour
    store.start("owner", culprit["id"], culprit["attempt"])
    store.request_cancel("owner", culprit["id"], culprit["attempt"])
    cancelled = store.interrupt_for_retry("owner", culprit["id"], culprit["attempt"],
                                          failure=_failure(culprit), max_failures=1)
    assert cancelled["status"] == "cancelled"
    assert "recovery" not in cancelled


def test_quarantine_cannot_be_revived_by_automatic_settings_retry(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    job = _queued_job(store)
    store.start("owner", job["id"], job["attempt"])
    settings = dict(job["diarization"], desired_diarize=True)
    store.set_diarization("owner", job["id"], job["attempt"], settings)
    job = store.interrupt_for_retry("owner", job["id"], job["attempt"],
                                    failure=_failure(job), max_failures=1)
    assert job["status"] == "error"
    assert store.retry_if_diarization_changed("owner", job["id"], job["attempt"]) is None
    assert store.recover_interrupted() == []
    # The owner can explicitly apply the changed setting to release quarantine.
    retry = store.set_diarization("owner", job["id"], job["attempt"], settings)
    assert retry["status"] == "queued"
    assert "recovery" not in retry


def test_phase_watchdog_ignores_heartbeats_and_old_generation_but_tracks_real_work():
    from runtime_support import PhaseWatchdog, WorkerExecution
    now = [0.0]
    shared = {"snapshot": {"generation": 2, "phase": "idle"}}
    execution = WorkerExecution(shared)
    watchdog = PhaseWatchdog({"inference": 10}, clock=lambda: now[0])
    execution.enter("request", ("job", 4), "inference", unit="chunk:7", progress=0, checkpoint=7)
    assert watchdog.observe(shared["snapshot"], 2) is None
    now[0] = 9
    execution.advance(0)
    assert watchdog.observe(dict(shared["snapshot"], heartbeat=99), 2) is None
    execution.advance(1)
    assert watchdog.observe(shared["snapshot"], 2) is None
    now[0] = 18
    # A stale idle snapshot cannot clear a current-generation deadline.
    assert watchdog.observe({"generation": 1, "phase": "idle"}, 2) is None
    assert watchdog.observe(shared["snapshot"], 2) is None
    now[0] = 19
    assert watchdog.observe(shared["snapshot"], 2)["reason"] == "phase_timeout"
    execution.idle()
    assert watchdog.observe(shared["snapshot"], 2) is None
    assert watchdog.view()["phase"] == "idle"


def test_phase_budgets_allow_cold_cpu_load_and_long_diarization():
    from runtime_support import PhaseWatchdog, WorkerExecution
    now = [0.0]
    shared = {"snapshot": {"generation": 0, "phase": "idle"}}
    execution = WorkerExecution(shared)
    watchdog = PhaseWatchdog(clock=lambda: now[0])
    execution.enter("request", ("job", 1), "preparing_cpu")
    watchdog.observe(shared["snapshot"], 0)
    now[0] = 600
    assert watchdog.observe(shared["snapshot"], 0) is None
    execution.phase("waiting_gpu")
    watchdog.observe(shared["snapshot"], 0)
    now[0] = 1200
    assert watchdog.observe(shared["snapshot"], 0) is None
    execution.phase("loading_gpu")
    watchdog.observe(shared["snapshot"], 0)
    now[0] = 1300
    assert watchdog.observe(shared["snapshot"], 0) is None
    execution.phase("diarization")
    watchdog.observe(shared["snapshot"], 0)
    now[0] = 4800
    assert watchdog.observe(shared["snapshot"], 0) is None
    now[0] = 4900
    assert watchdog.observe(shared["snapshot"], 0)["phase"] == "diarization"


@pytest.mark.asyncio
async def test_supervisor_attributes_crash_only_to_running_request(monkeypatch):
    import queue
    from runtime_support import RestartBudget

    class Channel(queue.Queue):
        def cancel_join_thread(self): pass
        def close(self): pass
    class Process:
        def __init__(self, alive=False): self.alive = alive
        def is_alive(self): return self.alive
        def join(self, timeout=0): pass
        def start(self): self.alive = True; main.app.state.lifecycle_stop.set()
    class Context:
        def Queue(self): return Channel()
        def Process(self, **kwargs): return Process()

    state = main.app.state
    old_listener = asyncio.create_task(asyncio.sleep(0))
    values = {
        "model_process": Process(), "worker_ready": False, "worker_accepting": False,
        "worker_generation": 4, "worker_restarts": 0, "lifecycle_stop": asyncio.Event(),
        "stop_event": threading.Event(), "model_usage": {}, "job_cancel_flags": {},
        "worker_restart_policy": RestartBudget(base_delay=0, max_delay=0),
        "worker_execution_state": {"snapshot": {"generation": 4, "request_id": "active",
            "job_key": ["culprit", 3], "phase": "inference", "unit": "chunk:7",
            "progress": 7, "checkpoint": 7}},
        "request_queue": Channel(), "response_queue": Channel(),
        "response_listener_retire": asyncio.Event(), "response_listener_task": old_listener,
        "worker_watchdog": None,
    }
    for key, value in values.items(): monkeypatch.setattr(state, key, value, raising=False)
    active, pending = asyncio.get_running_loop().create_future(), asyncio.get_running_loop().create_future()
    monkeypatch.setattr(main, "pending_results", {"active": active, "pending": pending})
    monkeypatch.setattr(main, "pending_progress", {})
    monkeypatch.setattr(main, "pending_streams", {})
    try:
        await asyncio.wait_for(main._worker_supervisor(Context()), 4)
        active_error, pending_error = await asyncio.gather(active, pending, return_exceptions=True)
        assert isinstance(active_error, main.WorkerStopped)
        assert active_error.failure["job_key"] == ["culprit", 3]
        assert active_error.failure["checkpoint"] == 7
        assert isinstance(pending_error, main.WorkerStopped)
        assert pending_error.failure is None
        assert state.worker_generation == 5
        assert state.worker_execution_state["snapshot"] == {"generation": 5, "phase": "idle"}
    finally:
        listener = state.response_listener_task
        listener.cancel()
        await asyncio.gather(listener, return_exceptions=True)


@pytest.mark.parametrize("point", ["after_cpu", "before_gpu", "after_gpu"])
def test_diarizer_cancel_discards_prepared_weights_without_inference(monkeypatch, point):
    from contextlib import contextmanager
    events = []
    class Model:
        device = "cpu"
        def to_device(self, device): self.device = device; events.append("promote"); return self
        def unload(self): self.device = "unloaded"; events.append("unload")
    model = Model()
    class Managed:
        def __init__(self, **kwargs): self.loader = kwargs["loader"]
        def acquire(self): events.append("acquire"); self.model = self.loader()
        @contextmanager
        def inference(self): events.append("inference"); yield self.model
        def release(self): events.append("release")
    def load(device): events.append("cpu"); return model
    checks = [0]
    def cancel():
        checks[0] += 1
        if ((point == "after_cpu" and checks[0] == 2)
                or (point == "before_gpu" and checks[0] == 3)
                or (point == "after_gpu" and "promote" in events)):
            raise main.WebJobCancelled()
    monkeypatch.setattr(main, "ManagedModel", Managed)
    monkeypatch.setattr(main, "DIARIZATION_DEVICE", "cuda")
    monkeypatch.setattr(main, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setattr(main, "_load_diarization_model", load)
    phases = []
    with pytest.raises(main.WebJobCancelled):
        with main._diarization_model_session(cancel_check=cancel, phase_hook=phases.append):
            pytest.fail("cancelled preparation must never reach inference")
    assert events.count("unload") == 1
    assert "inference" not in events
    if point == "after_cpu":
        assert events == ["cpu", "unload"]
    else:
        assert events[-1] == "release"
    assert phases[0] == "preparing_cpu"

def test_idempotent_setting_change_keeps_quarantine_reason_and_partial_result(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    job = _queued_job(store)
    store.start("owner", job["id"], job["attempt"])
    store.append_progress("owner", job["id"], job["attempt"],
                          segment={"start": 0, "end": 1, "text": "saved prefix"})
    stopped = store.interrupt_for_retry("owner", job["id"], job["attempt"],
                                        failure=_failure(job), max_failures=1)
    settings = dict(stopped["diarization"], desired_diarize=False)
    unchanged = store.set_diarization("owner", job["id"], job["attempt"], settings)
    assert unchanged["status"] == "error"
    assert unchanged["error"] == stopped["error"]
    assert unchanged["result"] == stopped["result"]
    assert unchanged["recovery"] == stopped["recovery"]
