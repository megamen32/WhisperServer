"""Real SQLite contention must not discard worker recovery or block the API loop."""
import asyncio
import sqlite3
import threading
import time

import pytest

import main
from webui_jobs import JobStore
from webui_uploads import UploadSpool


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["queued", "quarantined", "cancelled", "shutdown_retry", "shutdown_failure", "stale_generation"])
async def test_worker_retry_survives_sqlite_writer_without_blocking_loop(tmp_path, monkeypatch, outcome):
    monkeypatch.setenv("WHISPER_JOB_CRASH_LIMIT", "1" if outcome == "quarantined" else "3")
    store = JobStore(tmp_path / "jobs.sqlite", timeout=.08)
    spool = UploadSpool(tmp_path / "uploads", 1024, 16, 4096)
    job = store.create("owner", {"filename": "fixture.wav", "size": 3,
        "model": "small", "diarize": False, "desired_diarize": False})
    job_id = job["id"]
    store.enqueue("owner", job_id, 1)
    spool.append(job_id, 0, b"abc", 3)
    monkeypatch.setattr(main, "_job_store", lambda: store)
    monkeypatch.setattr(main, "_job_spool", lambda: spool)
    monkeypatch.setattr(main.app.state, "worker_accepting", False, raising=False)
    monkeypatch.setattr(main.app.state, "worker_ready", False, raising=False)
    monkeypatch.setattr(main.app.state, "deferred_web_jobs", {}, raising=False)
    monkeypatch.setattr(main.app.state, "job_cancel_flags", {}, raising=False)
    held, release = threading.Event(), threading.Event()
    worker_returned = asyncio.Event()
    thread = None
    fault = {"job_key": [job_id, 1], "checkpoint": 2, "phase": "asr",
             "unit": "chunk-3", "reason": "worker_exited"}

    def hold_writer():
        with sqlite3.connect(store.path, isolation_level=None) as db:
            db.execute("BEGIN IMMEDIATE")
            held.set()
            assert release.wait(3)
            db.rollback()

    async def stopped_worker(*args, **kwargs):
        nonlocal thread
        if outcome == "cancelled":
            store.request_cancel("owner", job_id, 1)
        thread = threading.Thread(target=hold_writer)
        thread.start()
        assert await asyncio.to_thread(held.wait, 1)
        worker_returned.set()
        if outcome == "shutdown_failure":
            raise RuntimeError("fixture worker failure")
        return main.JSONResponse({"code": "worker_stopped", "worker_failure": fault}, status_code=503)

    monkeypatch.setattr(main, "_transcribe_impl", stopped_worker)
    loop_delays = []
    async def heartbeat():
        while True:
            started = time.monotonic()
            await asyncio.sleep(.01)
            loop_delays.append(time.monotonic() - started)

    heart = asyncio.create_task(heartbeat())
    if outcome == "stale_generation":
        thread = threading.Thread(target=hold_writer)
        thread.start()
        assert await asyncio.to_thread(held.wait, 1)
        worker_returned.set()
        execution = asyncio.create_task(main._run_web_job("owner", job_id, 1, generation=-1))
    else:
        execution = asyncio.create_task(main._execute_web_job("owner", job_id, 1))
    errors = []
    try:
        await asyncio.wait_for(worker_returned.wait(), 1)
        await asyncio.sleep(.12)
        still_waiting = not execution.done()
        if outcome.startswith("shutdown"):
            execution.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(execution), .15)
    finally:
        release.set()
        if thread:
            await asyncio.to_thread(thread.join, 1)
        result = await asyncio.gather(execution, return_exceptions=True)
        errors = [type(value).__name__ for value in result if isinstance(value, BaseException)]
        heart.cancel()
        await asyncio.gather(heart, return_exceptions=True)
    saved = store.get("owner", job_id)
    if outcome.startswith("shutdown"):
        assert errors == ["CancelledError"]
        assert saved["status"] == "processing" and saved["attempt"] == 1
        assert "recovery" not in saved
        assert not main.app.state.deferred_web_jobs
        assert spool.offset(job_id) == 3
        return
    assert not errors, errors
    assert loop_delays and max(loop_delays) < .18, "SQLite contention blocked the event loop"
    assert still_waiting, "Recovery intent ended before the writer released its lock"
    if outcome == "cancelled":
        assert saved["status"] == "cancelled" and saved["attempt"] == 1
        assert "recovery" not in saved
        assert not main.app.state.deferred_web_jobs
        assert spool.offset(job_id) == 0
        return
    if outcome == "stale_generation":
        assert saved["status"] == "queued" and saved["attempt"] == 2
        assert "recovery" not in saved
        assert ("owner", job_id, 2) in main.app.state.deferred_web_jobs
        assert spool.offset(job_id) == 3
        return
    quarantined = outcome == "quarantined"
    assert saved["status"] == ("error" if quarantined else "queued")
    assert saved["attempt"] == (1 if quarantined else 2)
    assert saved["recovery"] == {"failures": 1, "checkpoint": 2, "phase": "asr",
        "unit": "chunk-3", "reason": "worker_exited", "quarantined": quarantined}
    assert (("owner", job_id, 2) in main.app.state.deferred_web_jobs) == (not quarantined)
    assert spool.offset(job_id) == 3


@pytest.mark.asyncio
async def test_non_contention_sqlite_error_is_not_retried():
    calls = []
    def broken_disk():
        calls.append(1)
        raise sqlite3.OperationalError("disk I/O error")
    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        await main._web_job_store_operation(broken_disk)
    assert len(calls) == 1
