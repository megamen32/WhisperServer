"""Real IPC regressions for a worker killed while writing a queue frame."""
import asyncio
import multiprocessing as mp
import os
import struct
import threading

import pytest

import main
from runtime_support import RestartBudget


def _partial_frame_child(response):
    os.write(response._writer.fileno(), struct.pack("!i", 100_000) + b"partial")
    os._exit(0)


def _ready_child(response, stop):
    response.put({"__worker_ready__": True})
    stop.wait(15)


@pytest.mark.asyncio
async def test_partial_ipc_frame_retires_reader_and_restarts_worker(monkeypatch):
    ctx = mp.get_context("spawn")
    request, response, stop = ctx.Queue(), ctx.Queue(), ctx.Event()
    initial = ctx.Process(target=_partial_frame_child, args=(response,))
    replacements, channels, readers = [], [request, response], []
    original_thread = threading.Thread

    def track_thread(*args, **kwargs):
        thread = original_thread(*args, **kwargs)
        if str(kwargs.get("name", "")).startswith("worker-response-reader"):
            readers.append(thread)
        return thread

    class Context:
        def Queue(self):
            channel = ctx.Queue()
            channels.append(channel)
            return channel

        def Process(self, **kwargs):
            process = ctx.Process(target=_ready_child, args=(kwargs["args"][1], stop))
            replacements.append(process)
            return process

    monkeypatch.setattr(main.threading, "Thread", track_thread)
    initial.start()
    getattr(main, "_close_parent_response_writer", lambda channel: None)(response)
    state = main.app.state
    for key, value in {
        "model_process": initial, "request_queue": request, "response_queue": response,
        "stop_event": stop, "lifecycle_stop": asyncio.Event(), "model_usage": {},
        "job_cancel_flags": {}, "worker_execution_state": {},
        "worker_generation": 0, "worker_restarts": 0,
        "worker_accepting": True, "worker_ready": True, "worker_start_deadline": None,
        "worker_restart_policy": RestartBudget(limit=2, window=1, base_delay=0, max_delay=0),
        "response_listener_retire": asyncio.Event(), "deferred_web_jobs": {},
    }.items():
        monkeypatch.setattr(state, key, value, raising=False)
    pending = asyncio.get_running_loop().create_future()
    monkeypatch.setattr(main, "pending_results", {"lost-request": pending})
    monkeypatch.setattr(main, "pending_streams", {})
    monkeypatch.setattr(main, "pending_progress", {})
    listener = asyncio.create_task(main.response_listener(
        stop, response, state.response_listener_retire, generation=0))
    monkeypatch.setattr(state, "response_listener_task", listener, raising=False)
    supervisor = asyncio.create_task(main._worker_supervisor(Context()))
    try:
        done, _ = await asyncio.wait({pending}, timeout=8)
        assert pending in done, f"Partial IPC frame halted recovery: {state.worker_failure_reason}"
        with pytest.raises(main.WorkerStopped):
            pending.result()
        for _ in range(400):
            if replacements and main._worker_accepts_requests():
                break
            await asyncio.sleep(.01)
        assert main._worker_accepts_requests()
        assert state.worker_generation == 1 and state.worker_restarts == 1
        assert response._writer.closed and state.response_queue._writer.closed
        assert readers and not readers[0].is_alive()
        assert not main.pending_results
    finally:
        state.lifecycle_stop.set()
        stop.set()
        supervisor.cancel()
        await asyncio.gather(supervisor, return_exceptions=True)
        for task in {listener, state.response_listener_task}:
            task.cancel()
        await asyncio.gather(listener, state.response_listener_task, return_exceptions=True)
        for process in [initial, *replacements]:
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
        for channel in channels:
            # This also unblocks the intentionally broken baseline during RED.
            channel._writer.close()
            channel.cancel_join_thread()
            channel.close()
        for reader in readers:
            reader.join(timeout=1)
        if not pending.done():
            pending.cancel()
        elif not pending.cancelled():
            pending.exception()


@pytest.mark.asyncio
async def test_retired_generation_cannot_dispatch_a_late_result(monkeypatch):
    import queue
    monkeypatch.setattr(main.app.state, "worker_generation", 2, raising=False)
    pending = asyncio.get_running_loop().create_future()
    monkeypatch.setattr(main, "pending_results", {"current-request": pending})
    monkeypatch.setattr(main, "pending_streams", {})
    monkeypatch.setattr(main, "pending_progress", {})
    old = queue.Queue()
    old.put({"request_id": "current-request", "result": {"text": "obsolete"}})
    old.put(None)
    await asyncio.wait_for(main.response_listener(None, old, generation=1), 2)
    assert not pending.done(), "Retired generation overwrote a current request"
    current = queue.Queue()
    current.put({"request_id": "current-request", "result": {"text": "current"}})
    current.put(None)
    await asyncio.wait_for(main.response_listener(None, current, generation=2), 2)
    assert await pending == {"text": "current"}
