"""A snapshot reader must not stall streamed writes or pin JSON decoding in WAL."""
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor


import webui_jobs
from webui_jobs import JobStore


def active_job(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite", timeout=.2)
    job = store.create("owner", {})
    job = store.enqueue("owner", job["id"], 1)
    store.start("owner", job["id"], 1)
    store.append_progress("owner", job["id"], 1, segment={"text": "first", "start": 0, "end": 1})
    return store, job["id"]


def test_snapshot_reader_does_not_block_stream_commit(tmp_path):
    store, job_id = active_job(tmp_path)
    reader = sqlite3.connect(store.path, isolation_level=None)
    try:
        reader.execute("BEGIN")
        reader.execute("SELECT * FROM jobs").fetchall()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(store.append_progress, "owner", job_id, 1,
                                 segment={"text": "second", "start": 1, "end": 2})
            try:
                result = future.result(timeout=1)
            finally:
                reader.rollback()
        assert result["segment_count"] == 2
    finally:
        reader.close()


def test_snapshot_closes_transaction_before_decoding_large_text(tmp_path, monkeypatch):
    store, job_id = active_job(tmp_path)
    # Keep one connection open so WAL checkpoint behaviour is observable.
    keeper = sqlite3.connect(store.path, isolation_level=None, timeout=.2)
    paused, resume = threading.Event(), threading.Event()
    decode = webui_jobs.json.loads

    def slow_decode(value, *args, **kwargs):
        if threading.current_thread().name.startswith("snapshot") and '"text":"first"' in str(value):
            paused.set()
            assert resume.wait(3)
        return decode(value, *args, **kwargs)

    monkeypatch.setattr(webui_jobs.json, "loads", slow_decode)
    try:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="snapshot") as pool:
            future = pool.submit(store.get, "owner", job_id)
            try:
                assert paused.wait(2)
                store.append_progress("owner", job_id, 1,
                                      segment={"text": "second", "start": 1, "end": 2})
                checkpoint = keeper.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                assert checkpoint[0] == 0, "JSON decoding holds an old WAL snapshot"
            finally:
                resume.set()
            snapshot = future.result(timeout=3)
        assert [part["text"] for part in snapshot["result"]["segments"]] == ["first"]
        assert store.get("owner", job_id)["segment_count"] == 2
    finally:
        keeper.close()
