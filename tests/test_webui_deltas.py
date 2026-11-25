"""Focused regressions for durable deltas, nonblocking readers and independent uploads."""
import json
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import pytest

from webui_jobs import JobStore
from webui_uploads import UploadSpool


def test_partial_storage_does_not_rewrite_the_transcript_prefix(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    job = store.create("owner", {})
    job_id = job["id"]
    store.enqueue("owner", job_id, 1)
    store.start("owner", job_id, 1)
    segments = []
    for index in range(25):
        segments.append({"start": index, "end": index + 1, "text": "segment " + str(index) + " " + "x" * 80})
        store.update_progress("owner", job_id, 1, {
            "text": " ".join(s["text"] for s in segments), "segments": list(segments),
            "progress": {"phase": "transcribing", "processed_seconds": index + 1, "total_seconds": 30}})
    with sqlite3.connect(store.path) as db:
        hot_payload = db.execute("SELECT result FROM jobs WHERE id=?", (job_id,)).fetchone()[0]
    assert len(hot_payload.encode()) < 500, "each update still stores the whole transcript prefix"
    assert store.get("owner", job_id)["result"]["segments"] == segments


def test_read_snapshot_does_not_compete_for_writer_lock(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite", timeout=.1)
    job = store.create("owner", {})
    with sqlite3.connect(store.path, isolation_level=None) as writer:
        writer.execute("BEGIN IMMEDIATE")
        try:
            assert store.get("owner", job["id"])["id"] == job["id"]
        finally:
            writer.rollback()


def test_slow_fsync_of_one_upload_does_not_block_another(tmp_path, monkeypatch):
    spool = UploadSpool(tmp_path, max_file_bytes=4, max_chunk_bytes=4, max_total_bytes=8)
    entered, release = threading.Event(), threading.Event()
    original = os.fsync

    def slow(fd):
        try:
            filename = os.readlink("/proc/self/fd/" + str(fd))
        except OSError:
            filename = ""
        if filename == str(tmp_path / "slow"):
            entered.set()
            assert release.wait(3), "test did not release slow fsync"
        return original(fd)

    monkeypatch.setattr(os, "fsync", slow)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(spool.append, "slow", 0, b"aaaa", 4)
        assert entered.wait(1)
        assert spool.offset('slow') == 0
        second = pool.submit(spool.append, "fast", 0, b"bbbb", 4)
        try:
            assert second.result(timeout=.5) == 4
        except TimeoutError:
            pytest.fail("another upload waited behind unrelated file fsync")
        finally:
            release.set()
            first.result(timeout=3)
            second.result(timeout=3)

def test_legacy_database_keeps_text_summaries_and_cursor_on_upgrade(tmp_path):
    """A rolling upgrade reads the old schema without rewriting existing results."""
    path = tmp_path / 'legacy.sqlite'
    pieces = [{'start': 0, 'end': 1, 'text': 'old text'}]
    result = {'text': 'old text', 'segments': pieces, 'progress': {'phase': 'transcribing'}}
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE jobs (
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, status TEXT NOT NULL,
                attempt INTEGER NOT NULL, diarization TEXT NOT NULL,
                result TEXT, error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE summaries (
                job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                name TEXT NOT NULL, value TEXT NOT NULL, PRIMARY KEY(job_id,name));
        """)
        for job_id, status in [('done', 'done'), ('partial', 'processing')]:
            db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,NULL,?,?)',
                       (job_id, 'owner', status, 1, '{"diarize":false}', json.dumps(result), 'before', 'before'))
            db.execute('INSERT INTO summaries VALUES (?,?,?)', (job_id, 'summary', '{"text":"keep summary"}'))
    store = JobStore(path)
    for job_id in ('done', 'partial'):
        assert store.get('owner', job_id)['result'] == result
        assert store.get('owner', job_id)['summaries'] == {'summary': {'text': 'keep summary'}}
        initial = store.stream_delta('owner', job_id)
        assert initial['segments'] == pieces
        assert initial['cursor'] == '1:0:1'
        assert store.stream_delta('owner', job_id, '1:0:1')['segments'] == []
    next_piece = {'start': 1, 'end': 2, 'text': 'new text'}
    store.append_progress('owner', 'partial', 1, segment=next_piece, progress={'phase': 'transcribing'})
    assert store.get('owner', 'partial')['result']['segments'] == pieces + [next_piece]
    assert store.get('owner', 'partial')['summaries'] == {'summary': {'text': 'keep summary'}}
    with sqlite3.connect(path) as db:
        assert json.loads(db.execute("SELECT result FROM jobs WHERE id='done'").fetchone()[0]) == result
    retry = store.retry('owner', 'done', {'diarize': False})
    assert retry['attempt'] == 2 and retry['summaries'] == {}
    assert store.stream_delta('owner', 'done', '1:0:1')['segments'] == []
    store.start('owner', 'done', 2)
    store.append_progress('owner', 'done', 2, segment=next_piece)
    assert store.get('owner', 'done')['result']['segments'] == [next_piece]

def test_worker_interruption_requeues_only_the_current_attempt(tmp_path):
    from webui_jobs import InvalidTransition
    store = JobStore(tmp_path / 'retry.sqlite')
    job_id = store.create('owner', {'diarize': False})['id']
    store.enqueue('owner', job_id, 1)
    store.start('owner', job_id, 1)
    store.append_progress('owner', job_id, 1, segment={'text': 'old', 'end': 1})
    next_job = store.interrupt_for_retry('owner', job_id, 1)
    assert next_job['status'] == 'queued' and next_job['attempt'] == 2
    assert next_job['result'] is None
    assert store.stream_delta('owner', job_id, '1:0:1')['segments'] == []
    with pytest.raises(InvalidTransition):
        store.interrupt_for_retry('owner', job_id, 1)
    store.start('owner', job_id, 2)
    store.request_cancel('owner', job_id, 2)
    cancelled = store.interrupt_for_retry('owner', job_id, 2)
    assert cancelled['status'] == 'cancelled' and cancelled['attempt'] == 2


def test_parallel_uploads_reserve_the_declared_capacity_atomically(tmp_path):
    spool = UploadSpool(tmp_path, max_file_bytes=4, max_chunk_bytes=4, max_total_bytes=6)
    ready = threading.Barrier(2)
    def upload(job_id):
        ready.wait()
        try:
            return spool.append(job_id, 0, b'ab', 4)
        except ValueError as error:
            return str(error)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(upload, ['first', 'second']))
    assert sorted(result for result in results if isinstance(result, int)) == [2]
    assert sum(isinstance(result, str) and 'limit' in result for result in results) == 1
    assert spool.usage() == {'reserved_bytes': 4, 'committed_bytes': 2, 'limit_bytes': 6}


def test_existing_audio_is_accounted_without_expiring_or_changing_it(tmp_path):
    original = tmp_path / 'old-job'
    original.write_bytes(b'existing audio')
    spool = UploadSpool(tmp_path, max_file_bytes=100, max_chunk_bytes=4, max_total_bytes=100)
    assert spool.usage()['reserved_bytes'] == len(b'existing audio')
    assert spool.offset('old-job') == len(b'existing audio')
    assert original.read_bytes() == b'existing audio'


def test_retry_recovers_a_partial_uncommitted_disk_suffix(tmp_path):
    spool = UploadSpool(tmp_path, max_file_bytes=8, max_chunk_bytes=5, max_total_bytes=8)
    spool.append('recover', 0, b'abc', 8)
    # Process crash after some new bytes reached the file, before fsync/ledger commit.
    with (tmp_path / 'recover').open('ab') as output:
        output.write(b'def')
    assert spool.offset('recover') == 3
    assert spool.append('recover', 3, b'defgh', 8) == 8
    assert (tmp_path / 'recover').read_bytes() == b'abcdefgh'
    assert spool.offset('recover') == 8
    # Confirmed bytes remain immutable on a mismatching retry.
    from webui_uploads import UploadConflict
    with pytest.raises(UploadConflict):
        spool.append('recover', 3, b'DEFGH', 8)
    assert (tmp_path / 'recover').read_bytes() == b'abcdefgh'
