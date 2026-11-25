"""Durable resumable uploads with short quota reservations and per-file locks."""
import fcntl
import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path


class UploadConflict(ValueError):
    """The offset, replay bytes or declared size differ from the stored upload."""


class UploadSpool:
    def __init__(self, root, max_file_bytes, max_chunk_bytes, max_total_bytes):
        self.root = Path(root)
        self.max_file_bytes = max_file_bytes
        self.max_chunk_bytes = max_chunk_bytes
        self.max_total_bytes = max_total_bytes
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._locks = self.root / ".locks"
        self._locks.mkdir(mode=0o700, exist_ok=True)
        self._ledger = self.root / ".quota.sqlite"
        self._ledger_ready = False

    def _path(self, job_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_id):
            raise ValueError("Invalid job ID")
        return self.root / job_id

    def _ensure_ledger(self):
        if self._ledger_ready:
            return
        # This scan runs once when upgrading an existing spool. It never deletes audio.
        with (self.root / ".quota-init.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with sqlite3.connect(self._ledger, timeout=5) as db:
                db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                db.execute("""CREATE TABLE IF NOT EXISTS reservations (
                    job_id TEXT PRIMARY KEY, reserved INTEGER NOT NULL,
                    total INTEGER, committed INTEGER NOT NULL DEFAULT 0)""")
                if not db.execute("SELECT 1 FROM metadata WHERE key='seeded'").fetchone():
                    for path in self.root.iterdir():
                        if path.is_file() and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", path.name):
                            size = path.stat().st_size
                            db.execute("INSERT OR IGNORE INTO reservations VALUES (?, ?, NULL, ?)",
                                       (path.name, size, size))
                    db.execute("INSERT INTO metadata VALUES ('seeded','1')")
            self._ledger_ready = True

    @contextmanager
    def _file_lock(self, job_id):
        self._path(job_id)
        with (self._locks / job_id).open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def _reserve(self, job_id, total):
        self._ensure_ledger()
        with sqlite3.connect(self._ledger, timeout=5, isolation_level=None) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT reserved,total FROM reservations WHERE job_id=?", (job_id,)).fetchone()
            if row and (total < row[0] or (row[1] is not None and row[1] != total)):
                raise UploadConflict("Declared upload size changed")
            if not row or row[0] != total or row[1] is None:
                used = db.execute("SELECT COALESCE(SUM(reserved),0) FROM reservations").fetchone()[0]
                if used - (row[0] if row else 0) + total > self.max_total_bytes:
                    raise ValueError("Upload storage limit exceeded")
                db.execute("""INSERT INTO reservations(job_id,reserved,total,committed) VALUES (?,?,?,0)
                              ON CONFLICT(job_id) DO UPDATE SET reserved=excluded.reserved,total=excluded.total""",
                           (job_id, total, total))
            db.commit()

    def _commit_offset(self, job_id, offset):
        with sqlite3.connect(self._ledger, timeout=5) as db:
            db.execute("UPDATE reservations SET committed=? WHERE job_id=?", (offset, job_id))

    def append(self, job_id, offset, data, total):
        if not 0 <= offset <= total <= self.max_file_bytes:
            raise ValueError("Invalid upload size or offset")
        if not data or len(data) > self.max_chunk_bytes or offset + len(data) > total:
            raise ValueError("Invalid upload chunk")
        self._ensure_ledger()
        with self._file_lock(job_id):
            self._reserve(job_id, total)
            path = self._path(job_id)
            with path.open("a+b") as file:
                file.seek(0, os.SEEK_END)
                size = file.tell()
                with sqlite3.connect(self._ledger, timeout=5) as db:
                    committed = db.execute('SELECT committed FROM reservations WHERE job_id=?',
                                           (job_id,)).fetchone()[0]
                if size > committed:
                    # A crash can leave a partial, unacknowledged chunk. Replace only
                    # that suffix; previously acknowledged bytes remain immutable.
                    file.truncate(committed)
                    size = committed
                if offset < size:
                    if offset + len(data) > size:
                        raise UploadConflict("Chunk overlaps the stored boundary")
                    file.seek(offset)
                    if file.read(len(data)) != data:
                        raise UploadConflict("Chunk differs from the stored bytes")
                    os.fsync(file.fileno())
                    return size
                if offset != size:
                    raise UploadConflict("Incorrect upload offset")
                file.write(data)
                file.flush()
                os.fsync(file.fileno())
                if committed == 0:
                    directory_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                self._commit_offset(job_id, size + len(data))
                return size + len(data)

    def offset(self, job_id):
        path = self._path(job_id)
        try:
            actual = path.stat().st_size
        except FileNotFoundError:
            return 0
        if self._ledger.exists():
            with sqlite3.connect(self._ledger, timeout=5) as db:
                try:
                    row = db.execute("SELECT committed FROM reservations WHERE job_id=?", (job_id,)).fetchone()
                except sqlite3.OperationalError:
                    row = None
                if row:
                    return min(actual, row[0])
        return actual

    def remove(self, job_id):
        self._ensure_ledger()
        with self._file_lock(job_id):
            self._path(job_id).unlink(missing_ok=True)
            directory_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            with sqlite3.connect(self._ledger, timeout=5) as db:
                db.execute("DELETE FROM reservations WHERE job_id=?", (job_id,))

    def usage(self):
        self._ensure_ledger()
        with sqlite3.connect(self._ledger, timeout=5) as db:
            reserved, committed = db.execute(
                "SELECT COALESCE(SUM(reserved),0),COALESCE(SUM(committed),0) FROM reservations").fetchone()
        return {"reserved_bytes": reserved, "committed_bytes": committed, "limit_bytes": self.max_total_bytes}
