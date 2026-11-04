"""Bounded, resumable upload spool for browser transcription jobs."""
import fcntl
import os
import re
from pathlib import Path


class UploadConflict(ValueError):
    """The requested offset or replay disagrees with the stored upload."""


class UploadSpool:
    def __init__(self, root, max_file_bytes, max_chunk_bytes, max_total_bytes):
        self.root = Path(root)
        self.max_file_bytes = max_file_bytes
        self.max_chunk_bytes = max_chunk_bytes
        self.max_total_bytes = max_total_bytes
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)

    def _path(self, job_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_id):
            raise ValueError("Invalid job ID")
        return self.root / job_id

    def append(self, job_id, offset, data, total):
        if not 0 <= offset <= total <= self.max_file_bytes:
            raise ValueError("Invalid upload size or offset")
        if not data or len(data) > self.max_chunk_bytes or offset + len(data) > total:
            raise ValueError("Invalid upload chunk")
        # Serialize accounting across uploads, including concurrent writers.
        with (self.root / ".lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self._path(job_id)
            with path.open("a+b") as file:
                file.seek(0, os.SEEK_END)
                size = file.tell()
                if offset < size:
                    if offset + len(data) > size:
                        raise UploadConflict("Chunk overlaps the stored boundary")
                    file.seek(offset)
                    if file.read(len(data)) != data:
                        raise UploadConflict("Chunk differs from the stored bytes")
                    return size
                if offset != size:
                    raise UploadConflict("Incorrect upload offset")
                used = sum(p.stat().st_size for p in self.root.iterdir() if p.is_file() and p.name != ".lock")
                if used + len(data) > self.max_total_bytes:
                    raise ValueError("Upload storage limit exceeded")
                file.write(data)
                file.flush()
                os.fsync(file.fileno())
                # Persist the directory entry as well as the chunk bytes, so a
                # confirmed first chunk remains discoverable after a crash.
                directory_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                return size + len(data)

    def offset(self, job_id):
        path = self._path(job_id)
        try:
            return path.stat().st_size
        except FileNotFoundError:
            return 0

    def remove(self, job_id):
        with (self.root / ".lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._path(job_id).unlink(missing_ok=True)
