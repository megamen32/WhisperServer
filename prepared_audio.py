"""Private durable PCM sources and context plans with crash-safe borrow locks."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import uuid


def _key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json(path, value):
    encoded = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) > 8 * 1024 ** 2:
        raise ValueError("Audio preparation metadata exceeds its storage limit")
    temporary = path.with_name("." + path.name + "-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _take_lock(handle, mode, cancel_check=None):
    while True:
        if cancel_check is not None:
            cancel_check()
        try:
            fcntl.flock(handle, mode | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            time.sleep(.05)


class PreparedAudio:
    def __init__(self, cache, directory, handle, metadata, reused):
        self.cache, self.directory, self.handle = cache, directory, handle
        self.metadata, self.reused = metadata, reused
        self.path = directory / "data" / "audio.pcm"

    def context(self, options, build, cancel_check=None):
        """Publish immutable context boundaries independently of the source PCM."""
        key = _key({"schema": "audio-context-v1", "source": self.directory.name, "options": options})
        plans = self.directory / "data" / "plans"
        plans.mkdir(exist_ok=True)
        _sync_directory(self.directory / "data")
        path = plans / (key + ".json")
        with (plans / ".lock").open("a+b") as lock:
            _take_lock(lock, fcntl.LOCK_EX, cancel_check)
            try:
                value = json.loads(path.read_text())
                chunks = value["chunks"]
                if value.get("key") != key or any(
                        not (type(a) is int and type(b) is int and 0 <= a < b <= self.metadata["samples"])
                        for a, b in chunks):
                    raise ValueError("Invalid audio plan")
                return [tuple(item) for item in chunks], True
            except (OSError, ValueError, KeyError, TypeError):
                pass
            if cancel_check is not None:
                cancel_check()
            chunks = build()
            if cancel_check is not None:
                cancel_check()
            value = {"key": key, "chunks": chunks}
            size = len(json.dumps(value, separators=(",", ":")).encode())
            with self.cache._global(cancel_check):
                # Bound distinct contexts for one recording; active callers keep
                # their already-read boundaries in memory, not a live JSON file.
                old = sorted(plans.glob("*.json"), key=lambda item: item.stat().st_mtime)
                while len(old) >= self.cache.max_contexts:
                    old.pop(0).unlink()
                previous_size = path.stat().st_size if path.exists() else 0
                self.cache._prune_locked(max(0, size - previous_size), keep={self.directory.name})
                _atomic_json(path, value)
            return chunks, False

    def close(self):
        if self.handle is not None:
            try:
                os.utime(self.directory, None)
            except FileNotFoundError:
                pass
            finally:
                self.handle.close()
                self.handle = None


class PreparedAudioCache:
    """Complete PCM is published atomically; flock references die with a worker."""
    def __init__(self, root, *, max_bytes=8 * 1024 ** 3, ttl_seconds=86400,
                 max_source_bytes=2 * 1024 ** 3, min_free_bytes=1024 ** 3,
                 max_entries=1000, max_contexts=128):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.max_bytes, self.ttl_seconds = max_bytes, ttl_seconds
        self.max_source_bytes, self.min_free_bytes = max_source_bytes, min_free_bytes
        self.max_entries, self.max_contexts = max_entries, max_contexts

    @contextmanager
    def _global(self, cancel_check=None):
        with (self.root / ".cache.lock").open("a+b") as lock:
            _take_lock(lock, fcntl.LOCK_EX, cancel_check)
            yield

    @staticmethod
    def _same_lock(directory, handle):
        try:
            return (directory / ".lock").stat().st_ino == os.fstat(handle.fileno()).st_ino
        except FileNotFoundError:
            return False

    def _entry_lock(self, key, mode, cancel_check=None):
        # A pruner can remove an idle directory while a borrower waits on its
        # old lock inode. Reopen in that case rather than using the retired lock.
        while True:
            directory = self.root / key
            handle = None
            try:
                directory.mkdir(exist_ok=True, mode=0o700)
                handle = (directory / ".lock").open("a+b")
                _take_lock(handle, mode, cancel_check)
                if self._same_lock(directory, handle):
                    return directory, handle
            except FileNotFoundError:
                pass
            except BaseException:
                if handle is not None:
                    handle.close()
                raise
            if handle is not None:
                handle.close()

    @staticmethod
    def _metadata(directory):
        try:
            metadata = json.loads((directory / "data" / "metadata.json").read_text())
            samples = metadata["samples"]
            if type(samples) is not int or samples < 0:
                return None
            if (directory / "data" / "audio.pcm").stat().st_size != samples * 4:
                return None
            return metadata
        except (OSError, ValueError, TypeError, KeyError):
            return None

    @staticmethod
    def _bytes(directory):
        # Include abandoned staging data, not just successfully published PCM.
        return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())

    def _evict(self, directory):
        try:
            with (directory / ".lock").open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if not self._same_lock(directory, lock):
                    return False
                shutil.rmtree(directory)
                return True
        except (BlockingIOError, FileNotFoundError):
            return False

    def _prune_locked(self, extra=0, *, keep=()):
        entries = []
        now = time.time()
        for directory in self.root.iterdir():
            if not directory.is_dir() or directory.is_symlink() or not re.fullmatch("[0-9a-f]{64}", directory.name):
                continue
            try:
                stamp = directory.stat().st_mtime
                metadata = self._metadata(directory)
                if directory.name not in keep and (metadata is None or now - stamp > self.ttl_seconds):
                    if self._evict(directory):
                        continue
                entries.append((stamp, directory, self._bytes(directory)))
            except FileNotFoundError:
                continue
        total = sum(size for _, _, size in entries)
        count = len(entries)
        for _, directory, size in sorted(entries):
            if total + extra <= self.max_bytes and count <= self.max_entries:
                break
            if directory.name not in keep and self._evict(directory):
                total -= size
                count -= 1
        if total + extra > self.max_bytes or count > self.max_entries:
            raise ValueError("Concurrent decoded-audio storage limit reached; retry after another job finishes")

    def prune(self):
        with (self.root / ".cache.lock").open("a+b") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            try:
                self._prune_locked()
            except ValueError:
                pass  # Active references can temporarily outlive a lowered quota.

    def evict_source(self, audio_hash):
        """Best-effort explicit source removal; active borrowers keep their loans."""
        removed = 0
        with (self.root / ".cache.lock").open("a+b") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return 0
            for directory in self.root.iterdir():
                if not directory.is_dir() or directory.is_symlink() or not re.fullmatch("[0-9a-f]{64}", directory.name):
                    continue
                metadata = self._metadata(directory)
                if metadata is not None and metadata.get("source") == audio_hash:
                    removed += bool(self._evict(directory))
        return removed

    def borrow(self, audio_hash, decoder_identity, decode, *, cancel_check=None):
        import numpy as np
        key = _key({"schema": "pcm-source-v1", "audio": audio_hash, "decoder": decoder_identity})
        directory, handle = self._entry_lock(key, fcntl.LOCK_SH, cancel_check)
        metadata = self._metadata(directory)
        reused = metadata is not None
        if metadata is None:
            handle.close()
            directory, handle = self._entry_lock(key, fcntl.LOCK_EX, cancel_check)
            try:
                # EX proves no builder/reader owns this source. A terminated
                # builder may have left arbitrarily large staging files.
                for abandoned in directory.glob(".building-*"):
                    if abandoned.is_dir() and not abandoned.is_symlink():
                        shutil.rmtree(abandoned)
                    else:
                        abandoned.unlink(missing_ok=True)
                metadata = self._metadata(directory)
                reused = metadata is not None
                if metadata is None:
                    old_data = directory / "data"
                    if old_data.exists():
                        shutil.rmtree(old_data)
                    with self._global(cancel_check):
                        if cancel_check is not None:
                            cancel_check()
                        samples = np.asarray(decode(), dtype="<f4")
                        if cancel_check is not None:
                            cancel_check()
                        if samples.ndim != 1 or samples.nbytes > self.max_source_bytes:
                            raise ValueError("Audio exceeds the decoded-audio storage limit")
                        metadata = {"samples": len(samples), "source": audio_hash,
                                    "decoder": decoder_identity, "schema": "pcm-source-v1"}
                        self._prune_locked(samples.nbytes + len(json.dumps(metadata).encode()) + 64,
                                           keep={key})
                        if shutil.disk_usage(self.root).free < samples.nbytes + self.min_free_bytes:
                            raise ValueError("Not enough temporary storage for audio decoding")
                        with tempfile.TemporaryDirectory(prefix=".building-", dir=directory) as stage:
                            stage = Path(stage)
                            with (stage / "audio.pcm").open("wb") as output:
                                samples.tofile(output)
                                output.flush()
                                os.fsync(output.fileno())
                            _atomic_json(stage / "metadata.json", metadata)
                            if cancel_check is not None:
                                cancel_check()
                            old_data = directory / "data"
                            if old_data.exists():
                                shutil.rmtree(old_data)
                            os.replace(stage, old_data)
                            _sync_directory(directory)
                        del samples
                fcntl.flock(handle, fcntl.LOCK_SH)
                if not self._same_lock(directory, handle):
                    handle.close()
                    return self.borrow(audio_hash, decoder_identity, decode, cancel_check=cancel_check)
            except BaseException:
                handle.close()
                raise
        try:
            os.utime(directory, None)
            return PreparedAudio(self, directory, handle, metadata, reused)
        except BaseException:
            handle.close()
            raise
