"""Durable, content-addressed completed ASR units and a cooperative scheduler.

Attempts own publication; immutable units belong to an audio/configuration
fingerprint and can be reused after an interrupted attempt.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import time


def computation_key(audio_hash: str, model: str, revision: str, options: dict) -> str:
    value = {"schema": "audio-units-v1", "audio": audio_hash, "model": model,
             "revision": revision, "options": options}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class AudioCheckpoints:
    """Only completed units are committed. Values are immutable per signature."""
    def __init__(self, path, *, ttl_seconds=7 * 86400, max_units=100_000,
                 max_payload_bytes=8_000_000, max_bytes=256 * 1024 ** 2):
        self.path = str(path)
        self.ttl_seconds = ttl_seconds
        self.max_units = max_units
        self.max_payload_bytes = max_payload_bytes
        self.max_bytes = max_bytes
        if min(ttl_seconds, max_units, max_payload_bytes, max_bytes) <= 0:
            raise ValueError("checkpoint limits must be positive")
        Path(path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS audio_units(
                signature TEXT NOT NULL, unit TEXT NOT NULL, payload TEXT NOT NULL,
                created REAL NOT NULL, payload_bytes INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(signature, unit))""")
            columns = {row[1] for row in db.execute("PRAGMA table_info(audio_units)")}
            if 'payload_bytes' not in columns:
                db.execute("ALTER TABLE audio_units ADD COLUMN payload_bytes INTEGER NOT NULL DEFAULT 0")
            db.execute("UPDATE audio_units SET payload_bytes=length(CAST(payload AS BLOB)) WHERE payload_bytes=0")
            db.execute("CREATE INDEX IF NOT EXISTS audio_units_created ON audio_units(created)")
            db.execute("DELETE FROM audio_units WHERE created < ?", (time.time() - ttl_seconds,))
            db.execute("CREATE TABLE IF NOT EXISTS audio_usage(id INTEGER PRIMARY KEY CHECK(id=1), units INTEGER NOT NULL, bytes INTEGER NOT NULL)")
            units, size = db.execute("SELECT count(*),coalesce(sum(payload_bytes),0) FROM audio_units").fetchone()
            db.execute("INSERT OR REPLACE INTO audio_usage VALUES (1,?,?)", (units, size))
            self._trim(db, units, size)
        try:
            Path(path).chmod(0o600)
        except OSError:
            pass

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            with db:
                yield db
        finally:
            db.close()

    def get(self, signature, unit):
        with self._db() as db:
            row = db.execute("SELECT payload FROM audio_units WHERE signature=? AND unit=? AND created>=?",
                             (signature, str(unit), time.time() - self.ttl_seconds)).fetchone()
        return json.loads(row[0]) if row else None

    def _trim(self, db, units, size):
        while units > self.max_units or size > self.max_bytes:
            rows = db.execute("SELECT rowid,payload_bytes FROM audio_units ORDER BY created LIMIT 32").fetchall()
            if not rows:
                break
            for rowid, payload_bytes in rows:
                db.execute("DELETE FROM audio_units WHERE rowid=?", (rowid,))
                units, size = units - 1, size - payload_bytes
                if units <= self.max_units and size <= self.max_bytes:
                    break
        db.execute("UPDATE audio_usage SET units=?,bytes=? WHERE id=1", (units, size))

    def put(self, signature, unit, payload):
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        payload_bytes = len(encoded.encode())
        if payload_bytes > min(self.max_payload_bytes, self.max_bytes):
            raise ValueError("audio checkpoint payload too large")
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute("SELECT payload,created,payload_bytes FROM audio_units WHERE signature=? AND unit=?",
                                  (signature, str(unit))).fetchone()
            if previous and previous[1] >= time.time() - self.ttl_seconds:
                return json.loads(previous[0])
            units, size = db.execute("SELECT units,bytes FROM audio_usage WHERE id=1").fetchone()
            if previous:
                db.execute("DELETE FROM audio_units WHERE signature=? AND unit=?", (signature, str(unit)))
                units, size = units - 1, size - previous[2]
            db.execute("INSERT INTO audio_units(signature,unit,payload,created,payload_bytes) VALUES (?,?,?,?,?)",
                       (signature, str(unit), encoded, time.time(), payload_bytes))
            self._trim(db, units + 1, size + payload_bytes)
        return payload


class CooperativeQueue:
    """New arrivals join before the just-finished continuation is requeued."""
    def __init__(self):
        self.ready = deque()

    def add(self, steps):
        self.ready.append(steps)

    def __bool__(self):
        return bool(self.ready)

    def pop(self):
        return self.ready.popleft()

    def close(self):
        while self.ready:
            self.ready.popleft().close()
