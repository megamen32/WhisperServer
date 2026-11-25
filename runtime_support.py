"""Small runtime helpers; importing this module never imports a model backend."""
from __future__ import annotations

from collections import deque
import importlib
import logging
import math
import os
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class LazyImport:
    """Resolve a module or one exported object only when it is used."""
    def __init__(self, module, name=None):
        self.module, self.name, self._resolved = module, name, None

    def _load(self):
        if self._resolved is None:
            module = importlib.import_module(self.module)
            self._resolved = getattr(module, self.name) if self.name else module
        return self._resolved

    def __getattr__(self, name):
        return getattr(self._load(), name)

    def __call__(self, *args, **kwargs):
        return self._load()(*args, **kwargs)


def cuda_device_visible():
    """Select an initial default without initializing CUDA in the HTTP process."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1"):
        return False
    return Path("/dev/nvidiactl").exists()


class RestartBudget:
    """Bound restart attempts in a rolling window, with finite backoff."""
    def __init__(self, limit=3, window=300.0, base_delay=1.0, max_delay=30.0,
                 clock=time.monotonic):
        self.limit = max(1, int(limit))
        self.window = max(.01, float(window))
        self.base_delay = max(0.0, float(base_delay))
        self.max_delay = max(self.base_delay, float(max_delay))
        self.clock = clock
        self.attempts = deque()

    def _prune(self):
        now = self.clock()
        while self.attempts and self.attempts[0] <= now - self.window:
            self.attempts.popleft()
        return now

    def delay(self):
        now = self._prune()
        backoff = min(self.max_delay, self.base_delay * 2 ** min(len(self.attempts), 10))
        if len(self.attempts) >= self.limit:
            return max(backoff, self.attempts[0] + self.window - now)
        return backoff

    def record(self):
        now = self._prune()
        self.attempts.append(now)

    def count(self):
        self._prune()
        return len(self.attempts)


# These budgets cover lack of real progress in one phase, not total job time.
PHASE_TIMEOUTS = {
    "preparing": 300.0,
    "preparing_cpu": 900.0,
    "waiting_gpu": 900.0,
    "loading_gpu": 300.0,
    "inference": 600.0,
    "diarization": 3600.0,
}


class WorkerExecution:
    """Publish one synchronous snapshot before entering crash-prone native code.

    Queue.put uses a feeder thread and can lose the final active-task marker on
    a crash. A Manager dict assignment returns only after the snapshot is stored.
    Progress is work completed, never a clock/heartbeat. Checkpoint is a monotonic
    durable watermark that can be compared across retries of the same job.
    """
    def __init__(self, shared=None):
        self.shared = shared
        initial = shared.get("snapshot", {}) if shared is not None else {}
        self.generation = initial.get("generation", 0)
        self.snapshot = {"generation": self.generation, "phase": "idle"}

    def _publish(self, snapshot):
        self.snapshot = snapshot
        if self.shared is not None:
            self.shared["snapshot"] = snapshot

    def enter(self, request_id, job_key, phase, *, unit=None, progress=None, checkpoint=None):
        if phase not in PHASE_TIMEOUTS:
            raise ValueError(f"Unknown worker phase: {phase}")
        self._publish({
            "generation": self.generation, "request_id": request_id,
            "job_key": list(job_key) if job_key is not None else None,
            "phase": phase, "unit": str(unit) if unit is not None else None,
            "progress": progress, "checkpoint": max(0, int(checkpoint or 0)),
        })

    def phase(self, phase):
        if self.snapshot.get("phase") == "idle":
            return
        if phase not in PHASE_TIMEOUTS:
            raise ValueError(f"Unknown worker phase: {phase}")
        if phase != self.snapshot["phase"]:
            self._publish(dict(self.snapshot, phase=phase, progress=None))

    def advance(self, progress=None, *, checkpoint=None):
        if self.snapshot.get("phase") == "idle":
            return
        snapshot = dict(self.snapshot)
        if progress is not None:
            snapshot["progress"] = progress
        if checkpoint is not None:
            snapshot["checkpoint"] = max(snapshot.get("checkpoint", 0), int(checkpoint))
        if snapshot != self.snapshot:
            self._publish(snapshot)

    def idle(self):
        self._publish({"generation": self.generation, "phase": "idle"})


class PhaseWatchdog:
    """Timeout one unchanged work boundary without treating liveness as progress."""
    def __init__(self, budgets=None, *, clock=time.monotonic):
        self.budgets = dict(PHASE_TIMEOUTS)
        if budgets is not None:
            self.budgets.update(budgets)
        if any(not math.isfinite(float(value)) or float(value) <= 0 for value in self.budgets.values()):
            raise ValueError("worker phase timeouts must be finite and positive")
        self.clock = clock
        self.snapshot = None
        self.since = None
        self._token = None

    @classmethod
    def from_environment(cls, **kwargs):
        return cls({phase: float(os.getenv(f"WHISPER_WATCHDOG_{phase.upper()}_SECONDS", default))
                    for phase, default in PHASE_TIMEOUTS.items()}, **kwargs)

    def observe(self, snapshot, generation):
        # Old-generation snapshots cannot revive attribution or its deadline.
        if not isinstance(snapshot, dict) or snapshot.get("generation") != generation:
            return None
        phase = snapshot.get("phase")
        if phase == "idle":
            self.snapshot, self.since, self._token = None, None, None
            return None
        if phase not in self.budgets or not snapshot.get("request_id"):
            return None
        token = (generation, snapshot.get("request_id"), snapshot.get("job_key"),
                 phase, snapshot.get("unit"), snapshot.get("progress"),
                 snapshot.get("checkpoint", 0))
        now = self.clock()
        if token != self._token:
            self._token, self.since = token, now
        self.snapshot = dict(snapshot)
        if now - self.since >= self.budgets[phase]:
            return dict(snapshot, reason="phase_timeout")
        return None

    def view(self):
        if self.snapshot is None:
            return {"phase": "idle", "stalled_seconds": None, "timeout_seconds": None}
        phase = self.snapshot["phase"]
        return {"phase": phase, "stalled_seconds": max(0.0, round(self.clock() - self.since, 2)),
                "timeout_seconds": self.budgets[phase]}


class IdleModelCache:
    """Keep at most one CPU model, bounded by measured tensor bytes and TTL."""
    def __init__(self, ttl=300.0, max_bytes=1024 ** 3, clock=time.monotonic):
        self.ttl = max(0.0, float(ttl))
        self.max_bytes = max(0, int(max_bytes))
        self.clock = clock
        self.model = None
        self.expires = 0.0
        self.bytes = 0

    @staticmethod
    def _dispose(model):
        try:
            model.unload()
        except Exception:
            logger.warning("Could not discard idle CPU model", exc_info=True)

    def clear(self):
        model, self.model = self.model, None
        self.bytes, self.expires = 0, 0.0
        if model is not None:
            self._dispose(model)

    def prune(self):
        if self.model is not None and self.clock() >= self.expires:
            self.clear()

    def take(self):
        self.prune()
        model, self.model = self.model, None
        self.bytes, self.expires = 0, 0.0
        return model

    def put(self, model):
        self.clear()
        try:
            size = int(model.cpu_size_bytes())
        except Exception:
            size = 0
        if (str(getattr(model, "device", "")) != "cpu" or self.ttl <= 0
                or size <= 0 or size > self.max_bytes):
            self._dispose(model)
            return False
        self.model, self.bytes = model, size
        self.expires = self.clock() + self.ttl
        return True
