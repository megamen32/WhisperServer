"""Small runtime helpers; importing this module never imports a model backend."""
from __future__ import annotations

from collections import deque
import importlib
import logging
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
