"""Prepare CPU weights before acquiring a short-lived broker GPU lease."""
from __future__ import annotations

import logging
from threading import RLock
import traceback

log = logging.getLogger(__name__)


def cpu_prepared_managed_model(managed_type, *, prepare, promote, discard, **options):
    """Keep the broker contract while preparing again after eviction or release.

    The returned subtype remains compatible with the worker's ManagedModel
    checks. No weights or optional ML modules are imported by this factory.
    """
    class PreparedManagedModel(managed_type):
        def __init__(self):
            self._preparation_lock = RLock()
            self._prepared_cpu = None
            super().__init__(loader=self._promote_prepared, **options)

        def _discard_prepared(self):
            model, self._prepared_cpu = self._prepared_cpu, None
            if model is not None:
                discard(model)

        def _promote_prepared(self):
            model, self._prepared_cpu = self._prepared_cpu, None
            if model is None:
                raise RuntimeError("CPU model was not prepared before GPU acquisition")
            try:
                return promote(model)
            except BaseException as exc:
                traceback.clear_frames(exc.__traceback__)
                discard(model)
                raise

        def acquire(self):
            # The broker heartbeat may evict a cached model. Hold the same lock
            # across the ready check and acquire so it cannot invalidate that
            # check and unexpectedly enter the loader without CPU preparation.
            with self._preparation_lock, getattr(self, "_lock", self._preparation_lock):
                try:
                    if getattr(self, "model", None) is None or getattr(self, "current_device", None) != "cuda":
                        self._prepared_cpu = prepare()
                    return super().acquire()
                except BaseException as exc:
                    traceback.clear_frames(exc.__traceback__)
                    # The underlying client can leave a granted lease behind
                    # when its loader raises. Release only after dropping the
                    # failed transfer's model references.
                    try:
                        super().release()
                    except Exception:
                        log.exception("Failed to release GPU lease after preparation failure")
                    raise
                finally:
                    self._discard_prepared()

        def release(self):
            with self._preparation_lock:
                try:
                    super().release()
                finally:
                    self._discard_prepared()

    return PreparedManagedModel()
