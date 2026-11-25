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
        supports_preparation_hooks = True

        def __init__(self):
            self._preparation_lock = RLock()
            self._prepared_cpu = None
            self._cancel_check = self._phase_hook = None
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
                if self._cancel_check is not None:
                    self._cancel_check()
                if self._phase_hook is not None:
                    self._phase_hook("loading_gpu")
                result = promote(model)
                if self._cancel_check is not None:
                    self._cancel_check()
                return result
            except BaseException as exc:
                traceback.clear_frames(exc.__traceback__)
                discard(model)
                raise

        def acquire(self, *, cancel_check=None, phase_hook=None):
            # Hooks are per call: a cached entry must never retain another
            # request's cancellation or execution identity.
            with self._preparation_lock, getattr(self, "_lock", self._preparation_lock):
                self._cancel_check, self._phase_hook = cancel_check, phase_hook
                result = None
                try:
                    if cancel_check is not None:
                        cancel_check()
                    if getattr(self, "model", None) is None or getattr(self, "current_device", None) != "cuda":
                        if phase_hook is not None:
                            phase_hook("preparing_cpu")
                        self._prepared_cpu = prepare()
                        if cancel_check is not None:
                            cancel_check()
                    if phase_hook is not None:
                        phase_hook("waiting_gpu")
                    result = super().acquire()
                    if cancel_check is not None:
                        cancel_check()
                    return result
                except BaseException as exc:
                    result = None
                    traceback.clear_frames(exc.__traceback__)
                    try:
                        super().release()
                    except Exception:
                        log.exception("Failed to release GPU lease after preparation failure")
                    raise
                finally:
                    self._cancel_check = self._phase_hook = None
                    self._discard_prepared()

        def release(self):
            with self._preparation_lock:
                try:
                    super().release()
                finally:
                    self._discard_prepared()

    return PreparedManagedModel()
