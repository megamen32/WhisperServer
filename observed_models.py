"""Observe broker wait/load boundaries without retaining a previous job's hooks."""
from __future__ import annotations

import logging
from threading import RLock
import traceback

log = logging.getLogger(__name__)


def observed_managed_model(managed_type, *, loader, **options):
    """Keep ManagedModel semantics while reporting phases for each acquisition."""
    class ObservedManagedModel(managed_type):
        supports_preparation_hooks = True

        def __init__(self):
            self._observation_lock = RLock()
            self._cancel_check = None
            self._phase_hook = None
            configured = dict(options)
            if configured.get("cpu_fallback") is not None:
                configured["cpu_fallback"] = self._load_cpu_fallback
            super().__init__(loader=self._load_observed, **configured)

        def _load_observed(self):
            if self._cancel_check is not None:
                self._cancel_check()
            if self._phase_hook is not None:
                self._phase_hook("loading_gpu")
            # Let ManagedModel own the loaded object before a post-load cancel
            # check, so release() can always discard it and its granted lease.
            return loader()

        def _load_cpu_fallback(self):
            if self._cancel_check is not None:
                self._cancel_check()
            if self._phase_hook is not None:
                self._phase_hook("preparing_cpu")
            return options["cpu_fallback"]()

        def acquire(self, *, cancel_check=None, phase_hook=None):
            with self._observation_lock, getattr(self, "_lock", self._observation_lock):
                self._cancel_check, self._phase_hook = cancel_check, phase_hook
                result = None
                try:
                    if cancel_check is not None:
                        cancel_check()
                    if phase_hook is not None:
                        phase_hook("waiting_gpu")
                    result = super().acquire()
                    if cancel_check is not None:
                        cancel_check()
                    return result
                except BaseException as exc:
                    # The currently executing frame cannot be cleared by
                    # clear_frames; discard its model before returning VRAM.
                    result = None
                    traceback.clear_frames(exc.__traceback__)
                    try:
                        super().release()
                    except Exception:
                        log.exception("Failed to release GPU lease after interrupted acquisition")
                    raise
                finally:
                    self._cancel_check = self._phase_hook = None

    return ObservedManagedModel()
