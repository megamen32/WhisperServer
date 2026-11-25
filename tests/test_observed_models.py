"""Acquisition phase observation must survive cached and cancelled requests."""
from threading import RLock

import pytest

from observed_models import observed_managed_model


def test_each_acquire_uses_current_hooks_and_reports_actual_loads():
    events = []
    class Managed:
        def __init__(self, loader, **kwargs):
            self.loader, self.model, self._lock = loader, None, RLock()
        def acquire(self):
            if self.model is None: self.model = self.loader()
            return self.model
        def release(self): self.model = None; events.append("release")
    entry = observed_managed_model(Managed, loader=lambda: events.append("load") or object())
    first, second = [], []
    entry.acquire(phase_hook=first.append)
    entry.acquire(phase_hook=second.append)
    assert first == ["waiting_gpu", "loading_gpu"]
    assert second == ["waiting_gpu"]
    assert events == ["load"]
    entry.release()
    entry.acquire(phase_hook=second.append)
    assert first == ["waiting_gpu", "loading_gpu"]
    assert second == ["waiting_gpu", "waiting_gpu", "loading_gpu"]
    assert entry._phase_hook is None and entry._cancel_check is None


@pytest.mark.parametrize("where", ["waiting", "loading", "after_load"])
def test_cancelled_acquisition_releases_lease_and_drops_callbacks(where):
    events = []
    class Cancelled(BaseException): pass
    class Managed:
        def __init__(self, loader, **kwargs): self.loader = loader; self.model = None
        def acquire(self):
            events.append("grant")
            self.model = self.loader()
            return self.model
        def release(self): self.model = None; events.append("release")
    def load(): events.append("load"); return object()
    calls = [0]
    def check():
        calls[0] += 1
        target = {"waiting": 1, "loading": 2, "after_load": 3}[where]
        if calls[0] == target: raise Cancelled()
    entry = observed_managed_model(Managed, loader=load)
    with pytest.raises(Cancelled):
        entry.acquire(cancel_check=check)
    assert entry.model is None
    assert events[-1] == "release"
    assert ("load" in events) is (where == "after_load")
    assert entry._phase_hook is None and entry._cancel_check is None


def test_post_load_cancel_drops_local_model_before_releasing_gpu():
    import weakref
    class Cancelled(BaseException): pass
    references, released = [], []
    class Model: pass
    class Managed:
        def __init__(self, loader, **kwargs): self.loader = loader; self.model = None
        def acquire(self): self.model = self.loader(); return self.model
        def release(self):
            self.model = None
            released.append(references[0]() is None)
    def load():
        model = Model()
        references.append(weakref.ref(model))
        return model
    def check():
        if references:
            raise Cancelled()
    entry = observed_managed_model(Managed, loader=load)
    with pytest.raises(Cancelled):
        entry.acquire(cancel_check=check)
    assert references[0]() is None
    assert released == [True], "lease released while cancelled call still owns GPU model"


def test_broker_cpu_fallback_has_its_own_cold_load_phase():
    phases = []
    class Managed:
        def __init__(self, loader, cpu_fallback): self.cpu_fallback = cpu_fallback
        def acquire(self): return self.cpu_fallback()
        def release(self): pass
    model = object()
    entry = observed_managed_model(Managed, loader=lambda: None, cpu_fallback=lambda: model)
    assert entry.acquire(phase_hook=phases.append) is model
    assert phases == ["waiting_gpu", "preparing_cpu"]
