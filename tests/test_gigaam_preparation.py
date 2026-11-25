"""GigaAM CPU preparation must not consume an already-granted GPU lease."""
from types import SimpleNamespace

import pytest

import main


@pytest.fixture
def prepared_runtime(monkeypatch):
    events, models = [], []
    clock = [0]
    failures = {"acquire": False, "promote": False}

    class Model:
        def __init__(self, device):
            self.device = device
            self.unloaded = False
            models.append(self)

        @classmethod
        def from_pretrained(cls, device):
            events.append(("prepare", device))
            clock[0] += 50
            return cls(device)

        def to_device(self, device):
            events.append(("promote", device))
            clock[0] += 1
            self.device = device
            if failures["promote"]:
                raise RuntimeError("GPU transfer failed")
            return self

        def unload(self):
            self.unloaded = True
            self.device = "unloaded"
            events.append(("discard", None))

    class Broker:
        def __init__(self, **kwargs):
            self.loader = kwargs["loader"]
            self.model = None
            self.current_device = "unloaded"

        def acquire(self):
            if self.model is not None and self.current_device == "cuda":
                events.append(("cached", None))
                return self.model
            events.append(("acquire", None))
            granted = clock[0]
            if failures["acquire"]:
                raise RuntimeError("GPU lease rejected")
            self.model = self.loader()
            self.current_device = "cuda"
            assert clock[0] - granted < 20, "GPU lease expired during cold GigaAM preparation"
            return self.model

        def release(self):
            events.append(("release", None))
            self.model = None
            self.current_device = "unloaded"

    monkeypatch.setattr(main, "GigaAMModel", Model)
    monkeypatch.setattr(main, "ManagedModel", Broker)
    monkeypatch.setattr(main, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    return SimpleNamespace(events=events, models=models, failures=failures, broker=Broker)


def test_cpu_preparation_finishes_before_gigaam_gpu_lease(prepared_runtime):
    runtime = prepared_runtime
    entry = main._create_model_entry("gigaam-v3")
    model = entry.acquire()
    assert isinstance(entry, runtime.broker)
    assert model.device == "cuda"
    assert runtime.events[:3] == [("prepare", "cpu"), ("acquire", None), ("promote", "cuda")]


def test_cached_cuda_model_is_not_prepared_again(prepared_runtime):
    runtime = prepared_runtime
    entry = main._create_model_entry("gigaam-v3")
    first = entry.acquire()
    assert entry.acquire() is first
    assert runtime.events.count(("prepare", "cpu")) == 1


@pytest.mark.parametrize("retired_state", ["evicted", "unloaded"])
def test_gigaam_prepares_fresh_weights_after_eviction_or_release(prepared_runtime, retired_state):
    runtime = prepared_runtime
    entry = main._create_model_entry("gigaam-v3")
    first = entry.acquire()
    if retired_state == "unloaded":
        entry.release()
    else:
        entry.model = None
        entry.current_device = "evicted"
    second = entry.acquire()
    assert second is not first
    assert second.device == "cuda"
    assert runtime.events.count(("prepare", "cpu")) == 2


@pytest.mark.parametrize("failure", ["acquire", "promote"])
def test_failed_gpu_acquisition_discards_cpu_preparation_and_can_retry(prepared_runtime, failure):
    runtime = prepared_runtime
    entry = main._create_model_entry("gigaam-v3")
    runtime.failures[failure] = True
    with pytest.raises(RuntimeError, match="GPU"):
        entry.acquire()
    assert runtime.models[0].unloaded
    assert entry.model is None and entry.current_device == "unloaded"
    if failure == "promote":
        assert runtime.events.index(("discard", None)) < runtime.events.index(("release", None))
    runtime.failures[failure] = False
    assert entry.acquire() is runtime.models[1]
    assert not runtime.models[1].unloaded


def test_prepared_subtype_uses_real_broker_inference_and_eviction_contract(prepared_runtime, monkeypatch):
    from contextlib import contextmanager
    import cudabroker_client.managed as client

    class Lease:
        local_mode = False
        def __init__(self):
            self.released = False
        def touch(self, **kwargs):
            return "active"
        def evict_ack(self):
            pass
        def release(self):
            self.released = True

    leases = []
    @contextmanager
    def lease_context(*args, **kwargs):
        lease = Lease()
        leases.append(lease)
        try:
            yield lease
        finally:
            lease.release()

    monkeypatch.setattr(client, "gpu_lease", lease_context)
    monkeypatch.setattr(client, "memory_allocated_mb", lambda: 0)
    monkeypatch.setattr(client.ManagedModel, "_start_heartbeat", lambda self: None)
    monkeypatch.setattr(client.ManagedModel, "_empty_cuda_cache", staticmethod(lambda: None))
    monkeypatch.setattr(main, "ManagedModel", client.ManagedModel)
    entry = main._create_model_entry("gigaam-v3")
    try:
        first = entry.acquire()
        with entry.inference() as active:
            assert active is first
            assert entry._active_count == 1
        assert entry._active_count == 0
        entry._evict_to_cpu()
        assert leases[0].released
        second = entry.acquire()
        assert second is not first and second.device == "cuda"
        with entry.inference() as active:
            assert active is second
    finally:
        entry.release()
    assert len(leases) == 2 and all(lease.released for lease in leases)
