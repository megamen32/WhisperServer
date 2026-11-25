import pytest

def test_cold_diarizer_is_prepared_before_gpu_lease(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    import main
    from runtime_support import IdleModelCache
    events = []
    clock = [0]
    class Model:
        device = 'cpu'
        def to_device(self, device):
            events.append(('device', device))
            self.device = device
            return self
        def park_cpu(self):
            self.device = 'cpu'
        def cpu_size_bytes(self):
            return 100
        def unload(self):
            events.append(('unload', self.device))
    def prepare(device):
        events.append(('prepare', device))
        clock[0] += 50
        model = Model()
        model.device = device
        return model
    class Broker:
        def __init__(self, **kwargs):
            self.loader = kwargs['loader']
        def acquire(self):
            events.append(('acquire', 'gpu'))
            granted_at = clock[0]
            self.model = self.loader()
            assert clock[0] - granted_at < 20, 'GPU lease expired during cold preparation'
        @contextmanager
        def inference(self):
            yield self.model
        def release(self):
            events.append(('release', 'gpu'))
    monkeypatch.setattr(main, 'Community1Model', SimpleNamespace(from_pretrained=prepare))
    monkeypatch.setattr(main, 'ManagedModel', Broker)
    monkeypatch.setattr(main, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setattr(main, 'DIARIZATION_DEVICE', 'cuda')
    monkeypatch.setattr(main, '_diarization_cpu_cache', IdleModelCache())
    with main._diarization_model_session() as model:
        assert model.device == 'cuda'
    assert events.index(('prepare', 'cpu')) < events.index(('acquire', 'gpu'))
    assert events.index(('acquire', 'gpu')) < events.index(('device', 'cuda'))
    assert ('release', 'gpu') in events

def test_prepared_diarizer_is_unloaded_when_gpu_acquire_fails(monkeypatch):
    from types import SimpleNamespace
    import main
    from runtime_support import IdleModelCache
    events = []
    class Model:
        def unload(self):
            events.append('unloaded')
        def to_device(self, device):
            return self
    class Broker:
        def __init__(self, **kwargs):
            pass
        def acquire(self):
            raise RuntimeError('GPU lease rejected')
        def release(self):
            events.append('released')
    monkeypatch.setattr(main, 'Community1Model', SimpleNamespace(from_pretrained=lambda device: Model()))
    monkeypatch.setattr(main, 'ManagedModel', Broker)
    monkeypatch.setattr(main, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setattr(main, 'DIARIZATION_DEVICE', 'cuda')
    monkeypatch.setattr(main, '_diarization_cpu_cache', IdleModelCache())
    with pytest.raises(RuntimeError, match='GPU lease rejected'):
        with main._diarization_model_session():
            pass
    assert events == ['unloaded', 'released']
