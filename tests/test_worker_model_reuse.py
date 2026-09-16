from contextlib import contextmanager
from queue import Queue
from types import SimpleNamespace

import main


def test_worker_reuses_models_and_marks_inference_idle_after_each_request(monkeypatch):
    entries = []
    calls = []
    class Managed:
        def __init__(self, name):
            self.name = name
            self.active = False
            entries.append(self)
        def acquire(self):
            return self
        @contextmanager
        def inference(self):
            self.active = True
            calls.append(('begin', self.name))
            try:
                yield self
            finally:
                self.active = False
                calls.append(('end', self.name))
        def touch(self, **kwargs):
            pass
        def transcribe(self, *args, **kwargs):
            assert self.active, 'lease must be pinned for the entire inference'
            return [], SimpleNamespace(language='ru', language_probability=1)
    monkeypatch.setattr(main, 'ManagedModel', Managed)
    monkeypatch.setattr(main, '_create_model_entry', Managed)
    monkeypatch.setattr(main, 'audio_duration_seconds', lambda p: 1)
    monkeypatch.setattr(main, 'append_metric', lambda m: None)
    monkeypatch.setattr(main, 'reset_cuda_peak_memory', lambda: None)
    monkeypatch.setattr(main, 'peak_vram_mb', lambda: 0)
    requests, responses = Queue(), Queue()
    for i, model in enumerate(['gigaam-v3', 'large-v3', 'gigaam-v3']):
        requests.put({'request_id': i, 'model': model, 'audio_bytes': b'audio'})
    checks = iter([False, True])
    main.model_worker(requests, responses, SimpleNamespace(is_set=lambda: next(checks)), {})
    results = [responses.get_nowait() for _ in range(3)]
    assert all(r.get('final') for r in results), results
    assert len(entries) == 2
    assert sum(kind == 'begin' for kind, _ in calls) == 3
    assert sum(kind == 'end' for kind, _ in calls) == 3
    assert not any(e.active for e in entries)
