"""Global speaker clustering remains cancellable between pipeline batches."""
import asyncio
from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import diarization_backend as backend


@pytest.fixture
def decoded_audio(monkeypatch):
    def decode(command, **kwargs):
        Path(command[-1]).write_bytes(np.zeros(16000, dtype='<f4').tobytes())
    monkeypatch.setattr(backend.subprocess, 'run', decode)
    monkeypatch.setattr(backend.shutil, 'disk_usage', lambda path: SimpleNamespace(free=100 * 1024**3))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(get_num_threads=lambda: 2, set_num_threads=lambda n: None))


def test_global_diarization_reports_pipeline_progress(decoded_audio):
    seen, released = [], []
    turns = [{'start': 0.0, 'end': 1.0, 'speaker': 'SPEAKER_00'}]
    class Model:
        def diarize(self, samples, hook=None):
            assert len(samples) == 16000
            for count in (0, 1):
                hook('segmentation', total=1, completed=count)
            return turns
    @contextmanager
    def provider():
        try:
            yield Model()
        finally:
            released.append(True)
    result = backend.diarize_audio('fixture.wav', model_provider=provider,
                                   progress_hook=lambda step, **kw: seen.append((step, kw['completed'])))
    assert result == turns
    assert seen == [('segmentation', 0), ('segmentation', 1)]
    assert released == [True]


def test_cancel_from_pipeline_hook_releases_provider(decoded_audio):
    released = []
    class Model:
        def diarize(self, samples, hook=None):
            hook('embeddings', total=10, completed=2)
            pytest.fail('cancel should stop this global pass')
    @contextmanager
    def provider():
        try:
            yield Model()
        finally:
            released.append(True)
    def cancel(step, **kwargs):
        raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        backend.diarize_audio('fixture.wav', model_provider=provider, progress_hook=cancel)
    assert released == [True]
