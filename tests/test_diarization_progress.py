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


@pytest.mark.parametrize("cancel_stage", ["decode", "model", "inference"])
def test_phase_cancel_gates_stop_before_next_expensive_step(decoded_audio, monkeypatch, cancel_stage):
    cancelled, calls = [False], []
    decode = backend.subprocess.run
    def run(*args, **kwargs):
        decode(*args, **kwargs)
        calls.append("decode")
        cancelled[0] = cancel_stage == "decode"
    monkeypatch.setattr(backend.subprocess, "run", run)
    class Model:
        def diarize(self, samples):
            calls.append("inference")
            cancelled[0] = cancel_stage == "inference"
            return [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]
    @contextmanager
    def provider():
        calls.append("model")
        cancelled[0] = cancel_stage == "model"
        try:
            yield Model()
        finally:
            calls.append("released")
    def check():
        if cancelled[0]:
            raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        backend.diarize_audio("fixture.wav", model_provider=provider, cancel_check=check)
    assert calls == {
        "decode": ["decode"],
        "model": ["decode", "model", "released"],
        "inference": ["decode", "model", "inference", "released"],
    }[cancel_stage]
