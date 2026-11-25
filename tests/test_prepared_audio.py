"""Preparation publication, bounded retention and crash-safe source references."""
import shutil

import numpy as np
import pytest

import prepared_audio
from prepared_audio import PreparedAudioCache


def cache_at(tmp_path, **options):
    return PreparedAudioCache(tmp_path / "pcm", min_free_bytes=0, **options)


def test_completed_pcm_and_context_are_reused_with_exact_option_identity(tmp_path):
    cache = cache_at(tmp_path)
    calls = []
    def decode():
        calls.append("decode")
        return np.arange(20, dtype=np.float32)
    source = cache.borrow("source", {"decoder": "v1"}, decode)
    options = {"vad": True, "speakers": [{"start": 0, "end": 1, "speaker": "SPEAKER_00"}],
               "adapter": "v1"}
    assert source.context(options, lambda: calls.append("vad") or [(2, 17)]) == ([(2, 17)], False)
    expected = source.path.read_bytes()
    source.close()
    restored = cache.borrow("source", {"decoder": "v1"}, decode)
    assert restored.reused and restored.path.read_bytes() == expected
    assert restored.context(options, lambda: pytest.fail("VAD repeated")) == ([(2, 17)], True)
    assert restored.context({**options, "adapter": "v2"}, lambda: [(3, 16)]) == ([(3, 16)], False)
    assert calls == ["decode", "vad"]
    restored.close()


def test_active_reference_survives_ttl_quota_and_explicit_source_removal(tmp_path):
    cache = cache_at(tmp_path, max_bytes=1000, ttl_seconds=-1)
    source = cache.borrow("active", {}, lambda: np.zeros(100, dtype=np.float32))
    cache.prune()
    assert source.path.exists()
    assert cache.evict_source("active") == 0
    with pytest.raises(ValueError, match="storage limit"):
        cache.borrow("second", {}, lambda: np.zeros(100, dtype=np.float32))
    assert source.path.exists()
    source.close()
    restored = cache.borrow("second", {}, lambda: np.zeros(100, dtype=np.float32))
    assert not source.path.exists()
    restored.close()
    assert cache.evict_source("second") == 1


def test_orphan_build_is_accounted_and_removed_before_rebuild(tmp_path):
    cache = cache_at(tmp_path, max_bytes=1000)
    source = cache.borrow("source", {}, lambda: np.zeros(10, dtype=np.float32))
    directory = source.directory
    source.close()
    shutil.rmtree(directory / "data")
    stage = directory / ".building-killed-worker"
    stage.mkdir()
    (stage / "audio.pcm").write_bytes(b"x" * 5000)
    assert cache._bytes(directory) >= 5000
    restored = cache.borrow("source", {}, lambda: np.zeros(10, dtype=np.float32))
    assert not stage.exists()
    assert restored.path.stat().st_size == 40
    restored.close()
    assert cache._bytes(directory) <= cache.max_bytes


def test_cancelled_publication_never_exposes_partial_pcm(tmp_path, monkeypatch):
    cache = cache_at(tmp_path)
    cancelled = [False]
    atomic = prepared_audio._atomic_json
    def write_then_cancel(path, value):
        atomic(path, value)
        cancelled[0] = True
    monkeypatch.setattr(prepared_audio, "_atomic_json", write_then_cancel)
    def check():
        if cancelled[0]:
            raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        cache.borrow("source", {}, lambda: np.zeros(100, dtype=np.float32), cancel_check=check)
    assert not list(cache.root.glob("*/data"))
    assert not list(cache.root.glob("*/.building-*"))
    monkeypatch.setattr(prepared_audio, "_atomic_json", atomic)
    restored = cache.borrow("source", {}, lambda: np.ones(100, dtype=np.float32))
    assert not restored.reused
    restored.close()


def test_failed_access_time_update_releases_reference(tmp_path, monkeypatch):
    cache = cache_at(tmp_path)
    source = cache.borrow("source", {}, lambda: np.zeros(10, dtype=np.float32))
    def fail(*args):
        raise OSError("read-only timestamp")
    monkeypatch.setattr(prepared_audio.os, "utime", fail)
    with pytest.raises(OSError, match="timestamp"):
        source.close()
    assert source.handle is None
    assert cache.evict_source("source") == 1


def test_expired_dead_worker_lock_does_not_keep_source_alive(tmp_path):
    import subprocess
    import sys
    cache = cache_at(tmp_path, ttl_seconds=-1)
    script = """
import os, sys
import numpy as np
from prepared_audio import PreparedAudioCache
cache = PreparedAudioCache(sys.argv[1], min_free_bytes=0)
loan = cache.borrow('crashed', {}, lambda: np.zeros(100, dtype=np.float32))
os._exit(0)
"""
    subprocess.run([sys.executable, "-c", script, str(cache.root)], check=True)
    assert list(cache.root.glob("*/data"))
    cache.prune()
    assert not list(cache.root.glob("*/data"))


def test_context_atomic_publish_does_not_replace_ready_plan_on_cancel(tmp_path):
    cache = cache_at(tmp_path)
    loan = cache.borrow("source", {}, lambda: np.zeros(100, dtype=np.float32))
    cancelled = [False]
    def build():
        cancelled[0] = True
        return [(0, 30)]
    def check():
        if cancelled[0]:
            raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        loan.context({"vad": True}, build, check)
    assert not list((loan.directory / "data" / "plans").glob("*.json"))
    assert loan.context({"vad": True}, lambda: [(0, 30)]) == ([(0, 30)], False)
    loan.close()
