"""Physical-VRAM admission, batch sizing and CUDA OOM handling.

The broker only sees the leases it issued, so WhisperServer has to check the
physical card itself: scale the decode batch for long recordings, release the
models it owns before loading a heavy one, and turn a CTranslate2 allocation
failure into a retryable error instead of a raw 500.
"""

from types import SimpleNamespace

import main


def test_long_recordings_decode_with_a_smaller_batch():
    assert main.resolve_batch_size(2445.7) == main.LONG_AUDIO_BATCH_SIZE
    assert main.resolve_batch_size(2445.7) < main.resolve_batch_size(45.0)


def test_short_recordings_keep_the_default_batch():
    assert main.resolve_batch_size(45.0) == main.DEFAULT_BATCH_SIZE
    assert main.resolve_batch_size(0) == main.DEFAULT_BATCH_SIZE


def test_batch_size_never_drops_below_one():
    assert main.resolve_batch_size(100000.0) >= 1


def test_ctranslate2_oom_message_is_detected():
    assert main.is_cuda_oom(RuntimeError("CUDA failed with error out of memory"))
    assert main.is_cuda_oom(RuntimeError("CUDA out of memory"))


def test_unrelated_runtime_error_is_not_an_oom():
    assert not main.is_cuda_oom(RuntimeError("boom"))
    assert not main.is_cuda_oom(ValueError("bad payload"))


def test_gpu_oom_error_carries_a_retryable_code():
    error = main.GpuOutOfMemoryError("the GPU is busy")
    assert error.code == "gpu_out_of_memory"
    assert error.retryable is True


def test_release_cached_models_releases_and_drops_other_entries(monkeypatch):
    class FakeManaged:
        def __init__(self):
            self.released = False

        def release(self):
            self.released = True

    monkeypatch.setattr(main, "ManagedModel", FakeManaged)
    kept = FakeManaged()
    dropped = FakeManaged()
    cache = {"large-v3": kept, "gigaam-v3": dropped}
    main.release_cached_models(cache, keep="large-v3")
    assert list(cache) == ["large-v3"]
    assert dropped.released is True
    assert kept.released is False


def test_ensure_vram_releases_cached_models_when_the_card_is_full(monkeypatch):
    released = []

    class FakeManaged:
        def __init__(self, name):
            self.name = name

        def release(self):
            released.append(self.name)

    monkeypatch.setattr(main, "ManagedModel", FakeManaged)
    monkeypatch.setattr(main.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(main, "physical_free_vram_mb", lambda: 10.0)
    cache = {"gigaam-v3": FakeManaged("gigaam-v3")}
    main.ensure_vram_for_model("large-v3", cache)
    assert released == ["gigaam-v3"]
    assert cache == {}


def test_ensure_vram_keeps_models_when_the_card_has_room(monkeypatch):
    released = []

    class FakeManaged:
        def __init__(self, name):
            self.name = name

        def release(self):
            released.append(self.name)

    monkeypatch.setattr(main, "ManagedModel", FakeManaged)
    monkeypatch.setattr(main.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(main, "physical_free_vram_mb", lambda: 12000.0)
    cache = {"gigaam-v3": FakeManaged("gigaam-v3")}
    main.ensure_vram_for_model("large-v3", cache)
    assert released == []
    assert list(cache) == ["gigaam-v3"]


def _fake_community(monkeypatch):
    created = []

    class FakeCommunity:
        @classmethod
        def from_pretrained(cls, device="cpu"):
            created.append(device)
            return SimpleNamespace(device=device)

    monkeypatch.setattr(main, "Community1Model", FakeCommunity)
    return created


def test_diarization_cpu_is_an_explicit_opt_in(monkeypatch):
    created = _fake_community(monkeypatch)
    monkeypatch.setattr(main, "DIARIZATION_DEVICE", "cpu")
    monkeypatch.setattr(main.torch.cuda, "is_available", lambda: True)
    entry = main._create_diarization_entry()
    assert created == ["cpu"]
    assert entry.device == "cpu"


def test_diarization_gpu_keeps_the_broker_path(monkeypatch):
    monkeypatch.setattr(main, "DIARIZATION_DEVICE", "cuda")
    monkeypatch.setattr(main.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(main, "ManagedModel", lambda **kwargs: kwargs)
    entry = main._create_diarization_entry()
    assert entry["model_id"] == "whisper-community-1"
    assert entry["vram_mb"] == 1536
    assert entry["client_id"] == main.BROKER_CLIENT_ID


def test_diarization_auto_downgrades_only_when_the_card_is_full(monkeypatch):
    created = _fake_community(monkeypatch)
    monkeypatch.setattr(main, "DIARIZATION_DEVICE", "auto")
    monkeypatch.setattr(main, "physical_free_vram_mb", lambda: 10.0)
    entry = main._create_diarization_entry()
    assert created == ["cpu"]
    assert entry.device == "cpu"


def test_diarization_auto_uses_the_gpu_when_there_is_room(monkeypatch):
    monkeypatch.setattr(main, "DIARIZATION_DEVICE", "auto")
    monkeypatch.setattr(main, "physical_free_vram_mb", lambda: 12000.0)
    monkeypatch.setattr(main.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(main, "ManagedModel", lambda **kwargs: kwargs)
    entry = main._create_diarization_entry()
    assert entry["model_id"] == "whisper-community-1"
