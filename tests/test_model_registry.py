"""Tests for model selection and the public model registry contract."""

import main
from model_registry import MODEL_PRIORITY, MODEL_SUPERSEDES, WHISPER_MODEL_IDS


def test_parakeet_is_advertised_as_a_local_model():
    """The model list source must expose Parakeet to API clients."""
    assert "parakeet-v3" in MODEL_PRIORITY
    assert "gigaam-v3" in MODEL_PRIORITY


def test_large_v3_can_serve_large_v2_and_tiny_can_use_base():
    """Cover the explicit compatibility rules requested by the API contract."""
    assert "large-v2" in MODEL_SUPERSEDES["large-v3"]
    assert "tiny" in MODEL_SUPERSEDES["base"]


def test_large_v3_is_the_strongest_automatic_whisper_model():
    """Automatic Whisper selection must prefer Large V3 over legacy variants."""
    assert WHISPER_MODEL_IDS[-1] == "large-v3"
    assert "parakeet-v3" not in WHISPER_MODEL_IDS
    assert "gigaam-v3" not in WHISPER_MODEL_IDS


def test_cpu_model_loading_bypasses_broker_when_cuda_is_unavailable(monkeypatch):
    """A missing GPU must not leave CPU requests waiting for a broker lease."""
    class UnexpectedManagedModel:
        def __init__(self, *args, **kwargs):
            raise AssertionError("CPU loading must bypass ManagedModel")

    monkeypatch.setattr(main, "ManagedModel", UnexpectedManagedModel)
    monkeypatch.setattr(main.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        main,
        "_load_model",
        lambda model_name, device, compute_type: (model_name, device, compute_type),
    )

    assert main._create_model_entry("large-v3") == ("large-v3", "cpu", "int8")


def test_gpu_model_loading_bypasses_missing_broker_client(monkeypatch):
    """A working GPU must remain usable when the optional broker client is absent."""
    monkeypatch.setattr(main, "ManagedModel", None)
    monkeypatch.setattr(main.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        main,
        "_load_model",
        lambda model_name, device, compute_type: (model_name, device, compute_type),
    )

    assert main._create_model_entry("large-v3") == ("large-v3", "cuda", "float16")


def test_openai_alias_uses_cpu_fallback_without_cuda(monkeypatch):
    """The automatic alias must select a responsive CPU model without a GPU."""
    monkeypatch.setattr(main.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(main, "CPU_FALLBACK_MODEL", "small")

    assert main._select_openai_whisper_model({}) == "small"
