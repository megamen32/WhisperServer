"""Tests for explicit loaded-model substitution rules."""

from model_registry import select_compatible_loaded_model


def test_loaded_stronger_whisper_model_can_serve_weaker_request():
    """Use only an explicitly declared stronger model from the same family."""
    loaded = {"large-v3": object(), "medium": object()}

    assert select_compatible_loaded_model("base", loaded) == "large-v3"
    assert select_compatible_loaded_model("large-v3", loaded) is None
    assert select_compatible_loaded_model("large-v2", {"large-v3": object()}) == "large-v3"
    assert select_compatible_loaded_model("tiny", {"base": object()}) == "base"


def test_incompatible_parakeet_model_is_never_substituted():
    """Keep different backend and language guarantees separate."""
    assert select_compatible_loaded_model("base", {"parakeet-v3": object()}) is None
