"""Tests for the public model registry contract."""

from model_registry import MODEL_PRIORITY, MODEL_SUPERSEDES


def test_parakeet_is_advertised_as_a_local_model():
    """The model list source must expose Parakeet to API clients."""
    assert "parakeet-v3" in MODEL_PRIORITY


def test_large_v3_can_serve_large_v2_and_tiny_can_use_base():
    """Cover the explicit compatibility rules requested by the API contract."""
    assert "large-v2" in MODEL_SUPERSEDES["large-v3"]
    assert "tiny" in MODEL_SUPERSEDES["base"]
