"""Model priorities and conservative same-family substitution rules."""

from __future__ import annotations


MODEL_PRIORITY = {
    "tiny": 1,
    "base": 2,
    "small": 3,
    "medium": 4,
    "distil-large-v3": 5,
    "large-v3": 6,
    "large-v2": 7,
    "large": 8,
    "parakeet-v3": 9,
}

# Only same-family Whisper models are interchangeable. Parakeet has different
# language and decoding guarantees, so it is intentionally excluded.
MODEL_SUPERSEDES = {
    "base": {"tiny"},
    "small": {"tiny", "base"},
    "medium": {"tiny", "base", "small"},
    "distil-large-v3": {"tiny", "base", "small", "medium"},
    "large-v2": {"tiny", "base", "small", "medium", "distil-large-v3", "large"},
    "large": {"tiny", "base", "small", "medium", "distil-large-v3"},
    "large-v3": {
        "tiny",
        "base",
        "small",
        "medium",
        "distil-large-v3",
        "large-v2",
        "large",
    },
}


def model_priority(model_id: str) -> int:
    """Return queue/model-selection priority; unknown models sort last."""
    return MODEL_PRIORITY.get(model_id, 999)


def select_compatible_loaded_model(requested_model: str, model_cache: dict) -> str | None:
    """Return a loaded compatible model that explicitly supersedes a request."""
    if requested_model in model_cache:
        return None
    candidates = [
        model_id
        for model_id in model_cache
        if requested_model in MODEL_SUPERSEDES.get(model_id, set())
    ]
    return max(candidates, key=model_priority) if candidates else None
