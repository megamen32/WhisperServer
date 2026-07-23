"""Black-box audio tests against a running WhisperServer instance."""

from __future__ import annotations

import io
import os
import random
import shutil
import struct
import subprocess
import wave
from pathlib import Path
from typing import Generator

import httpx
import pytest


pytestmark = pytest.mark.integration


def _enabled(value: str | None) -> bool:
    """Return whether an environment flag is enabled."""
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _wav_bytes(samples: list[int], sample_rate: int = 16_000) -> bytes:
    """Encode mono signed-16-bit PCM samples as a WAV file."""
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))
    return output.getvalue()


def _silence_audio(duration_seconds: float = 1.5, sample_rate: int = 16_000) -> bytes:
    """Generate digital silence."""
    return _wav_bytes([0] * int(duration_seconds * sample_rate), sample_rate)


def _noise_audio(duration_seconds: float = 1.5, sample_rate: int = 16_000) -> bytes:
    """Generate low-amplitude deterministic white noise for the VAD path."""
    generator = random.Random(20260723)
    amplitude = 500
    samples = [generator.randint(-amplitude, amplitude) for _ in range(int(duration_seconds * sample_rate))]
    return _wav_bytes(samples, sample_rate)


def _normalized_words(text: str) -> set[str]:
    """Extract simple case-folded words for tolerant TTS assertions."""
    return {word.strip(".,!?;:\"'()[]{}") for word in text.casefold().split()}


@pytest.fixture(scope="module")
def blackbox_client() -> Generator[httpx.Client, None, None]:
    """Connect to the configured server or skip the opt-in integration suite."""
    if not _enabled(os.getenv("BLACKBOX_TESTS")):
        pytest.skip("set BLACKBOX_TESTS=1 to run tests against a live WhisperServer")

    base_url = os.getenv("BLACKBOX_BASE_URL", "http://127.0.0.1:7653").rstrip("/")
    api_key = os.getenv("BLACKBOX_API_KEY", os.getenv("API_KEY", "bad-key"))
    client = httpx.Client(
        base_url=base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=httpx.Timeout(180.0, connect=5.0),
    )
    try:
        response = client.get("/status")
    except httpx.HTTPError as exc:
        client.close()
        pytest.skip(f"black-box server is unavailable: {exc}")
    if response.status_code != 200:
        client.close()
        pytest.skip(f"black-box server returned /status={response.status_code}")

    yield client
    client.close()


def _transcribe(client: httpx.Client, audio: bytes, model: str) -> str:
    """Submit an audio payload through the public OpenAI-compatible endpoint."""
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("blackbox.wav", audio, "audio/wav")},
        data={"model": model, "language": "en", "vad_filter": "true"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert isinstance(payload.get("text"), str), payload
    return payload["text"].strip()


@pytest.fixture(scope="module")
def blackbox_model(blackbox_client: httpx.Client) -> str:
    """Return the model used by the black-box audio cases."""
    model = os.getenv("BLACKBOX_MODEL", "parakeet-v3")
    response = blackbox_client.get("/v1/models")
    assert response.status_code == 200, response.text
    available = {entry["id"] for entry in response.json().get("data", [])}
    assert model in available, f"{model!r} is not advertised by /v1/models: {sorted(available)}"
    return model


@pytest.fixture(scope="module")
def tts_audio(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    """Synthesize a short English sentence with the local espeak executable."""
    executable = shutil.which("espeak") or shutil.which("espeak-ng")
    if executable is None:
        pytest.skip("espeak or espeak-ng is required for the TTS black-box test")

    output_path = Path(tmp_path_factory.mktemp("tts")) / "speech.wav"
    subprocess.run(
        [executable, "-v", "en", "-w", str(output_path), "This is a speech recognition test."],
        check=True,
        capture_output=True,
        text=True,
    )
    return output_path.read_bytes()


def test_tts_synthesis_is_transcribed(blackbox_client: httpx.Client, blackbox_model: str, tts_audio: bytes) -> None:
    """Verify the end-to-end path from local TTS synthesis to transcription."""
    text = _transcribe(blackbox_client, tts_audio, blackbox_model)
    expected = _normalized_words("This is a speech recognition test")
    matched = expected & _normalized_words(text)

    assert len(matched) >= 3, f"expected at least 3 TTS words in transcription, got {text!r}"


def test_silence_is_transcribed_as_silence(blackbox_client: httpx.Client, blackbox_model: str) -> None:
    """Verify that digital silence does not create hallucinated speech."""
    assert _transcribe(blackbox_client, _silence_audio(), blackbox_model) == ""


def test_noise_is_transcribed_as_silence(blackbox_client: httpx.Client, blackbox_model: str) -> None:
    """Verify that low-level white noise is suppressed by VAD/decoder guards."""
    assert _transcribe(blackbox_client, _noise_audio(), blackbox_model) == ""
