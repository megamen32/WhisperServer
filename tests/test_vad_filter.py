"""Request-scoped VAD behavior."""

from main import (
    WHISPER_VAD_MIN_SILENCE_MS,
    WHISPER_VAD_MIN_SPEECH_MS,
    WHISPER_VAD_SPEECH_PAD_MS,
    WHISPER_VAD_THRESHOLD,
    app,
    build_vad_transcribe_options,
    transcription_cache_key,
)


def test_vad_is_enabled_by_default_in_all_public_endpoints():
    schema = app.openapi()

    native = schema["paths"]["/transcribe"]["post"]["parameters"]
    web = schema["paths"]["/web/transcribe"]["post"]["parameters"]
    openai_form = schema["components"]["schemas"]["Body_openai_transcribe_v1_audio_transcriptions_post"]

    assert next(p for p in native if p["name"] == "vad_filter")["schema"]["default"] is True
    assert next(p for p in web if p["name"] == "vad_filter")["schema"]["default"] is True
    assert openai_form["properties"]["vad_filter"]["default"] is True


def test_enabled_vad_passes_configured_silero_parameters():
    options = build_vad_transcribe_options(True)

    assert options == {
        "vad_filter": True,
        "vad_parameters": {
            "threshold": WHISPER_VAD_THRESHOLD,
            "min_speech_duration_ms": WHISPER_VAD_MIN_SPEECH_MS,
            "min_silence_duration_ms": WHISPER_VAD_MIN_SILENCE_MS,
            "speech_pad_ms": WHISPER_VAD_SPEECH_PAD_MS,
        },
    }


def test_disabled_vad_reaches_whisper_without_vad_parameters():
    assert build_vad_transcribe_options(False) == {
        "vad_filter": False,
        "vad_parameters": None,
    }


def test_cache_keys_are_separate_for_vad_enabled_and_disabled():
    enabled = transcription_cache_key(
        "openai", "medium", "ru", "abc123", vad_filter=True
    )
    disabled = transcription_cache_key(
        "openai", "medium", "ru", "abc123", vad_filter=False
    )

    assert enabled != disabled
    assert ":vad-v1:1:" in enabled
    assert ":vad-v1:0:" in disabled
