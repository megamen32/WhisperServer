"""Fast unit/focused integration, expected 12s/max 45s, no weights or GPU.

Detects speaker changes lost inside ASR segments, mixed caches, dropped API
fields, unlabelled streaming segments, and GigaAM merging a whole dialogue.
"""
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import main
import gigaam_backend
from diarization_backend import attribute_segment, speaker_for_interval, speaker_text
from metrics import transcription_response_payload

TURNS = [
    {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
    {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"},
    {"start": 3.0, "end": 4.0, "speaker": "SPEAKER_00"},
]


def test_gated_model_error_explains_required_access():
    from diarization_backend import _load_pipeline
    from huggingface_hub.errors import GatedRepoError
    class Gated:
        @staticmethod
        def from_pretrained(*a, **k):
            raise GatedRepoError('restricted model')
    with pytest.raises(RuntimeError, match='Community-1.*HF_TOKEN'):
        _load_pipeline(Gated)


def test_word_changes_split_segment_and_preserve_returning_speaker():
    words = [dict(start=0.1, end=0.8, word=" Привет!"),
             dict(start=1.1, end=1.8, word=" Здравствуйте!"),
             dict(start=3.1, end=3.8, word=" Как дела?")]
    segments = attribute_segment(dict(start=0, end=4, text="Привет! Здравствуйте! Как дела?"), words, TURNS)
    assert [s["speaker"] for s in segments] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert [(s["start"], s["end"]) for s in segments] == [(0.1, 0.8), (1.1, 1.8), (3.1, 3.8)]
    assert speaker_text(segments) == "Спикер 1: Привет!\nСпикер 2: Здравствуйте!\nСпикер 1: Как дела?"
    assert [w["speaker"] for w in words] == [s["speaker"] for s in segments]


def test_overlap_and_unknown_do_not_invent_identity():
    assert speaker_for_interval(0.9, 1.8, TURNS) == "SPEAKER_01"
    assert speaker_for_interval(2.2, 2.8, TURNS) is None
    assert speaker_for_interval(1, 1, TURNS) == "SPEAKER_01"
    assert attribute_segment(dict(start=2.2, end=2.8, text="Неясно"), [], TURNS)[0]["speaker"] is None


def test_parakeet_tokens_keep_spaces_when_speaker_changes():
    words = [dict(start=.1, end=.3, word="Привет"), dict(start=.4, end=.8, word="мир"),
             dict(start=1.1, end=1.8, word="Пока")]
    groups = attribute_segment(dict(start=0, end=2, text="Привет мир Пока"), words, TURNS)
    assert [s["text"] for s in groups] == ["Привет мир", "Пока"]


def test_diarization_cache_is_separate_and_old_keys_stay_valid():
    args = ("native", "large-v3", "ru", "hash")
    plain = main.transcription_cache_key(*args, vad_filter=True, words=False)
    diarized = main.transcription_cache_key(*args, vad_filter=True, words=False, diarize=True)
    assert plain == "native:vad-v1:1:large-v3:ru:False:hash"
    assert plain != diarized


@pytest.mark.parametrize("stream", [False, True])
def test_worker_returns_labelled_segments_and_words(monkeypatch, stream):
    options = []
    class ASR:
        def transcribe(self, *args, **kwargs):
            options.append(kwargs)
            words = [SimpleNamespace(start=.1, end=.8, word=" Привет"),
                     SimpleNamespace(start=1.1, end=1.8, word=" Пока")]
            return [SimpleNamespace(start=0, end=2, text="Привет Пока", words=words)], SimpleNamespace(language="ru", language_probability=1)
    monkeypatch.setattr(main, "diarize_audio", lambda p: TURNS)
    monkeypatch.setattr(main, "_create_model_entry", lambda p: ASR())
    monkeypatch.setattr(main, "ManagedModel", None)
    monkeypatch.setattr(main, "audio_duration_seconds", lambda p: 4)
    monkeypatch.setattr(main, "append_metric", lambda m: None)
    monkeypatch.setattr(main, "reset_cuda_peak_memory", lambda: None)
    monkeypatch.setattr(main, "peak_vram_mb", lambda: 0)
    requests, responses = Queue(), Queue()
    requests.put(dict(request_id="test", model="large-v3", audio_bytes=b"audio", diarize=True, words=True, stream=stream))
    checks = iter([False, True])
    main.model_worker(requests, responses, SimpleNamespace(is_set=lambda: next(checks)), {})
    messages = []
    while not responses.empty():
        messages.append(responses.get_nowait())
    result = messages[-1]["result"]
    assert options[0]["word_timestamps"] is True
    assert [s["speaker"] for s in result["segments"]] == ["SPEAKER_00", "SPEAKER_01"]
    assert result["speaker_text"] == "Спикер 1: Привет\nСпикер 2: Пока"
    assert result["words"][1]["speaker"] == "SPEAKER_01"
    if stream:
        assert [m["segment"]["speaker"] for m in messages[:-1]] == ["SPEAKER_00", "SPEAKER_01"]
        assert [len(m["words"]) for m in messages[:-1]] == [1, 1]


def test_gigaam_transcribes_each_turn_with_native_offsets(monkeypatch):
    monkeypatch.setattr(gigaam_backend, "decode_audio", lambda *a, **k: np.zeros(4 * 16000))
    monkeypatch.setattr(gigaam_backend, "get_speech_timestamps", lambda *a: pytest.fail("turns already define speech"))
    calls = []
    class Decoder:
        def transcribe(self, path):
            calls.append(path)
            return "реплика"
    segments, _ = gigaam_backend.GigaAMModel(Decoder()).transcribe("audio.wav", speaker_turns=TURNS)
    assert [(s.start, s.end) for s in segments] == [(0, 1), (1, 2), (3, 4)]
    assert len(calls) == 3


@pytest.mark.parametrize("cached", [False, True])
def test_openai_fresh_and_cached_diarization(monkeypatch, cached):
    monkeypatch.setenv("API_KEY", "DIARIZATION_TEST_ONLY")
    monkeypatch.setattr(main, "pending_results", {})
    result = dict(text="Привет", speaker_text="Спикер 1: Привет", speaker_turns=TURNS,
                  diarization_model="pyannote/speaker-diarization-community-1",
                  segments=[dict(start=0, end=1, text="Привет", speaker="SPEAKER_00")])
    monkeypatch.setattr(main, "_get_cached_result", lambda k: result if cached else None)
    monkeypatch.setattr(main, "append_cache_metric", lambda *a: None)
    def put(request):
        assert request["diarize"] is True
        assert "community-1-v1" in request["cache_key"]
        main.pending_results.pop(request["request_id"]).set_result(result)
    monkeypatch.setattr(main.app.state, "request_queue", SimpleNamespace(put=put), raising=False)
    response = TestClient(main.app).post("/v1/audio/transcriptions",
        headers={"Authorization": "Bearer DIARIZATION_TEST_ONLY"},
        files={"file": ("voice.wav", b"fixture", "audio/wav")},
        data={"diarize": "true", "response_format": "verbose_json", "model": "large-v3"})
    assert response.status_code == 200
    assert response.json()["segments"][0]["speaker"] == "SPEAKER_00"
    assert response.json()["speaker_turns"] == TURNS
    assert transcription_response_payload(result)["speaker_text"] == "Спикер 1: Привет"


def test_native_cached_stream_keeps_word_speaker_labels(monkeypatch):
    result = dict(text="Привет", speaker_text="Спикер 1: Привет",
                  segments=[dict(start=0, end=1, text="Привет", speaker="SPEAKER_00")],
                  words=[dict(start=.1, end=.8, word="Привет", speaker="SPEAKER_00")])
    monkeypatch.setattr(main, "_get_cached_result", lambda k: result)
    monkeypatch.setattr(main, "append_cache_metric", lambda *a: None)
    monkeypatch.setattr(main, "ALLOWED_API_KEYS", {"DIARIZATION_TEST_ONLY"})
    response = TestClient(main.app).post("/transcribe",
        params=dict(model="large-v3", stream="true", words="true", diarize="true", api_key="DIARIZATION_TEST_ONLY"),
        files={"file": ("voice.wav", b"fixture", "audio/wav")})
    import json
    messages = [json.loads(line) for line in response.text.splitlines()]
    assert messages[0]["words"][0]["speaker"] == "SPEAKER_00"
    assert messages[-1]["result"]["speaker_text"] == "Спикер 1: Привет"


def test_openai_stream_surfaces_diarization_failure(monkeypatch):
    monkeypatch.setenv("API_KEY", "DIARIZATION_TEST_ONLY")
    monkeypatch.setattr(main, "pending_streams", {})
    monkeypatch.setattr(main, "_get_cached_result", lambda k: None)
    def put(request):
        queue = main.pending_streams[request["request_id"]]
        queue.put_nowait({"error": "Community-1 недоступна"})
        queue.put_nowait(None)
    monkeypatch.setattr(main.app.state, "request_queue", SimpleNamespace(put=put), raising=False)
    response = TestClient(main.app).post("/v1/audio/transcriptions",
        headers={"Authorization": "Bearer DIARIZATION_TEST_ONLY"},
        files={"file": ("voice.wav", b"fixture", "audio/wav")},
        data={"diarize": "true", "stream": "true", "model": "large-v3"})
    import json
    event = json.loads(response.text.removeprefix("data: ").strip())
    assert event == {"type": "error", "error": {"message": "Community-1 недоступна"}}
    assert not main.pending_streams
