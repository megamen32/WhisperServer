"""CPU regression for actual worker fairness at GigaAM chunk boundaries."""
import io
import queue
import threading
import wave
from types import SimpleNamespace

import numpy as np
import pytest

import gigaam_backend
import main


def wav_bytes(seconds, sample):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(np.full(int(seconds * 16000), sample, dtype="<i2").tobytes())
    return buffer.getvalue()


def fake_audio_decode(path, sampling_rate=16000):
    with wave.open(path, "rb") as source:
        assert source.getframerate() == sampling_rate
        return np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32) / 32768


@pytest.fixture
def worker_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("WHISPER_CHECKPOINT_DB", str(tmp_path / "checkpoints.sqlite"))
    monkeypatch.setenv("WHISPER_PCM_DIR", str(tmp_path / "pcm"))
    monkeypatch.setattr(gigaam_backend, "decode_audio", fake_audio_decode)
    monkeypatch.setattr(gigaam_backend, "get_speech_timestamps",
                        lambda audio, options: [{"start": 0, "end": len(audio)}])
    monkeypatch.setattr(main, "ManagedModel", None)
    monkeypatch.setattr(main, "append_metric", lambda metric: None)
    monkeypatch.setattr(main, "reset_cuda_peak_memory", lambda: None)
    monkeypatch.setattr(main, "peak_vram_mb", lambda: 0)
    monkeypatch.setattr(main, "ensure_vram_for_model", lambda *a: None)
    monkeypatch.setattr(main, "audio_duration_seconds",
                        lambda path: len(fake_audio_decode(path)) / 16000)
    monkeypatch.setattr(main, "transcription_cache", SimpleNamespace(set=lambda *a, **k: None))
    return tmp_path


def request(name, seconds, sample, attempt=1):
    return {"request_id": name, "audio_bytes": wav_bytes(seconds, sample), "model": "gigaam-v3",
            "vad_filter": True, "progress": True, "stream": False,
            "web_job_key": (name.split("-")[0], attempt)}


def run_worker(requests, stop_after_finals=1, on_event=None, cancel_flags=None):
    stop = threading.Event()
    class Responses(queue.Queue):
        finals = 0
        def put(self, event, *a, **k):
            super().put(event, *a, **k)
            if on_event:
                on_event(event, stop)
            if event.get("final") or "error" in event:
                self.finals += 1
                if self.finals >= stop_after_finals:
                    stop.set()
    responses = Responses()
    main.model_worker(requests, responses, stop, {}, cancel_flags)
    events = []
    while not responses.empty():
        events.append(responses.get_nowait())
    return events


def test_short_job_finishes_between_long_job_chunks(worker_runtime, monkeypatch):
    requests = queue.Queue()
    requests.put(request("long", 62, 6000))
    calls = []
    class Decoder:
        config = SimpleNamespace(_commit_hash="fixture-model-v1")
        def transcribe(self, path):
            audio = fake_audio_decode(path)
            name = "long" if audio[0] > .1 else "short"
            calls.append(name)
            if calls == ["long"]:
                requests.put(request("short", 5, 2000))
            return name
    monkeypatch.setattr(main, "_create_model_entry",
                        lambda name: gigaam_backend.GigaAMModel(Decoder()))
    events = run_worker(requests, stop_after_finals=2)
    finals = [event["request_id"] for event in events if event.get("final")]
    assert finals == ["short", "long"], "A short request must run before the long recording finishes"
    assert calls.index("short") < len(calls) - 1

def test_completed_chunk_is_reused_after_worker_restart(worker_runtime, monkeypatch):
    calls = []
    class Decoder:
        config = SimpleNamespace(_commit_hash="fixture-model-v1")
        def transcribe(self, path):
            calls.append(path)
            return "recognized"
    monkeypatch.setattr(main, "_create_model_entry",
                        lambda name: gigaam_backend.GigaAMModel(Decoder()))
    requests = queue.Queue()
    requests.put(request("meeting-old", 62, 6000))
    def interrupt_after_commit(event, stop):
        if "segment" in event:
            stop.set()
    first = run_worker(requests, on_event=interrupt_after_commit)
    assert len(calls) == 1 and not any(event.get("final") for event in first)
    assert not list((worker_runtime / "pcm").glob("gigaam-plan-*"))
    requests = queue.Queue()
    requests.put(request("meeting-new", 62, 6000, attempt=2))
    second = run_worker(requests)
    final = next(event["result"] for event in second if event.get("final"))
    assert final["resumed_chunks"] == 1
    assert final["computed_chunks"] == 3
    assert len(calls) == 4, "Only the three unfinished chunks should be decoded again"
    segments = [event["segment"] for event in second if "segment" in event]
    assert segments == final["segments"]
    assert [(segment["start"], segment["end"]) for segment in segments] == [
        (0, 20), (20, 40), (40, 60), (60, 62)]
    assert not list((worker_runtime / "pcm").glob("gigaam-plan-*"))


def test_changed_language_or_model_revision_does_not_reuse_chunks(worker_runtime, monkeypatch):
    calls = []
    revision = ["weights-v1"]
    class Decoder:
        def __init__(self):
            self.config = SimpleNamespace(_commit_hash=revision[0])
        def transcribe(self, path):
            calls.append(path)
            return "text"
    monkeypatch.setattr(main, "_create_model_entry",
                        lambda name: gigaam_backend.GigaAMModel(Decoder()))
    for index, language in enumerate([None, "en", None]):
        if index == 2:
            revision[0] = "weights-v2"
        item = request(f"same-{index}", 22, 6000, attempt=index + 1)
        item["language"] = language
        requests = queue.Queue()
        requests.put(item)
        events = run_worker(requests)
        final = next(event["result"] for event in events if event.get("final"))
        assert final["resumed_chunks"] == 0
    assert len(calls) == 6


def test_cancelled_attempt_does_not_publish_reused_chunks(worker_runtime, monkeypatch):
    calls = []
    class Decoder:
        config = SimpleNamespace(_commit_hash="fixture-model-v1")
        def transcribe(self, path):
            calls.append(path)
            return "text"
    monkeypatch.setattr(main, "_create_model_entry",
                        lambda name: gigaam_backend.GigaAMModel(Decoder()))
    requests = queue.Queue()
    requests.put(request("meeting-old", 22, 6000))
    run_worker(requests)
    requests = queue.Queue()
    requests.put(request("meeting-new", 22, 6000, attempt=2))
    events = run_worker(requests, cancel_flags={("meeting", 2): True})
    assert len(calls) == 2
    assert not any("segment" in event for event in events)
    assert any(event.get("error_code") == "web_job_cancelled" for event in events)


def test_gpu_scope_is_closed_before_scheduling_another_request(worker_runtime, monkeypatch):
    from contextlib import contextmanager
    entries = []
    class Decoder:
        config = SimpleNamespace(_commit_hash="fixture-model-v1")
        def transcribe(self, path):
            return "text"
    class Managed:
        def __init__(self):
            self.active = False
            self.model = gigaam_backend.GigaAMModel(Decoder())
            entries.append(self)
        def acquire(self):
            return self
        @contextmanager
        def inference(self):
            assert not self.active
            self.active = True
            try:
                yield self.model
            finally:
                self.active = False
        def touch(self, **kwargs):
            pass
        def release(self):
            assert not self.active
    monkeypatch.setattr(main, "ManagedModel", Managed)
    monkeypatch.setattr(main, "_create_model_entry", lambda name: Managed())
    requests = queue.Queue()
    requests.put(request("one", 22, 6000))
    requests.put(request("two", 5, 2000))
    def observe(event, stop):
        if "segment" in event:
            assert not any(entry.active for entry in entries), "A continuation must not retain a pinned model"
    events = run_worker(requests, stop_after_finals=2, on_event=observe)
    assert len([event for event in events if event.get("final")]) == 2
    assert not any(entry.active for entry in entries)


def test_global_diarization_is_reused_after_interruption(worker_runtime, monkeypatch):
    import diarization_backend
    diarizations = []
    monkeypatch.setattr(diarization_backend, "DIARIZATION_MODEL_REVISION", "pinned-diarizer-v1", raising=False)
    monkeypatch.setattr(diarization_backend, "DIARIZATION_ADAPTER_VERSION", "community1-v1", raising=False)
    def diarize(*args, **kwargs):
        diarizations.append(True)
        return [{"start": 0.0, "end": 42.0, "speaker": "SPEAKER_00"}]
    monkeypatch.setattr(main, "diarize_audio", diarize)
    class Decoder:
        config = SimpleNamespace(_commit_hash="fixture-model-v1")
        def transcribe(self, path):
            return "speech"
    monkeypatch.setattr(main, "_create_model_entry",
                        lambda name: gigaam_backend.GigaAMModel(Decoder()))
    first = request("talk-old", 42, 6000)
    first["diarize"] = True
    requests = queue.Queue()
    requests.put(first)
    run_worker(requests, on_event=lambda event, stop: stop.set() if "segment" in event and not event["segment"].get("provisional") else None)
    second = request("talk-new", 42, 6000, attempt=2)
    second["diarize"] = True
    requests = queue.Queue()
    requests.put(second)
    events = run_worker(requests)
    final = next(event["result"] for event in events if event.get("final"))
    assert len(diarizations) == 1
    assert final["diarization_reused"] and final["resumed_chunks"] == 1
    assert all(segment["speaker"] == "SPEAKER_00" for segment in final["segments"])


@pytest.mark.asyncio
async def test_web_jobs_are_admitted_before_the_first_job_completes(worker_runtime, monkeypatch):
    import asyncio
    entered = []
    release = asyncio.Event()
    monkeypatch.setattr(main, "WEBUI_JOB_DIR", worker_runtime / "jobs")
    async def transcribe(file, *args, **kwargs):
        entered.append(file.filename)
        await release.wait()
        return {"text": file.filename, "segments": []}
    monkeypatch.setattr(main, "_transcribe_impl", transcribe)
    tasks = []
    for name in ["long.wav", "short.wav"]:
        job = main._job_store().create("owner", {"filename": name, "size": 3, "model": "gigaam-v3",
                                                "diarize": False})
        main._job_spool().append(job["id"], 0, b"abc", 3)
        main._job_store().enqueue("owner", job["id"], 1)
        tasks.append(asyncio.create_task(main._run_web_job("owner", job["id"], 1)))
    try:
        for _ in range(50):
            if len(entered) == 2:
                break
            await asyncio.sleep(.01)
        assert entered == ["long.wav", "short.wav"]
    finally:
        release.set()
        await asyncio.gather(*tasks)

def test_diarization_hook_observes_cancel_inside_global_pass(worker_runtime, monkeypatch):
    monkeypatch.setenv("WHISPER_EARLY_TEXT_ENABLED", "0")
    flags = {}
    phases = []
    def diarize(path, *, vad_filter, model_provider, progress_hook):
        progress_hook("segmentation", total=3, completed=1)
        flags[("meeting", 1)] = True
        progress_hook("segmentation", total=3, completed=2)
        pytest.fail("Cancellation must interrupt the global pass at its next hook")
    monkeypatch.setattr(main, "diarize_audio", diarize)
    monkeypatch.setattr(main, "_create_model_entry", lambda name: pytest.fail("ASR must not start after cancellation"))
    item = request("meeting", 22, 6000)
    item["diarize"] = True
    requests = queue.Queue()
    requests.put(item)
    events = run_worker(requests, cancel_flags=flags)
    assert any(event.get("progress", {}).get("step") == "segmentation" for event in events)
    assert any(event.get("error_code") == "web_job_cancelled" for event in events)
    assert not any("segment" in event for event in events)


@pytest.mark.asyncio
async def test_stale_worker_generation_defers_a_new_attempt(monkeypatch):
    retried = {"id": "job", "attempt": 2, "status": "queued"}
    calls = []
    monkeypatch.setattr(main.app.state, "worker_generation", 2, raising=False)
    monkeypatch.setattr(main, "_job_store", lambda: SimpleNamespace(
        interrupt_for_retry=lambda owner, job_id, attempt: retried))
    monkeypatch.setattr(main, "_defer_or_launch_web_job",
                        lambda owner, job: calls.append((owner, job)), raising=False)
    await main._run_web_job("owner", "job", 1, generation=1)
    assert calls == [("owner", retried)]

def test_complete_result_cache_is_versioned_with_gigaam_weights(monkeypatch):
    before = main.transcription_cache_key("native", "gigaam-v3", None, "audio", vad_filter=True)
    monkeypatch.setattr(gigaam_backend, "GIGAAM_V3_REVISION", "new-immutable-revision")
    after = main.transcription_cache_key("native", "gigaam-v3", None, "audio", vad_filter=True)
    assert before != after
