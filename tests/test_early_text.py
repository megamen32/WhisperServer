"""Bounded draft timing, final-quality identity and reusable preparation."""
import queue
from types import SimpleNamespace

import pytest

import gigaam_backend
import main
from tests.test_worker_chunks import worker_runtime, fake_audio_decode, request, run_worker


@pytest.mark.parametrize("cold_seconds", [0, 30])
def test_preliminary_text_precedes_global_diarization_without_changing_final(worker_runtime, monkeypatch, cold_seconds):
    clock, timeline = [0], []
    class Decoder:
        config = SimpleNamespace(_commit_hash="early-text-model-v1")
        def transcribe(self, path):
            clock[0] += 2
            return f"samples {len(fake_audio_decode(path))}"
    def create(name):
        clock[0] += cold_seconds
        return gigaam_backend.GigaAMModel(Decoder())
    def diarize(*args, **kwargs):
        timeline.append(("diarization", clock[0]))
        clock[0] += 20
        return [{"start": 0.0, "end": 9.0, "speaker": "SPEAKER_00"},
                {"start": 9.0, "end": 42.0, "speaker": "SPEAKER_01"}]
    monkeypatch.setattr(main, "_create_model_entry", create)
    monkeypatch.setattr(main, "diarize_audio", diarize)
    item = request("draft", 42, 6000)
    item["diarize"] = True
    requests = queue.Queue()
    requests.put(item)
    def observe(event, stop):
        if "segment" in event:
            timeline.append(("preliminary" if event["segment"].get("provisional") else "final", clock[0]))
    events = run_worker(requests, on_event=observe)
    draft = [entry for entry in timeline if entry[0] == "preliminary"]
    assert draft, f"First text still waits for full diarization: {timeline}"
    assert len(draft) == 1
    assert timeline.index(draft[0]) < next(i for i, entry in enumerate(timeline) if entry[0] == "diarization")
    assert draft[0][1] == cold_seconds + 2
    result = next(event["result"] for event in events if event.get("final"))
    assert [(s["start"], s["end"], s["text"], s["speaker"]) for s in result["segments"]] == [
        (0.0, 9.0, "samples 144000", "SPEAKER_00"),
        (9.0, 29.0, "samples 320000", "SPEAKER_01"),
        (29.0, 42.0, "samples 208000", "SPEAKER_01"),
    ]
    assert not any(s.get("provisional") for s in result["segments"])
    final_reset = [e["progress"] for e in events if e.get("progress", {}).get("reset")
                   and e["progress"].get("phase") == "transcribing"]
    assert final_reset and final_reset[-1].get("provisional") is False


def test_restart_reuses_completed_pcm_and_vad_preparation(worker_runtime, monkeypatch):
    preparations = []
    def decode(path, sampling_rate=16000):
        preparations.append(path)
        return fake_audio_decode(path, sampling_rate)
    monkeypatch.setattr(gigaam_backend, "decode_audio", decode)
    class Decoder:
        config = SimpleNamespace(_commit_hash="prepared-audio-v1")
        def transcribe(self, path):
            return "recognized"
    monkeypatch.setattr(main, "_create_model_entry", lambda name: gigaam_backend.GigaAMModel(Decoder()))
    first = queue.Queue()
    first.put(request("meeting-first", 42, 6000))
    run_worker(first, on_event=lambda event, stop: stop.set() if "segment" in event else None)
    second = queue.Queue()
    second.put(request("meeting-second", 42, 6000, attempt=2))
    events = run_worker(second)
    assert len(preparations) == 1, "Completed audio preparation was decoded again after restart"
    result = next(event["result"] for event in events if event.get("final"))
    assert result["resumed_chunks"] == 1


def test_cancel_after_preview_keeps_draft_out_of_final_and_skips_diarization(worker_runtime, monkeypatch):
    flags = {}
    class Decoder:
        config = SimpleNamespace(_commit_hash="preview-cancel-v1")
        def transcribe(self, path):
            return "early text"
    monkeypatch.setattr(main, "_create_model_entry", lambda name: gigaam_backend.GigaAMModel(Decoder()))
    monkeypatch.setattr(main, "diarize_audio", lambda *a, **k: pytest.fail("Diarization began after cancellation"))
    item = request("cancel", 42, 6000)
    item["diarize"] = True
    source_hash = main.hash_bytes(item["audio_bytes"])
    requests = queue.Queue()
    requests.put(item)
    def cancel(event, stop):
        if event.get("segment", {}).get("provisional"):
            flags[("cancel", 1)] = True
    events = run_worker(requests, cancel_flags=flags, on_event=cancel)
    assert len([event for event in events if "segment" in event]) == 1
    assert any(event.get("error_code") == "web_job_cancelled" for event in events)
    assert not any("result" in event for event in events)
    assert not list((worker_runtime / "pcm").glob("gigaam-plan-*"))
    from prepared_audio import PreparedAudioCache
    cache = PreparedAudioCache(worker_runtime / "pcm" / "cache-v1")
    assert cache.evict_source(source_hash) == 1


def test_cancel_after_preview_decode_never_publishes_late_text(worker_runtime, monkeypatch):
    flags = {}
    class Decoder:
        config = SimpleNamespace(_commit_hash="preview-cancel-v2")
        def transcribe(self, path):
            flags[("cancel", 1)] = True
            return "too late"
    monkeypatch.setattr(main, "_create_model_entry", lambda name: gigaam_backend.GigaAMModel(Decoder()))
    monkeypatch.setattr(main, "diarize_audio", lambda *a, **k: pytest.fail("Diarization began after cancellation"))
    item = request("cancel", 42, 6000)
    item["diarize"] = True
    requests = queue.Queue()
    requests.put(item)
    events = run_worker(requests, cancel_flags=flags)
    assert not any("segment" in event for event in events)
    assert any(event.get("error_code") == "web_job_cancelled" for event in events)
    assert not list((worker_runtime / "pcm").glob("gigaam-plan-*"))


def test_worker_marks_each_active_quantum_and_clears_execution_while_idle(worker_runtime, monkeypatch):
    import threading
    snapshots = []
    class State(dict):
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            if key == "snapshot":
                snapshots.append(dict(value))
    state = State(snapshot={"generation": 7, "phase": "idle"})
    class Decoder:
        config = SimpleNamespace(_commit_hash="execution-v1")
        def transcribe(self, path):
            assert state["snapshot"]["phase"] == "inference"
            assert state["snapshot"]["request_id"] == "active"
            assert state["snapshot"]["checkpoint"] >= 3
            return "ready"
    monkeypatch.setattr(main, "_create_model_entry", lambda name: gigaam_backend.GigaAMModel(Decoder()))
    requests, stop = queue.Queue(), threading.Event()
    requests.put(request("active", 22, 6000))
    class Responses(queue.Queue):
        def put(self, event):
            super().put(event)
            if event.get("final") or event.get("error"):
                assert state["snapshot"]["request_id"] == "active"
                stop.set()
    main.model_worker(requests, Responses(), stop, {}, {}, state)
    assert state["snapshot"]["phase"] == "idle"
    assert "request_id" not in state["snapshot"]
    assert any(value.get("checkpoint") == 5 for value in snapshots)
    assert any(value.get("unit") == "finalizing" for value in snapshots)
    assert all(value["generation"] == 7 for value in snapshots)


def test_plan_close_releases_borrow_when_temporary_cleanup_fails():
    closed = []
    plan = object.__new__(gigaam_backend.GigaAMChunkPlan)
    plan.samples = SimpleNamespace(_mmap=SimpleNamespace(close=lambda: closed.append("memmap")))
    def fail():
        raise OSError("temporary cleanup")
    plan.directory = SimpleNamespace(cleanup=fail)
    plan._prepared = SimpleNamespace(close=lambda: closed.append("borrow"))
    with pytest.raises(OSError, match="temporary"):
        plan.close()
    assert closed == ["memmap", "borrow"]
    assert plan._prepared is None
