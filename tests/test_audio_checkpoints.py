"""CPU-only durable completed-unit identity, expiry and storage bounds."""
import sqlite3
import pytest
from audio_checkpoints import AudioCheckpoints, computation_key


def test_completed_unit_is_immutable_and_survives_reopening(tmp_path):
    path = tmp_path / "chunks.sqlite"
    key = computation_key("audio", "gigaam", "revision", {"language": "ru", "chunk": [0, 20]})
    store = AudioCheckpoints(path)
    assert store.get(key, 0) is None
    value = {"segments": [{"start": 0, "end": 20, "text": "complete"}]}
    store.put(key, 0, value)
    assert AudioCheckpoints(path).get(key, 0) == value
    assert store.put(key, 0, {"segments": []}) == value
    assert store.get(key, 0) == value
    assert computation_key("audio", "gigaam", "revision", {"chunk": [0, 20], "language": "ru"}) == key
    assert computation_key("audio", "gigaam", "revision2", {"language": "ru", "chunk": [0, 20]}) != key


def test_checkpoint_quota_and_expiry_reclaim_old_units(tmp_path, monkeypatch):
    import audio_checkpoints
    clock = [1000.0]
    monkeypatch.setattr(audio_checkpoints.time, "time", lambda: clock[0])
    store = AudioCheckpoints(tmp_path / "chunks.sqlite", ttl_seconds=10, max_units=2,
                             max_bytes=80, max_payload_bytes=60)
    for index in range(5):
        clock[0] += 1
        store.put("audio", index, {"text": "x" * 30})
    with sqlite3.connect(store.path) as db:
        count, size = db.execute("SELECT count(*),coalesce(sum(payload_bytes),0) FROM audio_units").fetchone()
        assert count <= 2 and size <= 80
        assert db.execute("SELECT units,bytes FROM audio_usage WHERE id=1").fetchone() == (count, size)
    clock[0] += 11
    assert store.get("audio", 4) is None
    reopened = AudioCheckpoints(store.path, ttl_seconds=10, max_units=2, max_bytes=80)
    with sqlite3.connect(reopened.path) as db:
        assert db.execute("SELECT count(*) FROM audio_units").fetchone()[0] == 0
    with pytest.raises(ValueError):
        store.put("audio", 8, {"text": "x" * 100})


def test_parameter_and_source_changes_have_distinct_computation_keys():
    base = {"vad": True, "beam": 5, "speaker_turns": [{"start": 0, "end": 20, "speaker": "A"}]}
    key = computation_key("file1", "model", "weights1", base)
    assert key != computation_key("file2", "model", "weights1", base)
    assert key != computation_key("file1", "model", "weights2", base)
    assert key != computation_key("file1", "model", "weights1", {**base, "beam": 1})
    assert key != computation_key("file1", "model", "weights1", {**base, "speaker_turns": []})
