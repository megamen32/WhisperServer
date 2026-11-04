"""Fast unit: resumable upload replay, boundaries, and finite disk quota.

Purpose: preserve the exact file after a lost response and reject conflicting
or oversized writes. Expected <1s; maximum 5s.
"""
import pytest

from webui_uploads import UploadConflict, UploadSpool


def test_upload_replay_and_resume(tmp_path):
    spool = UploadSpool(tmp_path, max_file_bytes=8, max_chunk_bytes=4, max_total_bytes=8)
    assert spool.append("job1", 0, b"abcd", 8) == 4
    assert spool.append("job1", 0, b"abcd", 8) == 4
    with pytest.raises(UploadConflict):
        spool.append("job1", 0, b"xxxx", 8)
    with pytest.raises(UploadConflict):
        spool.append("job1", 5, b"x", 8)
    assert spool.append("job1", 4, b"efgh", 8) == 8
    assert (tmp_path / "job1").read_bytes() == b"abcdefgh"
    with pytest.raises(ValueError):
        spool.append("job2", 0, b"x", 1)
    spool.remove("job1")
    assert spool.append("job2", 0, b"x", 1) == 1
