"""Tests for the Telegram streaming fallback response contract."""

import pytest

from whisperclient.transcriber import _server_stream_payload


def test_cli_stream_payload_matches_the_server_contract():
    assert _server_stream_payload({"type": "info", "language": "ru"}) is None
    assert _server_stream_payload({"type": "segment", "text": "часть"}) == {
        "segment": {"type": "segment", "text": "часть"}
    }
    assert _server_stream_payload({"type": "result", "text": "готово"}) == {
        "result": {"type": "result", "text": "готово"}
    }


def test_stream_error_triggers_the_existing_fallback():
    with pytest.raises(RuntimeError, match="temporary broker failure"):
        _server_stream_payload({"error": "temporary broker failure"})
