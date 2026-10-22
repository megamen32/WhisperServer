"""Opt-in client attribution stored next to the transcription cache.

A client may attach a free-form comment ("вебморда", "video_watching") to a
transcription request. The comment never influences the transcription itself:
the audio hash still decides the cache key, so the same file stays a cache hit
no matter who sends it. The comment is only remembered beside the cached
result, which tells an operator which client sent which file.

Requests without a comment behave exactly as before.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from diskcache import Cache

logger = logging.getLogger(__name__)

MAX_COMMENT_LENGTH = 200
MAX_RECORDS_PER_FILE = 50
ATTRIBUTION_TTL_SECONDS = 60 * 60 * 24 * 90
ATTRIBUTION_SIZE_LIMIT = 256 * 1024 ** 2


def normalize_comment(raw) -> Optional[str]:
    """Return a trimmed comment, or None when the client opted out."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if len(text) > MAX_COMMENT_LENGTH:
        text = text[:MAX_COMMENT_LENGTH]
    return text


class AttributionStore:
    """Persistent map from audio hash and comment to the clients that sent it.

    Two indexes are kept: every file remembers the comments it arrived with,
    every comment remembers the files it was sent with.
    """

    def __init__(self, directory=None, ttl=ATTRIBUTION_TTL_SECONDS, size_limit=ATTRIBUTION_SIZE_LIMIT):
        self._directory = directory or os.getenv("WHISPER_ATTRIBUTION_DIR", "whisper_attribution")
        self._ttl = ttl
        self._cache = Cache(self._directory, size_limit=size_limit)

    @staticmethod
    def _file_key(audio_hash: str) -> str:
        return f"file:{audio_hash}"

    @staticmethod
    def _tag_key(comment: str) -> str:
        return f"tag:{comment}"

    def record(self, audio_hash: str, comment, requested_model=None) -> Optional[dict]:
        """Remember that this client sent this file with this comment."""
        clean = normalize_comment(comment)
        if not audio_hash or not clean:
            return None
        now = time.time()
        entry = {"comment": clean, "ts": now, "requested_model": requested_model}
        try:
            key = self._file_key(audio_hash)
            records = list(self._cache.get(key) or [])
            records.append(entry)
            self._cache.set(key, records[-MAX_RECORDS_PER_FILE:], expire=self._ttl)

            tag_key = self._tag_key(clean)
            files = dict(self._cache.get(tag_key) or {})
            file_entry = files.get(audio_hash) or {
                "audio_hash": audio_hash,
                "count": 0,
                "first_ts": now,
                "last_ts": now,
            }
            file_entry["count"] += 1
            file_entry["last_ts"] = now
            files[audio_hash] = file_entry
            self._cache.set(tag_key, files, expire=self._ttl)
        except Exception:
            logger.exception("Could not record client attribution")
            return None
        return entry

    def lookup(self, audio_hash: Optional[str] = None, comment: Optional[str] = None, limit: int = 100) -> dict:
        """Return what is known about a file or about a client comment."""
        try:
            if audio_hash:
                return {
                    "audio_hash": audio_hash,
                    "records": list(self._cache.get(self._file_key(audio_hash)) or []),
                }
            if comment:
                clean = normalize_comment(comment)
                files = self._cache.get(self._tag_key(clean)) or {} if clean else {}
                ordered = sorted(files.values(), key=lambda item: item.get("last_ts", 0), reverse=True)
                return {"comment": clean, "files": ordered[:limit]}
        except Exception:
            logger.exception("Could not read client attribution")
        return {}

    def close(self) -> None:
        try:
            self._cache.close()
        except Exception:
            logger.debug("Could not close the attribution store", exc_info=True)
