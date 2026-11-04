import asyncio
import importlib.util
import multiprocessing as mp
import signal
import subprocess
import sys
import tempfile
import secrets
import threading
import sqlite3
import queue as thread_queue
from uuid import uuid4
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Query, Form, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse, Response
from markdown_it import MarkdownIt
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import logging
import traceback
import time
import os
import gc
import hashlib
import hmac
import weakref
from contextlib import asynccontextmanager, contextmanager, ExitStack
from faster_whisper import WhisperModel
from gigaam_backend import GigaAMModel
from parakeet_backend import ParakeetModel
from metrics import diarized_segment
from diarization_backend import diarize_audio, attribute_segment, speaker_text, Community1Model
try:
    from cudabroker_client import ManagedModel
except Exception:
    ManagedModel = None  # type: ignore
from diskcache import Cache
from dotenv import load_dotenv
import torch
from metrics import (
    append_cache_metric,
    append_metric,
    audio_duration_seconds,
    configure_metrics,
    model_metadata,
    peak_vram_mb,
    reset_cuda_peak_memory,
    transcription_response_payload,
)
from model_registry import MODEL_PRIORITY, MODEL_SUPERSEDES, WHISPER_MODEL_IDS
from model_registry import select_compatible_loaded_model as _select_compatible_loaded_model
from attribution import AttributionStore, normalize_comment
from webui_jobs import JobStore, JobNotFound, InvalidTransition, CapacityExceeded
from webui_uploads import UploadSpool, UploadConflict

load_dotenv()

SUMMARY_MARKDOWN = MarkdownIt("commonmark", {"html": False, "linkify": False}).enable("table")
_summary_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="web-summary")
_summary_slots = threading.BoundedSemaphore(2)
_summary_cache_locks = weakref.WeakValueDictionary()


def _web_asset(name, *, template=False):
    """Locate web assets in a checkout/release or an installed wheel."""
    local = Path(__file__).parent / ("templates" if template else "") / name
    return local if local.is_file() else Path(sys.prefix) / "share" / "whisperserver" / name


async def _summary_open(opener, upstream):
    """Limit blocking gateway calls without occupying the event loop or a queue."""
    if not _summary_slots.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="Summariser is busy, try later")

    def fetch():
        with opener.open(upstream, timeout=OMNIROUTE_SUMMARY_TIMEOUT) as response:
            return response.read().decode("utf-8")

    try:
        future = _summary_executor.submit(fetch)
    except BaseException:
        _summary_slots.release()
        raise
    future.add_done_callback(lambda _: _summary_slots.release())
    return await asyncio.wrap_future(future)


async def _run_byok(config, system, prompt):
    """Run the BYOK adapter with credentials in a private stdin pipe."""
    if not _summary_slots.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="Суммаризация занята, попробуйте позже")
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            "node", str(_web_asset("byok_bridge.mjs")),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "NODE_OPTIONS": "--max-old-space-size=192"},
        )
        data = json.dumps({"config": config, "system": system, "prompt": prompt}).encode()
        stdout, _ = await asyncio.wait_for(process.communicate(data), timeout=250)
        result = json.loads(stdout)
        if process.returncode or not isinstance(result.get("text"), str) or not result["text"].strip():
            raise HTTPException(status_code=502, detail="Не удалось получить ответ провайдера. Проверьте ключ, адрес и модель.")
        return result["text"].strip()
    except (OSError, ValueError, asyncio.TimeoutError):
        raise HTTPException(status_code=502, detail="Не удалось подключиться к провайдеру")
    finally:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        _summary_slots.release()

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CPU_FALLBACK_MODEL = os.getenv("CPU_FALLBACK_MODEL", "small")
if CPU_FALLBACK_MODEL not in WHISPER_MODEL_IDS:
    logger.warning("Unknown CPU_FALLBACK_MODEL=%s; falling back to small", CPU_FALLBACK_MODEL)
    CPU_FALLBACK_MODEL = "small"
DEFAULT_MODEL = os.getenv("MODEL") or (
    "gigaam-v3" if torch.cuda.is_available() else CPU_FALLBACK_MODEL
)
METRICS_JSONL_PATH = os.getenv("WHISPER_METRICS_JSONL", "whisper_metrics.jsonl")
configure_metrics(METRICS_JSONL_PATH)

BROKER_CLIENT_ID = os.getenv("CUDABROKER_CLIENT_ID", "whisper")
BROKER_TTL_SECONDS = float(os.getenv("CUDABROKER_WHISPER_TTL_SECONDS", "900"))
BROKER_VRAM_MB = {
    "tiny": 800,
    "base": 1000,
    "small": 1600,
    "medium": 3200,
    "distil-large-v3": 4200,
    "large-v3": 5200,
    "large-v2": 5200,
    "large": 5200,
    "parakeet-v3": 2800,
    "gigaam-v3": 2000,
}
BROKER_CPU_CAPABLE = {"tiny", "base", "small", "parakeet-v3", "gigaam-v3"}


def _broker_vram_mb(model_id: str) -> int:
    override = os.getenv(f"CUDABROKER_VRAM_MB_{model_id.replace('-', '_').upper()}")
    if override:
        return int(float(override))
    return int(BROKER_VRAM_MB.get(model_id, 5200))


def _broker_cpu_capable(model_id: str) -> bool:
    override = os.getenv(f"CUDABROKER_CPU_CAPABLE_{model_id.replace('-', '_').upper()}")
    if override is not None:
        return _is_truthy(override)
    return model_id in BROKER_CPU_CAPABLE


# --- Physical GPU admission -------------------------------------------------
# The broker only accounts for the leases it issued. Other GPU tenants (ollama,
# the TTS server, captcha solvers) are invisible to it, so a granted lease does
# not guarantee the memory is really there. Admission therefore checks the
# physical device, and the decode batch is scaled down for long recordings.
VRAM_HEADROOM_MB = int(os.getenv("WHISPER_VRAM_HEADROOM_MB", "768"))
RETRY_BEAM_SIZE = int(os.getenv("WHISPER_RETRY_BEAM_SIZE", "1"))
DIARIZATION_DEVICE = os.getenv("WHISPER_DIARIZATION_DEVICE", "cuda").strip().lower()


class GpuOutOfMemoryError(RuntimeError):
    """The physical GPU could not satisfy a transcription request."""

    def __init__(self, message: str, *, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable
        self.code = "gpu_out_of_memory"


def physical_free_vram_mb() -> float | None:
    """Free VRAM on the physical device, ignoring tenants the broker cannot see."""
    try:
        if not torch.cuda.is_available():
            return None
        free_bytes, _total = torch.cuda.mem_get_info()
        return free_bytes / (1024 * 1024)
    except Exception:
        logger.debug("mem_get_info failed", exc_info=True)
        return None


def is_cuda_oom(exc: BaseException) -> bool:
    """Detect a GPU allocation failure raised by torch or by CTranslate2.

    CTranslate2 raises a native ``RuntimeError("CUDA failed with error out of
    memory")`` that never reaches torch's exception type.
    """
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    text = str(exc).lower()
    return "out of memory" in text or ("cuda failed with error" in text and "memory" in text)


def release_cached_models(model_cache: dict, keep: str | None = None) -> None:
    """Release cached models so their VRAM can be reused by the next request."""
    for name, entry in list(model_cache.items()):
        if keep is not None and name == keep:
            continue
        if ManagedModel is not None and isinstance(entry, ManagedModel):
            try:
                entry.release()
            except Exception:
                logger.warning("Failed to release cached model %s", name, exc_info=True)
        model_cache.pop(name, None)
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        logger.debug("empty_cache failed", exc_info=True)


def ensure_vram_for_model(model_name: str, model_cache: dict) -> None:
    """Free what we control before the broker hands out a GPU lease.

    The broker grants leases from its own registry and cannot see the unmanaged
    tenants sharing the card, so a lease may be granted while the memory is
    already gone. Evict our own cached models first: that is the part we own.
    """
    if ManagedModel is None or not torch.cuda.is_available():
        return
    required = _broker_vram_mb(model_name) + VRAM_HEADROOM_MB
    free = physical_free_vram_mb()
    if free is None or free >= required:
        return
    logger.warning(
        "Free VRAM %.0f MiB is below the %.0f MiB requested for %s; releasing cached models",
        free,
        required,
        model_name,
    )
    release_cached_models(model_cache, keep=model_name)
    free = physical_free_vram_mb()
    if free is not None and free < required:
        logger.warning(
            "Free VRAM is still %.0f MiB after releasing cached models (%.0f MiB requested)",
            free,
            required,
        )


ALLOWED_API_KEYS = {os.getenv("API_KEY", "bad-key")}
WEBUI_SESSION_COOKIE = "whisper_web_session"
WEBUI_CSRF_HEADER = "x-whisper-csrf"
WEBUI_SESSION_TTL_SECONDS = int(os.getenv("WEBUI_SESSION_TTL_SECONDS", "2592000"))
_webui_sessions = {}

# Job state is file-backed; configure this path on persistent storage.
WEBUI_JOB_DIR = Path(os.getenv("WEBUI_JOB_DIR", "whisper_cache/webui_jobs"))
WEBUI_MAX_FILE_BYTES = int(os.getenv("WEBUI_MAX_FILE_BYTES", str(256 * 1024 * 1024)))
WEBUI_MAX_CHUNK_BYTES = 2 * 1024 * 1024
WEBUI_MAX_SPOOL_BYTES = int(os.getenv("WEBUI_MAX_SPOOL_BYTES", str(2 * 1024 ** 3)))
_job_tasks = set()
_job_lock = asyncio.Lock()
_job_worker_slot = asyncio.Semaphore(1)


def _job_store():
    WEBUI_JOB_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    return JobStore(WEBUI_JOB_DIR / "jobs.sqlite", max_payload_bytes=8_000_000)


def _job_spool():
    return UploadSpool(WEBUI_JOB_DIR / "uploads", WEBUI_MAX_FILE_BYTES,
                       WEBUI_MAX_CHUNK_BYTES, WEBUI_MAX_SPOOL_BYTES)


def _session_db():
    WEBUI_JOB_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    db = sqlite3.connect(WEBUI_JOB_DIR / "sessions.sqlite", timeout=5)
    db.execute("BEGIN IMMEDIATE")
    db.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, token TEXT NOT NULL, expires REAL NOT NULL)")
    existed = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='csrf_tokens'").fetchone()
    db.execute("CREATE TABLE IF NOT EXISTS csrf_tokens (session_id TEXT NOT NULL, token TEXT NOT NULL, expires REAL NOT NULL, PRIMARY KEY(session_id, token))")
    if not existed:
        db.execute("INSERT OR IGNORE INTO csrf_tokens SELECT id, token, expires FROM sessions WHERE expires > ?", (time.time(),))
    db.commit()
    return db


def _save_webui_session(sid):
    session = _webui_sessions[sid]
    with _session_db() as db:
        db.execute("INSERT OR REPLACE INTO sessions VALUES (?, ?, ?)", (sid, session['token'], session['expires']))
        db.execute("DELETE FROM csrf_tokens WHERE expires <= ?", (time.time(),))
        db.execute("INSERT INTO csrf_tokens VALUES (?, ?, ?)", (sid, session['token'], session['expires']))


def _load_webui_session(sid):
    if not sid:
        return None
    with _session_db() as db:
        row = db.execute("SELECT token, expires FROM sessions WHERE id=? AND expires>?", (sid, time.time())).fetchone()
    if row:
        session = {'token': row[0], 'expires': row[1]}
        _webui_sessions[sid] = session
        return session
    return None

# --- Web UI transcript store -------------------------------------------------
# The browser never receives the summariser credential and never sends the
# transcript back for summarisation. The server keeps the text under an opaque
# id and the browser only sends that id; the server then forwards the text to
# OmniRoute itself. Ids are unguessable and bound to the session that produced
# them, so one visitor cannot ask for another visitor's transcript.
WEBUI_TRANSCRIPT_TTL_SECONDS = int(os.getenv("WEBUI_TRANSCRIPT_TTL_SECONDS", "7200"))
WEBUI_TRANSCRIPT_MAX_CHARS = int(os.getenv("WEBUI_TRANSCRIPT_MAX_CHARS", "2000000"))
WEBUI_TRANSCRIPT_MAX_ENTRIES = int(os.getenv("WEBUI_TRANSCRIPT_MAX_ENTRIES", "64"))
_webui_transcripts = {}

# The public form must not become a way to spend the owner's model budget:
# summarisation is rate limited per session and in total.
WEBUI_SUMMARY_MAX_PER_SESSION = int(os.getenv("WEBUI_SUMMARY_MAX_PER_SESSION", "8"))
WEBUI_SUMMARY_MAX_PER_HOUR = int(os.getenv("WEBUI_SUMMARY_MAX_PER_HOUR", "60"))
WEBUI_SUMMARY_WINDOW_SECONDS = 3600
_webui_summary_calls = []

# --- Summariser (OmniRoute) --------------------------------------------------
# Credentials live only in the server-side environment file, never in the page.
OMNIROUTE_BASE_URL = os.getenv("OMNIROUTE_BASE_URL", "http://192.168.2.100:20128/v1").rstrip("/")
# The key file lives next to the service environment, outside the repository.
# The release directory is a git archive, so the file stays in the working tree.
OMNIROUTE_API_KEY_FILE_CANDIDATES = [
    path
    for path in (
        os.getenv("OMNIROUTE_API_KEY_FILE_DEFAULT", "").strip(),
        str(Path(__file__).parent / ".env.omniroute"),
        "/home/roomhacker/PycharmProjects/WhisperServer/.env.omniroute",
    )
    if path
]
OMNIROUTE_SUMMARY_MODEL = os.getenv("OMNIROUTE_SUMMARY_MODEL", "minimax-combo-m3.1")
# A dead upstream provider must not kill the button: try the requested model
# first and fall back to the owner's other MiniMax routes.
OMNIROUTE_SUMMARY_FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv(
        "OMNIROUTE_SUMMARY_FALLBACK_MODELS",
        "minimax/MiniMax-M3.1-Flash,minimax-combo-m3",
    ).split(",")
    if model.strip()
]
OMNIROUTE_SUMMARY_TIMEOUT = int(os.getenv("OMNIROUTE_SUMMARY_TIMEOUT", "300"))
OMNIROUTE_SUMMARY_MAX_CHARS = int(os.getenv("OMNIROUTE_SUMMARY_MAX_CHARS", "120000"))
OMNIROUTE_SUMMARY_MAX_TOKENS = int(os.getenv("OMNIROUTE_SUMMARY_MAX_TOKENS", "8000"))
SUMMARY_SYSTEM_PROMPT = (
    "Ты — помощник, который делает краткие выжимки расшифровок аудио на русском языке. "
    "Пиши по делу, структурно, без воды и без выдуманных фактов."
)


def _omniroute_api_key():
    """Read the summariser key from the server-side environment file."""
    candidates = []
    configured = os.getenv("OMNIROUTE_API_KEY_FILE", "").strip()
    if configured:
        candidates.append(configured)
    candidates.extend(OMNIROUTE_API_KEY_FILE_CANDIDATES)
    for key_file in candidates:
        if not key_file:
            continue
        try:
            with open(key_file, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("OMNIROUTE_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
                    return line
        except OSError:
            continue
    if candidates:
        logger.warning("No OmniRoute key found in %s", candidates)
    return ""


def _prune_webui_transcripts(now=None):
    now = now or time.time()
    stale = [tid for tid, data in _webui_transcripts.items() if data.get("expires", 0) < now]
    for tid in stale:
        _webui_transcripts.pop(tid, None)
    if len(_webui_transcripts) > WEBUI_TRANSCRIPT_MAX_ENTRIES:
        oldest = sorted(_webui_transcripts.items(), key=lambda item: item[1].get("created", 0))
        for tid, _ in oldest[: len(_webui_transcripts) - WEBUI_TRANSCRIPT_MAX_ENTRIES]:
            _webui_transcripts.pop(tid, None)


def _store_webui_transcript(session_id, text):
    """Keep the transcript text server-side and return its opaque id."""
    _prune_webui_transcripts()
    if not text:
        return ""
    transcript_id = secrets.token_urlsafe(24)
    _webui_transcripts[transcript_id] = {
        "session_id": session_id,
        "text": text[:WEBUI_TRANSCRIPT_MAX_CHARS],
        "created": time.time(),
        "expires": time.time() + WEBUI_TRANSCRIPT_TTL_SECONDS,
    }
    return transcript_id


def _get_webui_transcript(session_id, transcript_id):
    _prune_webui_transcripts()
    entry = _webui_transcripts.get(transcript_id or "")
    if not entry:
        try:
            job = _job_store().get(session_id, transcript_id)
            return _transcript_text_from_payload(job['result']) if job['status'] == 'done' else None
        except JobNotFound:
            return None
    # The transcript belongs to the session that produced it.
    if entry.get("session_id") != session_id:
        return None
    return entry.get("text") or ""


def _summary_rate_limit(session_id):
    """Keep the public form from spending the owner's model budget."""
    now = time.time()
    cutoff = now - WEBUI_SUMMARY_WINDOW_SECONDS
    for key in [k for k in _webui_summary_calls if k[1] < cutoff]:
        _webui_summary_calls.remove(key)
    per_session = sum(1 for key in _webui_summary_calls if key[0] == session_id)
    if per_session >= WEBUI_SUMMARY_MAX_PER_SESSION:
        raise HTTPException(status_code=429, detail="Too many summaries for this session")
    if len(_webui_summary_calls) >= WEBUI_SUMMARY_MAX_PER_HOUR:
        raise HTTPException(status_code=429, detail="Summariser is busy, try later")
    _webui_summary_calls.append((session_id, now))


def _new_webui_transcript(session_id):
    """Reserve an opaque id before the stream starts; the text arrives later."""
    _prune_webui_transcripts()
    transcript_id = secrets.token_urlsafe(24)
    _webui_transcripts[transcript_id] = {
        "session_id": session_id,
        "text": "",
        "created": time.time(),
        "expires": time.time() + WEBUI_TRANSCRIPT_TTL_SECONDS,
    }
    return transcript_id


def _fill_webui_transcript(transcript_id, text):
    """Attach the transcript text to an already reserved id."""
    if not text:
        return
    entry = _webui_transcripts.get(transcript_id or "")
    if not entry:
        return
    entry["text"] = text[:WEBUI_TRANSCRIPT_MAX_CHARS]


def _transcript_text_from_payload(payload):
    """Pull the full transcript text out of one NDJSON stream payload."""
    if not isinstance(payload, dict):
        return ""
    result = payload.get("result")
    if isinstance(result, dict):
        payload = result
    return payload.get("speaker_text") or payload.get("text") or ""


async def _capture_transcript_stream(source, transcript_id):
    """Pass the stream through untouched while collecting the transcript text."""
    buffer = b""
    try:
        async for chunk in source:
            yield chunk
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8")
            buffer += chunk
            while True:
                index = buffer.find(b"\n")
                if index == -1:
                    break
                line, buffer = buffer[:index], buffer[index + 1:]
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line.decode("utf-8", "replace"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                text = _transcript_text_from_payload(parsed)
                if text:
                    _fill_webui_transcript(transcript_id, text)
                    buffer = b""
                    break
    finally:
        pass


def _new_webui_token():
    return secrets.token_urlsafe(32)


def _prune_webui_sessions(now=None):
    now = now or time.time()
    stale = [sid for sid, data in _webui_sessions.items() if data.get("expires", 0) < now]
    for sid in stale:
        _webui_sessions.pop(sid, None)


def _create_webui_session():
    now = time.time()
    sid = secrets.token_urlsafe(32)
    token = _new_webui_token()
    _webui_sessions[sid] = {"token": token, "expires": now + WEBUI_SESSION_TTL_SECONDS}
    _save_webui_session(sid)
    return sid, token


def _rotate_webui_token(sid):
    token = _new_webui_token()
    _webui_sessions[sid] = {"token": token, "expires": time.time() + WEBUI_SESSION_TTL_SECONDS}
    _save_webui_session(sid)
    return token


def _same_origin_request(request: Request):
    origin = request.headers.get("origin")
    if not origin:
        return False
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme).split(",", 1)[0].strip()
    host = request.headers.get("x-forwarded-host", request.headers.get("host", "")).split(",", 1)[0].strip()
    expected = f"{scheme}://{host}"
    return origin.rstrip("/") == expected.rstrip("/")


def _set_webui_session_cookie(response: Response, request: Request, sid: str) -> None:
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme).split(",", 1)[0].strip()
    response.set_cookie(
        WEBUI_SESSION_COOKIE,
        sid,
        max_age=WEBUI_SESSION_TTL_SECONDS,
        httponly=True,
        secure=(scheme == "https"),
        samesite="strict",
    )


def _consume_webui_token(request: Request):
    _prune_webui_sessions()
    sid = request.cookies.get(WEBUI_SESSION_COOKIE, "")
    supplied = request.headers.get(WEBUI_CSRF_HEADER, "")
    session = _load_webui_session(sid)
    if not sid or not supplied or not session:
        raise HTTPException(status_code=403, detail="Invalid or expired web UI token")
    if not _same_origin_request(request):
        _webui_sessions.pop(sid, None)
        with _session_db() as db:
            db.execute("DELETE FROM sessions WHERE id=?", (sid,))
            db.execute("DELETE FROM csrf_tokens WHERE session_id=?", (sid,))
        raise HTTPException(status_code=403, detail="Invalid request origin")
    with _session_db() as db:
        consumed = db.execute("DELETE FROM csrf_tokens WHERE session_id=? AND token=? AND expires>?",
                              (sid, supplied, time.time())).rowcount
    if not consumed:
        raise HTTPException(status_code=403, detail="Invalid or expired web UI token")
    token = _rotate_webui_token(sid)
    request.state.webui_next_csrf = token
    return token


# Map OpenAI model names to local speech-to-text models.
OPENAI_WHISPER_ALIAS = "whisper-1"
OPENAI_WHISPER_INTERNAL_MODEL = "__openai_whisper_1__"
OPENAI_DEFAULT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "gigaam-v3")
if OPENAI_DEFAULT_MODEL not in MODEL_PRIORITY:
    logger.warning("Unknown OPENAI_DEFAULT_MODEL=%s; falling back to gigaam-v3", OPENAI_DEFAULT_MODEL)
    OPENAI_DEFAULT_MODEL = "gigaam-v3"

# Speech filtering defaults. Silero VAD runs before decoding, so silence and
# short noise bursts never reach Whisper. The decoder thresholds provide a
# second guard against low-confidence hallucinations.
WHISPER_VAD_ENABLED = os.getenv("WHISPER_VAD_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
WHISPER_VAD_THRESHOLD = float(os.getenv("WHISPER_VAD_THRESHOLD", "0.50"))
WHISPER_VAD_MIN_SPEECH_MS = int(os.getenv("WHISPER_VAD_MIN_SPEECH_MS", "250"))
WHISPER_VAD_MIN_SILENCE_MS = int(os.getenv("WHISPER_VAD_MIN_SILENCE_MS", "700"))
WHISPER_VAD_SPEECH_PAD_MS = int(os.getenv("WHISPER_VAD_SPEECH_PAD_MS", "200"))
WHISPER_NO_SPEECH_THRESHOLD = float(os.getenv("WHISPER_NO_SPEECH_THRESHOLD", "0.60"))
WHISPER_LOG_PROB_THRESHOLD = float(os.getenv("WHISPER_LOG_PROB_THRESHOLD", "-1.0"))
WHISPER_COMPRESSION_RATIO_THRESHOLD = float(os.getenv("WHISPER_COMPRESSION_RATIO_THRESHOLD", "2.4"))
OPENAI_MODEL_MAP = {
    OPENAI_WHISPER_ALIAS: (OPENAI_WHISPER_INTERNAL_MODEL
                           if OPENAI_DEFAULT_MODEL in WHISPER_MODEL_IDS
                           else OPENAI_DEFAULT_MODEL),
}


def build_vad_transcribe_options(enabled: bool) -> dict:
    """Return faster-whisper kwargs for request-scoped VAD control."""
    if not enabled:
        return {"vad_filter": False, "vad_parameters": None}
    return {
        "vad_filter": True,
        "vad_parameters": {
            "threshold": WHISPER_VAD_THRESHOLD,
            "min_speech_duration_ms": WHISPER_VAD_MIN_SPEECH_MS,
            "min_silence_duration_ms": WHISPER_VAD_MIN_SILENCE_MS,
            "speech_pad_ms": WHISPER_VAD_SPEECH_PAD_MS,
        },
    }


def transcription_cache_key(prefix: str, model: str, language: str, audio_hash: str, *, vad_filter: bool, words: bool | None = None, diarize: bool = False) -> str:
    """Build a cache key that never mixes VAD-enabled and raw decoding."""
    words_part = "" if words is None else f":{words}"
    routed_model = OPENAI_MODEL_MAP.get(model, model)
    if model == OPENAI_WHISPER_ALIAS and routed_model != OPENAI_WHISPER_INTERNAL_MODEL:
        model = f"{model}@{routed_model}"
    speaker_part = ":community-1-v2" if diarize else ""
    return f"{prefix}:vad-v1:{int(vad_filter)}:{model}:{language}{words_part}:{audio_hash}{speaker_part}"


def _select_openai_whisper_model(model_cache: dict) -> str:
    """Reuse the strongest loaded Whisper model, or load Large V3 by default."""
    loaded_models = [model_id for model_id in model_cache if model_id in WHISPER_MODEL_IDS]
    if loaded_models:
        return max(loaded_models, key=MODEL_PRIORITY.get)
    if not torch.cuda.is_available():
        logger.info(
            "CUDA unavailable; resolving %s to CPU fallback model %s",
            OPENAI_WHISPER_ALIAS,
            CPU_FALLBACK_MODEL,
        )
        return CPU_FALLBACK_MODEL
    return OPENAI_DEFAULT_MODEL


def _load_model(model_name: str, device: str, compute_type: str):
    """Load a registered model using its native inference backend."""
    if model_name == "parakeet-v3":
        return ParakeetModel.from_pretrained(device=device)
    if model_name == "gigaam-v3":
        return GigaAMModel.from_pretrained(device=device)
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def _create_model_entry(model_name: str):
    """Create a broker-managed entry only when CUDA can actually be used."""
    cuda_available = torch.cuda.is_available()
    if ManagedModel is not None and cuda_available:
        return ManagedModel(
            model_id=f"whisper-{model_name}",
            loader=lambda mn=model_name: _load_model(
                mn,
                device="cuda",
                compute_type="float16",
            ),
            vram_mb=_broker_vram_mb(model_name),
            gpu_priority=MODEL_PRIORITY.get(model_name, 0),
            cpu_capable=_broker_cpu_capable(model_name),
            ttl_seconds=BROKER_TTL_SECONDS,
            cpu_fallback=(
                lambda mn=model_name: _load_model(mn, device="cpu", compute_type="int8")
            ) if _broker_cpu_capable(model_name) else None,
            client_id=BROKER_CLIENT_ID,
        )

    if cuda_available:
        logger.warning(
            "cudabroker_client unavailable; loading %s directly on GPU.",
            model_name,
        )
        return _load_model(model_name, device="cuda", compute_type="float16")

    logger.warning(
        "CUDA unavailable; loading %s directly on CPU without a broker lease.",
        model_name,
    )
    return _load_model(model_name, device="cpu", compute_type="int8")



DIARIZATION_VRAM_MB = 1536


def _create_diarization_entry():
    device = DIARIZATION_DEVICE
    if device == "auto":
        # The diarization pipeline shares the card with the ASR model. Keep it on
        # the CPU when the physical device has no room for it.
        free = physical_free_vram_mb()
        device = "cuda" if (free is None or free >= DIARIZATION_VRAM_MB + VRAM_HEADROOM_MB) else "cpu"
    if device == "cpu":
        return Community1Model.from_pretrained("cpu")
    if not torch.cuda.is_available():
        return Community1Model.from_pretrained("cpu")
    if ManagedModel is None:
        raise RuntimeError("Community-1 GPU requires cudabroker_client")
    return ManagedModel(
        model_id="whisper-community-1",
        loader=lambda: Community1Model.from_pretrained("cuda"),
        vram_mb=DIARIZATION_VRAM_MB,
        gpu_priority=MODEL_PRIORITY.get("gigaam-v3", 10) + 1,
        cpu_capable=False,
        ttl_seconds=900,
        client_id=BROKER_CLIENT_ID,
    )


@contextmanager
def _diarization_model_session():
    """Keep the lease active for inference; free it before the ASR stage."""
    entry = _create_diarization_entry()
    model = None
    scope = ExitStack()
    try:
        if ManagedModel is not None and isinstance(entry, ManagedModel):
            entry.acquire()
            model = scope.enter_context(entry.inference())
        else:
            model = entry
        logger.info("Community-1 inference on %s", model.device)
        yield model
    finally:
        try:
            if model is not None:
                model.unload()
        except Exception as exc:
            # Inactive exception frames otherwise retain the CUDA model itself.
            traceback.clear_frames(exc.__traceback__)
            raise
        finally:
            model = None
            try:
                scope.close()
            finally:
                if ManagedModel is not None and isinstance(entry, ManagedModel):
                    entry.release()
                entry = None


def _is_truthy(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _openai_sse(event):
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _openai_delta_event(text, segment_id=None):
    event = {"type": "transcript.text.delta", "delta": text or ""}
    if segment_id is not None:
        event["segment_id"] = str(segment_id)
    return event


def _openai_done_event(text, metadata: dict | None = None):
    event = {"type": "transcript.text.done", "text": text or ""}
    if metadata:
        event.update(metadata)
    return event

def _get_openai_model_ids():
    """Return OpenAI-compatible model ids: alias first, then concrete local models."""
    ids = ["whisper-1"]
    for model_id in MODEL_PRIORITY.keys():
        if model_id not in ids:
            ids.append(model_id)
    return ids


def _require_openai_api_key(request: Request):
    """Validate OpenAI-compatible Authorization: Bearer <API_KEY> or X-API-Key."""
    expected = os.getenv("API_KEY", "").strip()
    if not expected:
        return

    auth_header = request.headers.get("authorization", "").strip()
    supplied = ""
    if auth_header.lower().startswith("bearer "):
        supplied = auth_header[7:].strip()
    if not supplied:
        supplied = request.headers.get("x-api-key", "").strip()

    if supplied != expected:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Invalid API key",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
        )

# Cache with 10GB limit and 1 month TTL for entries
CACHE_TTL = 60 * 60 * 24 * 30
transcription_cache = Cache("whisper_cache", size_limit=10 * 1024 ** 3)
# Keep the short name as a module-level compatibility reference. Runtime code
# uses the explicit name so form fields cannot shadow the disk cache object.
cache = transcription_cache
# Opt-in client comments ("вебморда", "video_watching") live next to the
# transcription cache so an operator can tell which client sent which file.
attribution_store = None


def get_attribution_store() -> AttributionStore:
    """Return the process-wide attribution store, creating it on first use.

    The store is created lazily so that importing the module never touches the
    filesystem, which keeps test runs away from the production directory.
    """
    global attribution_store
    if attribution_store is None:
        attribution_store = AttributionStore()
    return attribution_store


def _get_cached_result(cache_key: str):
    """Return a cached transcription result and discard legacy invalid values."""
    if cache_key not in transcription_cache:
        return None

    cached_result = transcription_cache[cache_key]
    if isinstance(cached_result, dict):
        return cached_result

    # Older cache writers could store a boolean success marker. Treat it as a
    # miss so OpenAI-compatible clients receive a real transcription response.
    logger.warning("Discarding invalid transcription cache entry: %s", cache_key)
    transcription_cache.delete(cache_key)
    return None

request_queue = None
response_queue = None
pending_results = {}
pending_streams = {}
model_usage = None

TEMPLATE_PATH = _web_asset("webui.html", template=True)
BOT_SCRIPT_PATH = Path(__file__).parent / "telegram_bot.py"

app = FastAPI()


@app.middleware("http")
async def _return_consumed_webui_token(request: Request, call_next):
    """Return the replacement even when a handler raises an HTTP error."""
    response = await call_next(request)
    token = getattr(request.state, "webui_next_csrf", None)
    if token:
        response.headers["X-Whisper-Next-CSRF"] = token
        response.headers["Cache-Control"] = "no-store"
    return response

@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup_event()
    try:
        yield
    finally:
        await shutdown_event()

app.router.lifespan_context = lifespan

def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _bot_enabled() -> bool:
    flag = os.getenv("TG_BOT_ENABLED")
    if flag is not None:
        return _env_flag("TG_BOT_ENABLED")
    return bool(os.getenv("TG_BOT_TOKEN"))


def start_bot_subprocess():
    if not _bot_enabled():
        logger.info("Telegram bot subprocess disabled.")
        return None
    if not os.getenv("TG_BOT_TOKEN"):
        logger.warning("TG_BOT_ENABLED is set, but TG_BOT_TOKEN is empty. Bot will not start.")
        return None
    if importlib.util.find_spec("aiogram") is None:
        logger.error("aiogram is not installed. Telegram bot subprocess will not start.")
        return None
    if not BOT_SCRIPT_PATH.exists():
        logger.error("telegram_bot.py not found. Telegram bot subprocess will not start.")
        return None

    env = os.environ.copy()
    env.setdefault("TG_BOT_API_KEY", os.getenv("API_KEY", "bad-key"))
    env.setdefault("TG_BOT_MODEL", os.getenv("MODEL", DEFAULT_MODEL))
    env.setdefault("TG_BOT_SERVER_URL", "http://127.0.0.1:7653/transcribe")
    env.setdefault("WHISPER_URL", env["TG_BOT_SERVER_URL"])

    process = subprocess.Popen(
        [sys.executable, str(BOT_SCRIPT_PATH)],
        cwd=str(Path(__file__).parent),
        env=env,
    )
    logger.info("Telegram bot subprocess started with pid=%s", process.pid)
    return process


async def stop_bot_subprocess():
    process = getattr(app.state, "bot_process", None)
    if process is None or process.poll() is not None:
        return

    logger.info("Stopping Telegram bot subprocess...")
    loop = asyncio.get_running_loop()
    process.send_signal(signal.SIGINT)
    try:
        await loop.run_in_executor(None, process.wait, 10)
        logger.info("Telegram bot subprocess stopped.")
        return
    except subprocess.TimeoutExpired:
        logger.warning("Telegram bot subprocess did not stop in time, sending terminate.")

    process.terminate()
    try:
        await loop.run_in_executor(None, process.wait, 5)
        logger.info("Telegram bot subprocess terminated.")
    except subprocess.TimeoutExpired:
        logger.warning("Telegram bot subprocess is still alive, sending kill.")
        process.kill()
        await loop.run_in_executor(None, process.wait)


async def monitor_bot_subprocess():
    while not app.state.lifecycle_stop.is_set():
        process = getattr(app.state, "bot_process", None)
        if process is not None and process.poll() is not None:
            logger.warning(
                "Telegram bot subprocess exited with code %s. Restarting.",
                process.returncode,
            )
            app.state.bot_process = start_bot_subprocess()
        await asyncio.sleep(5)

class WebJobCancelled(Exception):
    """Worker acknowledged a cooperative cancellation at a safe checkpoint."""


class WorkerStopped(Exception):
    """The worker owning a request exited before responding."""


def model_worker(request_queue, response_queue, stop_event, usage_dict, cancel_flags=None):
    model_cache = {}

    while not stop_event.is_set():
        try:
            all_requests = []
            try:
                while True:
                    all_requests.append(request_queue.get_nowait())
            except:
                pass

            if not all_requests:
                time.sleep(0.05)
                continue

            all_requests.sort(key=lambda r: MODEL_PRIORITY.get(r["model"], 999))

            for request in all_requests:
                requested_model_name = request["model"]
                public_requested_model = request.get("requested_model", requested_model_name)
                audio_bytes = request["audio_bytes"]
                request_id = request["request_id"]
                lang = request.get("language")
                cache_key = request.get("cache_key")
                beam_size = request.get("beam_size", 5)
                temperature = request.get("temperature", 0.0)
                stream = request.get("stream")
                vad_filter = request.get("vad_filter", WHISPER_VAD_ENABLED)
                model_name = requested_model_name
                model_substituted = False
                substitution_reason = None
                temp_path = None
                success = False
                error_text = None
                audio_seconds = None
                load_ms = 0.0
                inference_ms = 0.0
                queue_ms = max(0.0, (time.perf_counter() - request.get("queued_at", time.perf_counter())) * 1000)
                cold_load = False
                info = None
                model = None
                segments = None
                model_session = ExitStack()
                cancelled = False
                terminal_payload = None
                job_key = request.get('web_job_key')

                def check_cancel():
                    if job_key is not None and cancel_flags is not None and cancel_flags.get(job_key, False):
                        raise WebJobCancelled('Web job cancelled')

                try:
                    check_cancel()
                    with tempfile.NamedTemporaryFile(delete=False) as temp:
                        temp.write(audio_bytes)
                        temp.flush()
                        temp_path = temp.name
                    audio_seconds = audio_duration_seconds(temp_path)
                    diarize = request.get("diarize", False)
                    check_cancel()
                    speaker_turns = diarize_audio(temp_path, vad_filter=vad_filter, model_provider=_diarization_model_session) if diarize else []
                    check_cancel()

                    if requested_model_name == OPENAI_WHISPER_INTERNAL_MODEL:
                        model_name = _select_openai_whisper_model(model_cache)
                        model_substituted = model_name != OPENAI_DEFAULT_MODEL
                        substitution_reason = "loaded_model_alias" if model_substituted else "default_model"
                        logger.info(
                            "Resolved %s to %s (loaded=%s)",
                            OPENAI_WHISPER_ALIAS,
                            model_name,
                            sorted(model_cache.keys(), key=MODEL_PRIORITY.get),
                        )
                    else:
                        compatible_model = _select_compatible_loaded_model(model_name, model_cache)
                        if compatible_model is not None:
                            model_name = compatible_model
                            model_substituted = True
                            substitution_reason = "compatible_loaded_model"
                            logger.info(
                                "Serving %s with already loaded compatible model %s",
                                public_requested_model,
                                model_name,
                            )

                    reset_cuda_peak_memory()
                    cold_load = model_name not in model_cache
                    load_started = time.perf_counter()
                    if model_name not in model_cache:
                        ensure_vram_for_model(model_name, model_cache)
                        model_cache[model_name] = _create_model_entry(model_name)
                    check_cancel()

                    model_entry = model_cache[model_name]
                    if ManagedModel is not None and isinstance(model_entry, ManagedModel):
                        model_entry.acquire()
                        model = model_session.enter_context(model_entry.inference())
                    else:
                        model = model_entry
                    load_ms = (time.perf_counter() - load_started) * 1000 if cold_load else 0.0
                    usage_dict[model_name] = usage_dict.get(model_name, 0) + 1
                    if public_requested_model != model_name:
                        usage_dict[public_requested_model] = usage_dict.get(public_requested_model, 0) + 1

                    if ManagedModel is not None and isinstance(model_entry, ManagedModel):
                        model_entry.touch(active=True)
                    inference_started = time.perf_counter()
                    all_segments = []
                    all_words = []

                    def _decode(current_beam: int):
                        """Decode the whole recording; faster-whisper decodes lazily."""
                        check_cancel()
                        segments, info = model.transcribe(
                            temp_path,
                            beam_size=current_beam,
                            language=lang,
                            temperature=temperature,
                            word_timestamps=request.get("words", False) or diarize,
                            **({"speaker_turns": speaker_turns} if diarize and model_name == "gigaam-v3" else {}),
                            condition_on_previous_text=False,
                            **build_vad_transcribe_options(vad_filter),
                            no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
                            log_prob_threshold=WHISPER_LOG_PROB_THRESHOLD,
                            compression_ratio_threshold=WHISPER_COMPRESSION_RATIO_THRESHOLD,
                        )
                        decoded_segments = []
                        decoded_words = []
                        for seg in segments:
                            check_cancel()
                            if ManagedModel is not None and isinstance(model_entry, ManagedModel):
                                model_entry.touch(active=True)
                            seg_text = (seg.text or "").strip()
                            if not seg_text:
                                logger.info("%s skipped empty/VAD-only segment %.2f-%.2f", model_name, seg.start, seg.end)
                                continue
                            seg_data = {
                                "start": seg.start,
                                "end": seg.end,
                                "text": seg_text,
                                "avg_logprob": getattr(seg, "avg_logprob", None),
                                "no_speech_prob": getattr(seg, "no_speech_prob", None),
                                "compression_ratio": getattr(seg, "compression_ratio", None),
                            }
                            logger.info("%s processed segment: %s", model_name, seg_data)
                            if (request.get("words", False) or diarize) and getattr(seg, "words", None):
                                words = [
                                    {"start": w.start, "end": w.end, "word": w.word}
                                    for w in seg.words
                                ]
                            else:
                                words = []

                            attributed = attribute_segment(seg_data, words, speaker_turns) if diarize else [seg_data]
                            decoded_segments.extend(attributed)
                            decoded_words.extend(words)
                            if stream:
                                for part in attributed:
                                    payload = {"request_id": request_id, "segment": part}
                                    if request.get("words", False):
                                        payload["words"] = [w for w in words if w["start"] >= part["start"] and w["end"] <= part["end"]]
                                    response_queue.put(payload)
                        return decoded_segments, decoded_words, info

                    try:
                        all_segments, all_words, info = _decode(beam_size)
                    except Exception as exc:
                        if not is_cuda_oom(exc):
                            raise
                        logger.warning(
                            "CUDA out of memory during %s inference (beam=%d): %s",
                            model_name,
                            beam_size,
                            exc,
                        )
                        if stream:
                            raise GpuOutOfMemoryError(
                                "CUDA out of memory during streaming inference; the GPU is busy"
                            ) from exc
                        release_cached_models(model_cache, keep=model_name)
                        try:
                            all_segments, all_words, info = _decode(RETRY_BEAM_SIZE)
                            logger.warning(
                                "Retry with beam=%d succeeded for %s",
                                RETRY_BEAM_SIZE,
                                model_name,
                            )
                        except Exception as retry_exc:
                            if not is_cuda_oom(retry_exc):
                                raise
                            raise GpuOutOfMemoryError(
                                "CUDA out of memory: the GPU is occupied by another tenant"
                            ) from retry_exc
                    inference_ms = (time.perf_counter() - inference_started) * 1000

                    check_cancel()
                    result = {
                        "text": " ".join([s["text"] for s in all_segments]),
                        "segments": all_segments,
                        "language": info.language,
                        "language_probability": info.language_probability,
                        "requested_model": public_requested_model,
                        "served_model": model_name,
                        "model_substituted": model_substituted,
                        "substitution_reason": substitution_reason,
                    }
                    client_comment = request.get("client_comment")
                    if client_comment:
                        result["client_comment"] = client_comment
                    if request.get("words", False):
                        result["words"] = all_words
                    if diarize:
                        result["duration"] = audio_seconds or 0.0
                        result["speaker_text"] = speaker_text(all_segments)
                        result["speaker_turns"] = speaker_turns
                        result["diarization_model"] = "pyannote/speaker-diarization-community-1"
                        result["diarization_device"] = "cuda" if torch.cuda.is_available() else "cpu"

                    if cache_key:
                        transcription_cache.set(cache_key, result, expire=CACHE_TTL)

                    terminal_payload = {
                        "request_id": request_id,
                        "result": result,
                        "final": True,
                    }
                    success = True

                except WebJobCancelled:
                    cancelled = True
                    error_text = 'Web job cancelled'
                except Exception as e:
                    error_text = str(e)
                    logger.exception("Ошибка в worker:")
                    error_payload = {
                        "request_id": request_id,
                        "error": str(e)
                    }
                    if isinstance(e, GpuOutOfMemoryError):
                        error_payload["error_code"] = e.code
                        error_payload["retryable"] = e.retryable
                    terminal_payload = error_payload
                finally:
                    # Drop worker-local references before marking the cached
                    # model idle so broker eviction can actually free VRAM.
                    model = None
                    segments = None
                    try:
                        model_session.close()
                    except Exception:
                        logger.exception("Failed to mark model idle")
                    if temp_path:
                        Path(temp_path).unlink(missing_ok=True)
                    duration = audio_seconds or getattr(info, "duration", None)
                    metric = {
                        "requested_model": public_requested_model,
                        "served_model": model_name,
                        "model_substituted": model_substituted,
                        "substitution_reason": substitution_reason,
                        "device": "cuda" if torch.cuda.is_available() else "cpu",
                        "cold_load": cold_load if "cold_load" in locals() else False,
                        "load_ms": round(load_ms, 3),
                        "queue_ms": round(queue_ms, 3),
                        "audio_seconds": round(duration, 3) if duration is not None else None,
                        "inference_ms": round(inference_ms, 3),
                        "rtf": round(inference_ms / 1000 / duration, 5) if duration and duration > 0 else None,
                        "peak_vram_mb": peak_vram_mb(),
                        "success": success,
                    }
                    if request.get("client_comment"):
                        metric["client_comment"] = request["client_comment"]
                    if error_text:
                        metric["error"] = error_text
                    append_metric(metric)
                if cancelled:
                    terminal_payload = {'request_id': request_id, 'error': 'Web job cancelled',
                                        'error_code': 'web_job_cancelled', 'final': True}
                if terminal_payload is not None:
                    response_queue.put(terminal_payload)

        except Exception as e:
            logger.exception("Фатальная ошибка в loop:")
            time.sleep(1)

    # Normal worker shutdown must release cached ASR leases as well.
    for entry in model_cache.values():
        if ManagedModel is not None and isinstance(entry, ManagedModel):
            entry.release()


async def response_listener(stop_event, response_queue, retire_event=None):
    loop = asyncio.get_running_loop()
    inbox = asyncio.Queue(maxsize=1)
    reader_stop = threading.Event()
    consumed = threading.Event()
    reader_finished = object()

    def read_responses():
        # One daemon reader per generation: if a killed writer leaves a partial
        # IPC frame, the reader cannot hold up interpreter shutdown. The single
        # handoff slot bounds buffering and preserves ordering.
        try:
            while not reader_stop.is_set():
                try:
                    item = response_queue.get(True, .2)
                except thread_queue.Empty:
                    if retire_event is not None and retire_event.is_set():
                        break
                    continue
                consumed.clear()
                loop.call_soon_threadsafe(inbox.put_nowait, item)
                while not consumed.wait(.2):
                    if reader_stop.is_set():
                        return
                if item is None or item.get('__shutdown__'):
                    break
        except Exception:
            logger.exception('Worker response reader failed')
        finally:
            if not reader_stop.is_set():
                try:
                    loop.call_soon_threadsafe(inbox.put_nowait, reader_finished)
                except RuntimeError:
                    pass  # The application loop already stopped.

    reader = threading.Thread(target=read_responses, name='worker-response-reader', daemon=True)
    reader.start()
    try:
        while True:
            try:
                result = await inbox.get()
                if result is reader_finished:
                    break
                if result is None or result.get("__shutdown__"):
                    break
                logger.info(f'response_listener get from queue result {result}')
                request_id = result["request_id"]
                if request_id in pending_results:
                    # A segment is not a terminal response for a non-stream request.
                    if "result" not in result and "error" not in result:
                        continue
                    future = pending_results.pop(request_id)
                    if future.done():
                        continue
                    if "error" in result:
                        if result.get("error_code") == "gpu_out_of_memory":
                            future.set_exception(GpuOutOfMemoryError(result["error"]))
                        else:
                            future.set_exception(Exception(result["error"]))
                    else:
                        future.set_result(result["result"])
                elif request_id in pending_streams:
                    queue = pending_streams[request_id]
                    if "error" in result:
                        await queue.put({"error": result["error"]})
                        await queue.put(None)
                        pending_streams.pop(request_id, None)
                    elif "segment" in result:
                        payload = {"segment": result["segment"]}
                        if "words" in result:
                            payload["words"] = result["words"]
                        await queue.put(payload)
                    elif result.get("final"):
                        await queue.put({"result": result["result"]})
                        await queue.put(None)
                        pending_streams.pop(request_id, None)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Failed to dispatch worker response; continuing")
            finally:
                consumed.set()
    finally:
        reader_stop.set()
        consumed.set()


def _worker_accepts_requests():
    # Legacy tests and embedders may provide their own request queue.
    if not hasattr(app.state, 'worker_accepting'):
        return getattr(app.state, 'request_queue', None) is not None
    process = getattr(app.state, 'model_process', None)
    return bool(app.state.worker_accepting and process is not None and process.is_alive())


async def _worker_supervisor(ctx):
    """Replace one dead generation without replaying its queued requests."""
    while not app.state.lifecycle_stop.is_set():
        await asyncio.sleep(.2)
        process = app.state.model_process
        if process is None or process.is_alive():
            continue
        app.state.worker_accepting = False
        app.state.worker_generation += 1
        logger.error('Model worker exited; closing admission')
        # Let the old listener dispatch replies already sent before the crash.
        await asyncio.sleep(1)
        if app.state.lifecycle_stop.is_set():
            return
        listener = app.state.response_listener_task
        app.state.response_listener_retire.set()
        done, _ = await asyncio.wait({listener}, timeout=2)
        if not done:
            # Never close a queue or discard pending requests while its reader
            # can still consume an answer. Keep admission closed on this fault.
            logger.error('Worker response reader did not retire; recovery halted')
            return
        if app.state.lifecycle_stop.is_set():
            return
        for request_id, future in list(pending_results.items()):
            pending_results.pop(request_id, None)
            if not future.done():
                future.set_exception(WorkerStopped('Model worker stopped before responding'))
        for request_id, stream in list(pending_streams.items()):
            pending_streams.pop(request_id, None)
            await stream.put({'error': 'Model worker stopped before responding'})
            await stream.put(None)
        for old_queue in (app.state.request_queue, app.state.response_queue):
            try:
                old_queue.cancel_join_thread()
                old_queue.close()
            except (OSError, ValueError, AttributeError):
                pass
        process.join(timeout=0)
        if app.state.worker_restarts >= 1 or app.state.lifecycle_stop.is_set():
            logger.error('Model worker restart limit reached; admission remains closed')
            return
        app.state.worker_restarts += 1
        request_queue, response_queue = ctx.Queue(), ctx.Queue()
        replacement = ctx.Process(target=model_worker, args=(request_queue, response_queue,
                                  app.state.stop_event, app.state.model_usage, app.state.job_cancel_flags))
        try:
            replacement.start()
        except Exception:
            logger.exception('Could not restart model worker')
            for new_queue in (request_queue, response_queue):
                new_queue.cancel_join_thread()
                new_queue.close()
            return
        app.state.request_queue = request_queue
        app.state.response_queue = response_queue
        app.state.model_process = replacement
        app.state.response_listener_retire = asyncio.Event()
        app.state.response_listener_task = asyncio.create_task(response_listener(
            app.state.stop_event, response_queue, app.state.response_listener_retire))
        app.state.worker_accepting = True
        logger.info('Model worker restarted with new queues')


def _recover_web_jobs(store):
    """Recover queued/in-flight work and retries interrupted after failure."""
    recovered = store.recover_interrupted()
    with sqlite3.connect(store.path, timeout=store.timeout) as db:
        terminal = db.execute("SELECT owner, id, attempt FROM jobs WHERE status IN ('done', 'error')").fetchall()
        cancelled = db.execute("SELECT id FROM jobs WHERE status='cancelled'").fetchall()
    for (job_id,) in cancelled:
        _job_spool().remove(job_id)
    for owner, job_id, attempt in terminal:
        try:
            retry = store.retry_if_diarization_changed(owner, job_id, attempt)
            if retry:
                recovered.append(retry)
        except (JobNotFound, InvalidTransition):
            continue
    return recovered


async def startup_event():
    global model_usage
    # CUDA cannot be initialized safely in a forked child after the parent
    # imports/inspects torch. Spawn gives the model worker a clean interpreter
    # and prevents the post-driver-reboot CUDA initialization failure.
    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    manager = ctx.Manager()
    model_usage = manager.dict()
    job_cancel_flags = manager.dict()
    request_queue = ctx.Queue()
    response_queue = ctx.Queue()
    process = ctx.Process(
        target=model_worker,
        args=(request_queue, response_queue, stop_event, model_usage, job_cancel_flags),
    )
    process.start()
    executor = ThreadPoolExecutor()
    app.state.stop_event = stop_event
    app.state.model_process = process
    app.state.executor = executor
    app.state.request_queue = request_queue
    app.state.response_queue = response_queue
    app.state.manager = manager
    app.state.job_cancel_flags = job_cancel_flags
    app.state.worker_generation = 0
    app.state.worker_restarts = 0
    app.state.worker_accepting = True
    app.state.response_listener_retire = asyncio.Event()
    app.state.response_listener_task = asyncio.create_task(
        response_listener(stop_event, response_queue, app.state.response_listener_retire)
    )
    store = _job_store()
    if hasattr(store, 'recover_interrupted') and hasattr(store, 'retry_if_diarization_changed'):
        for job in _recover_web_jobs(store):
            _launch_job(job['owner'], job)
    else:
        logger.warning('Web job restart recovery is unavailable: JobStore.recover_interrupted is missing')
    app.state.model_usage = model_usage
    app.state.lifecycle_stop = asyncio.Event()
    app.state.worker_supervisor_task = asyncio.create_task(_worker_supervisor(ctx))
    app.state.bot_process = start_bot_subprocess()
    app.state.bot_monitor_task = asyncio.create_task(monitor_bot_subprocess())

async def shutdown_event():
    global model_usage
    logger.info("Остановка процесса...")
    app.state.worker_accepting = False
    if hasattr(app.state, "lifecycle_stop"):
        app.state.lifecycle_stop.set()
    if hasattr(app.state, 'worker_supervisor_task'):
        app.state.worker_supervisor_task.cancel()
        await asyncio.gather(app.state.worker_supervisor_task, return_exceptions=True)
        del app.state.worker_supervisor_task
    if hasattr(app.state, "bot_monitor_task"):
        app.state.bot_monitor_task.cancel()
        try:
            await app.state.bot_monitor_task
        except asyncio.CancelledError:
            pass
    await stop_bot_subprocess()
    app.state.stop_event.set()
    process = app.state.model_process
    process.join(timeout=10)
    if process.is_alive():
        logger.warning("Модель не завершилась вовремя, отправляем terminate.")
        process.terminate()
        process.join(timeout=5)
    else:
        logger.info("Модель завершена.")

    reader_stopped = True
    if hasattr(app.state, 'response_listener_task'):
        retirement = getattr(app.state, 'response_listener_retire', None)
        if retirement is not None:
            retirement.set()
            done, _ = await asyncio.wait({app.state.response_listener_task}, timeout=2)
            reader_stopped = bool(done)
            if not reader_stopped:
                logger.error('Response reader did not retire during shutdown; leaving its queue open')
        else:
            app.state.response_listener_task.cancel()
            await asyncio.gather(app.state.response_listener_task, return_exceptions=True)

    # Job coroutines may still be awaiting a response from the stopped worker.
    # Finish their cleanup while the Manager-backed cancellation channel lives.
    loop = asyncio.get_running_loop()
    owned_tasks = [task for task in _job_tasks if task.get_loop() is loop and not task.done()]
    for task in owned_tasks:
        task.cancel()
    if owned_tasks:
        await asyncio.gather(*owned_tasks, return_exceptions=True)

    app.state.executor.shutdown(wait=False)

    queues = [app.state.request_queue]
    if reader_stopped:
        queues.append(app.state.response_queue)
    for q in queues:
        try:
            q.cancel_join_thread()
            q.close()
        except (AttributeError, OSError, ValueError):
            pass

    app.state.job_cancel_flags = None
    app.state.model_process = None
    app.state.request_queue = None
    app.state.response_queue = None
    del app.state.worker_accepting
    if hasattr(app.state, "manager"):
        try:
            app.state.manager.shutdown()
        except Exception:
            pass
        del app.state.manager
    if hasattr(app.state, 'model_usage'):
        del app.state.model_usage
    model_usage = None

async def _transcribe_impl(
    file: UploadFile,
    model: str,
    language: Optional[str],
    beam_size: Optional[int],
    temperature: float,
    stream: bool,
    words: bool,
    vad_filter: bool,
    diarize: bool = False,
    comment: Optional[str] = None,
    *, wait_for_terminal: bool = False,
    web_job_key=None,
):
    diarize = diarize is True
    client_comment = normalize_comment(comment)
    actual_model = OPENAI_MODEL_MAP.get(model, model)
    if actual_model != OPENAI_WHISPER_INTERNAL_MODEL and actual_model not in MODEL_PRIORITY:
        return JSONResponse({"error": "Unsupported model"}, status_code=400)

    audio_bytes = await file.read()
    audio_hash = hash_bytes(audio_bytes)
    cache_key = transcription_cache_key("native", model, language, audio_hash, vad_filter=vad_filter, words=words, diarize=diarize)

    # Remember who sent this file; the cache key stays independent of it.
    if client_comment:
        get_attribution_store().record(audio_hash, client_comment, requested_model=model)

    cached_result = _get_cached_result(cache_key)
    if cached_result is not None and not stream:
        logger.info(f"[CACHE HIT] {cache_key}")
        append_cache_metric(cached_result, model, client_comment)
        # The stored comment belongs to whoever produced the cache entry. Never
        # hand it to another client: report the current request instead.
        result = dict(cached_result)
        result.pop("client_comment", None)
        if client_comment:
            result["client_comment"] = client_comment
        return result
    if cached_result is not None and stream:
        logger.info(f"[CACHE HIT] {cache_key}")
        append_cache_metric(cached_result, model, client_comment)
        # The cached comment belongs to whoever produced the entry, so it is
        # replaced with the comment of the current request before it is sent.
        cached_payload = dict(cached_result)
        cached_payload.pop("client_comment", None)
        if client_comment:
            cached_payload["client_comment"] = client_comment
        async def cached():
            for segment in cached_result.get("segments", []):
                payload = {"segment": segment}
                if words:
                    payload["words"] = [w for w in cached_result.get("words", [])
                                        if w["start"] >= segment["start"] and w["end"] <= segment["end"]]
                yield json.dumps(payload) + "\n"
            yield json.dumps({"result": cached_payload}) + "\n"
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # для nginx
            "X-Content-Type-Options": "nosniff",
        }
        return StreamingResponse(cached(), media_type="application/x-ndjson", headers=headers)

    if not _worker_accepts_requests():
        return JSONResponse({'error': 'Model worker unavailable', 'code': 'worker_stopped'}, status_code=503)
    request_id = uuid4().hex
    loop = asyncio.get_event_loop()
    if stream:
        queue = asyncio.Queue()
        pending_streams[request_id] = queue
    else:
        future = loop.create_future()
        pending_results[request_id] = future

    job_process = getattr(app.state, 'model_process', None)
    if wait_for_terminal and not stream:
        process = job_process
        if process is None or not process.is_alive():
            pending_results.pop(request_id, None)
            future.cancel()
            return JSONResponse({'error': 'Model worker stopped before responding',
                                 'code': 'worker_stopped'}, status_code=503)

    app.state.request_queue.put({
        "request_id": request_id,
        "audio_bytes": audio_bytes,
        "model": actual_model,
        "requested_model": model,
        "language": language,
        "cache_key": cache_key,
        "beam_size": beam_size,
        "temperature": temperature,
        "stream": stream,
        "words": words,
        "vad_filter": vad_filter,
        "diarize": diarize,
        "client_comment": client_comment,
        "queued_at": time.perf_counter(),
        "web_job_key": web_job_key,
    })

    if stream:
        async def generator():
            deadline = time.monotonic() + (int(os.getenv("WHISPER_DIARIZATION_TIMEOUT_SECONDS", "86400")) if diarize else 600)
            try:
                while True:
                    if time.monotonic() >= deadline:
                        yield json.dumps({"error": "Transcription timeout"}) + "\n"
                        break
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=min(15, deadline - time.monotonic()))
                    except asyncio.TimeoutError:
                        yield json.dumps({"status": "processing"}) + "\n"
                        continue
                    if item is None:
                        break
                    yield json.dumps(item) + "\n"
            finally:
                pending_streams.pop(request_id, None)
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # для nginx
            "X-Content-Type-Options": "nosniff",
        }
        return StreamingResponse(generator(), media_type="application/x-ndjson", headers=headers)
    else:
        try:
            if wait_for_terminal:
                while not future.done():
                    process = job_process
                    if process is None or not process.is_alive():
                        supervisor = getattr(app.state, 'worker_supervisor_task', None)
                        if supervisor is not None:
                            # The supervisor owns the old listener and must
                            # finish its in-flight read before failing requests.
                            if not supervisor.done():
                                await asyncio.wait({future, supervisor}, timeout=4)
                            if future.done():
                                break
                            listener = getattr(app.state, 'response_listener_task', None)
                            if listener is not None and not listener.done():
                                # Keep the pending future registered: a blocked
                                # reader may still return an already-taken reply.
                                return JSONResponse({'error': 'Worker response channel did not drain; completion unknown'}, status_code=504)
                        # The worker can exit after publishing its final reply;
                        # allow the existing listener to dispatch it first.
                        await asyncio.wait({future}, timeout=1.0)
                        if future.done():
                            break
                        pending_results.pop(request_id, None)
                        future.cancel()
                        cached_result = _get_cached_result(cache_key)
                        if cached_result is not None:
                            result = dict(cached_result)
                            result.pop('client_comment', None)
                            if client_comment:
                                result['client_comment'] = client_comment
                            return result
                        return JSONResponse({'error': 'Model worker stopped before responding',
                                             'code': 'worker_stopped'}, status_code=503)
                    await asyncio.wait({future}, timeout=0.2)
                result = future.result()
            else:
                timeout = int(os.getenv("WHISPER_DIARIZATION_TIMEOUT_SECONDS", "86400")) if diarize else 600
                result = await asyncio.wait_for(future, timeout=timeout)
            if client_comment and isinstance(result, dict):
                result = dict(result)
                result["client_comment"] = client_comment
            return result
        except asyncio.TimeoutError:
            pending_results.pop(request_id, None)
            return JSONResponse({"error": "Timeout"}, status_code=504)
        except asyncio.CancelledError:
            pending_results.pop(request_id, None)
            future.cancel()
            raise
        except WorkerStopped as error:
            pending_results.pop(request_id, None)
            cached_result = _get_cached_result(cache_key)
            if cached_result is not None:
                result = dict(cached_result)
                result.pop('client_comment', None)
                if client_comment:
                    result['client_comment'] = client_comment
                return result
            return JSONResponse({'error': str(error), 'code': 'worker_stopped'}, status_code=503)
        except Exception as e:
            pending_results.pop(request_id, None)
            if isinstance(e, GpuOutOfMemoryError):
                return JSONResponse(
                    {"error": str(e), "code": e.code, "retryable": True},
                    status_code=503,
                    headers={"Retry-After": "30"},
                )
            return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Query(DEFAULT_MODEL),
    language: Optional[str] = Query(None),
    beam_size: Optional[int] = Query(5),
    temperature: float = Query(0.0),
    stream: bool = Query(False),
    words: bool = Query(False),
    vad_filter: bool = Query(True),
    api_key: str = Query(...),
    diarize: bool = Query(False),
    comment: Optional[str] = Query(None),
):
    if api_key not in ALLOWED_API_KEYS:
        return JSONResponse({"error": "Invalid API key"}, status_code=403)
    return await _transcribe_impl(file, model, language, beam_size, temperature, stream, words, vad_filter, diarize, comment)


@app.post("/web/session")
async def refresh_webui_session(request: Request):
    if not _same_origin_request(request):
        raise HTTPException(status_code=403, detail="Invalid request origin")
    sid = request.cookies.get(WEBUI_SESSION_COOKIE, "")
    if _load_webui_session(sid):
        token = _rotate_webui_token(sid)
    else:
        sid, token = _create_webui_session()
    response = JSONResponse({"token": token}, headers={"Cache-Control": "no-store"})
    _set_webui_session_cookie(response, request, sid)
    return response


def _job_owner(request):
    sid = request.cookies.get(WEBUI_SESSION_COOKIE, "")
    if not _load_webui_session(sid):
        raise HTTPException(status_code=403, detail="Invalid or expired web UI session")
    return sid


def _job_get(owner, job_id):
    try:
        return _job_store().get(owner, job_id)
    except JobNotFound:
        raise HTTPException(status_code=404, detail="Unknown job")


def _job_view(job):
    settings = job['diarization']
    return {**job, **{k: settings[k] for k in ('filename', 'size', 'model')},
            'diarize': settings.get('desired_diarize', settings['diarize']),
            'offset': _job_spool().offset(job['id']), 'transcript_id': job['id'] if job['status'] == 'done' else None}


def _create_idempotent_job(store, owner, request_id, settings):
    """Bind a request to its job in the same SQLite transaction as creation."""
    store._identity(owner, 'owner')
    encoded = store._json(settings)
    with store._transaction() as db:
        db.execute('''CREATE TABLE IF NOT EXISTS webui_create_requests (
            owner TEXT NOT NULL, request_id TEXT NOT NULL, settings TEXT NOT NULL,
            job_id TEXT NOT NULL,
            PRIMARY KEY (owner, request_id))''')
        previous = db.execute('SELECT job_id, settings FROM webui_create_requests WHERE owner=? AND request_id=?',
                              (owner, request_id)).fetchone()
        if previous:
            if previous['settings'] != encoded:
                raise HTTPException(status_code=409, detail='client_request_id already used with different job settings')
            try:
                return store._record(db, store._owned(db, owner, previous['job_id']))
            except JobNotFound:
                raise HTTPException(status_code=410, detail='Job for client_request_id was deleted')
        if db.execute('SELECT count(*) FROM jobs').fetchone()[0] >= store.max_jobs:
            raise CapacityExceeded('max_jobs reached')
        job_id = uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        db.execute('INSERT INTO jobs VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, ?)',
                   (job_id, owner, 'uploading', 1, encoded, now, now))
        db.execute('INSERT INTO webui_create_requests VALUES (?, ?, ?, ?)', (owner, request_id, encoded, job_id))
        return store._record(db, store._owned(db, owner, job_id))


def _launch_job(owner, job):
    generation = getattr(app.state, 'worker_generation', None)
    task = asyncio.create_task(_run_web_job(owner, job['id'], job['attempt'], generation))
    _job_tasks.add(task)
    task.add_done_callback(_job_tasks.discard)


async def _run_web_job(owner, job_id, attempt, generation=None):
    async with _job_worker_slot:
        if generation is not None and generation != getattr(app.state, 'worker_generation', None):
            try:
                _job_store().fail(owner, job_id, attempt, 'Model worker stopped before processing queued job')
            except (InvalidTransition, JobNotFound):
                pass
            return
        await _execute_web_job(owner, job_id, attempt)


async def _execute_web_job(owner, job_id, attempt):
    store = _job_store()
    terminal_signal = False
    unknown_completion = False
    interrupted = False
    try:
        job = store.start(owner, job_id, attempt)
        settings = job['diarization']
        with (_job_spool().root / job_id).open('rb') as source:
            result = await _transcribe_impl(UploadFile(file=source, filename=settings['filename']),
                                            settings['model'], None, 5, 0.0, False, False, True,
                                            settings['diarize'], wait_for_terminal=True,
                                            web_job_key=(job_id, attempt))
        if isinstance(result, JSONResponse):
            if result.status_code == 504:
                unknown_completion = True
                logger.error('Web job %s timed out with worker completion unknown', job_id)
                return
            terminal_signal = True
            payload = json.loads(result.body)
            raise RuntimeError(payload.get('error', 'Transcription failed'))
        terminal_signal = True
        result = dict(result)
        result['transcript_id'] = job_id
        store.complete(owner, job_id, attempt, result)
    except asyncio.CancelledError:
        interrupted = True
        raise
    except InvalidTransition:
        try:
            job = store.get(owner, job_id)
            if terminal_signal and job['attempt'] == attempt and job['status'] == 'cancel_requested':
                store.finish_cancel(owner, job_id, attempt)
                _job_spool().remove(job_id)
        except (InvalidTransition, JobNotFound):
            pass
    except Exception as error:
        logger.exception('Web job %s failed', job_id)
        try:
            job = store.get(owner, job_id)
            if job['status'] == 'cancel_requested' and terminal_signal:
                store.finish_cancel(owner, job_id, attempt)
                _job_spool().remove(job_id)
            elif job['status'] != 'cancel_requested':
                store.fail(owner, job_id, attempt, str(error)[:4000])
        except (InvalidTransition, JobNotFound):
            pass
    finally:
        flags = getattr(app.state, 'job_cancel_flags', None)
        if flags is not None and not unknown_completion:
            flags.pop((job_id, attempt), None)
        if not unknown_completion and not interrupted and hasattr(store, 'retry_if_diarization_changed'):
            try:
                retry = store.retry_if_diarization_changed(owner, job_id, attempt)
                if retry:
                    _launch_job(owner, retry)
            except (InvalidTransition, JobNotFound):
                pass


@app.get('/web/jobs')
async def web_jobs_list(request: Request):
    owner = _job_owner(request)
    return {'jobs': [_job_view(job) for job in _job_store().list(owner)]}


@app.post('/web/jobs')
async def web_jobs_create(request: Request):
    _consume_webui_token(request)
    owner = _job_owner(request)
    try:
        data = await request.json()
        filename, size, model, diarize = (data.get(k) for k in ('filename', 'size', 'model', 'diarize'))
        request_id = data.get('client_request_id')
        if (not isinstance(filename, str) or not filename or len(filename) > 255 or
            type(size) is not int or not 0 < size <= WEBUI_MAX_FILE_BYTES or
            not isinstance(model, str) or model not in MODEL_PRIORITY or type(diarize) is not bool):
            raise ValueError('Invalid job settings')
        if 'client_request_id' in data and (not isinstance(request_id, str) or not request_id.strip() or len(request_id.encode('utf-8')) > 128):
            raise ValueError('client_request_id must be a nonempty string of at most 128 bytes')
        settings = dict(filename=filename, size=size, model=model, diarize=diarize, desired_diarize=diarize)
        store = _job_store()
        job = store.create(owner, settings) if request_id is None else _create_idempotent_job(store, owner, request_id, settings)
        return _job_view(job)
    except (ValueError, AttributeError, TypeError) as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.get('/web/jobs/{job_id}')
async def web_jobs_get(job_id: str, request: Request):
    return _job_view(_job_get(_job_owner(request), job_id))


@app.put('/web/jobs/{job_id}/upload')
async def web_jobs_upload(job_id: str, request: Request, offset: int = Query(..., ge=0)):
    _consume_webui_token(request)
    owner = _job_owner(request)
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > WEBUI_MAX_CHUNK_BYTES:
            raise HTTPException(status_code=413, detail='Upload chunk too large')
    async with _job_lock:
        job = _job_get(owner, job_id)
        if job['status'] != 'uploading':
            raise HTTPException(status_code=409, detail='Job is not uploading')
        try:
            new_offset = _job_spool().append(job_id, offset, bytes(data), job['diarization']['size'])
        except UploadConflict as error:
            raise HTTPException(status_code=409, detail=str(error))
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error))
    return {'offset': new_offset}


@app.post('/web/jobs/{job_id}/start')
async def web_jobs_start(job_id: str, request: Request):
    _consume_webui_token(request)
    owner = _job_owner(request)
    async with _job_lock:
        job = _job_get(owner, job_id)
        if job['status'] != 'uploading':
            return _job_view(job)
        if not _worker_accepts_requests():
            raise HTTPException(status_code=503, detail='Model worker unavailable')
        if _job_spool().offset(job_id) != job['diarization']['size']:
            raise HTTPException(status_code=409, detail='Upload is incomplete')
        try:
            job = _job_store().enqueue(owner, job_id, job['attempt'])
        except InvalidTransition as error:
            raise HTTPException(status_code=409, detail=str(error))
        _launch_job(owner, job)
    return _job_view(job)


@app.post('/web/jobs/{job_id}/cancel')
async def web_jobs_cancel(job_id: str, request: Request):
    _consume_webui_token(request)
    owner = _job_owner(request)
    async with _job_lock:
        job = _job_get(owner, job_id)
        try:
            flags = getattr(app.state, 'job_cancel_flags', None)
            if job['status'] in ('processing', 'cancel_requested') and flags is None:
                raise HTTPException(status_code=503, detail='Worker cancellation channel unavailable')
            job = _job_store().request_cancel(owner, job_id, job['attempt'])
            if job['status'] == 'cancelled':
                _job_spool().remove(job_id)
            elif job['status'] == 'cancel_requested':
                try:
                    flags[(job_id, job['attempt'])] = True
                except (OSError, EOFError, BrokenPipeError):
                    raise HTTPException(status_code=503, detail='Worker cancellation channel unavailable')
        except InvalidTransition as error:
            raise HTTPException(status_code=409, detail=str(error))
    return _job_view(job)


@app.delete('/web/jobs/{job_id}')
async def web_jobs_delete(job_id: str, request: Request):
    _consume_webui_token(request)
    owner = _job_owner(request)
    async with _job_lock:
        job = _job_get(owner, job_id)
        if job['status'] not in ('done', 'error', 'cancelled'):
            raise HTTPException(status_code=409, detail='Only terminal jobs can be deleted')
        _job_spool().remove(job_id)
        try:
            _job_store().delete(owner, job_id)
        except InvalidTransition as error:
            raise HTTPException(status_code=409, detail=str(error))
    return {'deleted': True}


@app.patch('/web/jobs/{job_id}')
async def web_jobs_patch(job_id: str, request: Request):
    _consume_webui_token(request)
    owner = _job_owner(request)
    try:
        data = await request.json()
        if not isinstance(data, dict) or set(data) != {'diarize'} or type(data['diarize']) is not bool:
            raise ValueError('Expected boolean diarize')
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=400, detail=str(error))
    async with _job_lock:
        job = _job_get(owner, job_id)
        settings = {**job['diarization'], 'desired_diarize': data['diarize']}
        if job['status'] == 'uploading':
            settings['diarize'] = data['diarize']
        store = _job_store()
        if not hasattr(store, 'set_diarization'):
            raise HTTPException(status_code=503, detail='Changing diarization is temporarily unavailable')
        previous_attempt = job['attempt']
        try:
            job = store.set_diarization(owner, job_id, job['attempt'], settings)
        except InvalidTransition as error:
            raise HTTPException(status_code=409, detail=str(error))
        if job['status'] == 'queued' and job['attempt'] != previous_attempt:
            _launch_job(owner, job)
    return _job_view(job)


@app.post("/web/transcribe")
async def web_transcribe(
    request: Request,
    file: UploadFile = File(...),
    model: str = Query(DEFAULT_MODEL),
    language: Optional[str] = Query(None),
    beam_size: Optional[int] = Query(5),
    temperature: float = Query(0.0),
    stream: bool = Query(False),
    words: bool = Query(False),
    vad_filter: bool = Query(True),
    diarize: bool = Query(False),
):
    """First-party Web UI tunnel: API_KEY stays server-side; CSRF is session-scoped."""
    next_token = _consume_webui_token(request)
    session_id = request.cookies.get(WEBUI_SESSION_COOKIE, "")
    transcript_id = _new_webui_transcript(session_id)
    response = await _transcribe_impl(file, model, language, beam_size, temperature, stream, words, vad_filter, diarize)
    if not hasattr(response, "headers"):
        response = JSONResponse(response)
    response.headers["X-Whisper-Next-CSRF"] = next_token
    response.headers["X-Whisper-Transcript-Id"] = transcript_id
    if stream and hasattr(response, "body_iterator"):
        response.body_iterator = _capture_transcript_stream(response.body_iterator, transcript_id)
    else:
        try:
            payload = json.loads(bytes(response.body).decode("utf-8"))
        except Exception:
            payload = None
        text = _transcript_text_from_payload(payload)
        if text:
            _fill_webui_transcript(transcript_id, text)
    return response

@app.get("/web/byok/presets")
async def web_byok_presets():
    return JSONResponse(json.loads(_web_asset("byok-presets.json", template=True).read_text()))


@app.post("/web/summarize")
async def web_summarize(request: Request):
    """Reuse a session-scoped summary before spending a model request."""
    next_token = _consume_webui_token(request)
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    sid = request.cookies.get(WEBUI_SESSION_COOKIE, "")
    transcript_id = str(payload.get("transcript_id") or "")
    store = _job_store()
    try:
        job = store.get(sid, transcript_id)
    except JobNotFound:
        job = None
    text = (_transcript_text_from_payload(job['result']) if job['status'] == 'done' else None) if job is not None else _get_webui_transcript(sid, transcript_id)
    if not text:
        raise HTTPException(status_code=404, detail="Unknown transcript id")
    byok = payload.get("byok")
    if byok is not None and (not isinstance(byok, dict) or
        byok.get("apiFormat") not in {"anthropic", "chat-completions", "responses"} or
        not all(isinstance(byok.get(k), str) and byok[k].strip() for k in ("baseUrl", "modelId", "apiKey")) or
        len(byok["apiKey"]) > 4096 or len(byok["modelId"]) > 256 or len(byok["baseUrl"]) > 2048):
        raise HTTPException(status_code=400, detail="Укажите адрес, модель и ключ провайдера")
    instruction = str(payload.get("instruction") or "").strip()[:500]
    identity = {"text": text[:OMNIROUTE_SUMMARY_MAX_CHARS], "instruction": instruction,
                "prompt": SUMMARY_SYSTEM_PROMPT, "max_tokens": OMNIROUTE_SUMMARY_MAX_TOKENS,
                "provider": byok if byok is not None else {"baseUrl": OMNIROUTE_BASE_URL,
                    "model": OMNIROUTE_SUMMARY_MODEL, "fallback": OMNIROUTE_SUMMARY_FALLBACK_MODELS}}
    encoded = json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    cache_key = 'web-summary:v1:' + hmac.new(sid.encode(), encoded, hashlib.sha256).hexdigest()
    lock = _summary_cache_locks.setdefault(cache_key, asyncio.Lock())
    async with lock:
        cached = None if payload.get('force_refresh') is True else transcription_cache.get(cache_key)
        if isinstance(cached, dict) and isinstance(cached.get('summary'), str):
            if job is not None:
                try:
                    store.put_summary(sid, transcript_id, job['attempt'], 'latest', {
                        'summary': cached['summary'], 'summary_html': cached['summary_html'],
                        'model': cached['model'], 'instruction': instruction,
                    })
                except (JobNotFound, InvalidTransition):
                    raise HTTPException(status_code=409, detail='Transcript changed during summarization')
                except CapacityExceeded:
                    raise HTTPException(status_code=503, detail='Could not save summary')
            result = {**cached, 'transcript_id': transcript_id, 'source_chars': len(text), 'cached': True}
            return JSONResponse(result, headers={'X-Whisper-Next-CSRF': next_token})
        response = await _web_summarize_uncached(request, next_token)
        if response.status_code == 200:
            result = json.loads(response.body)
            # Do not associate a result with an attempt which changed while
            # the provider was running.
            try:
                unchanged = job is None or store.get(sid, transcript_id)['attempt'] == job['attempt']
            except JobNotFound:
                unchanged = False
            if unchanged:
                transcription_cache.set(cache_key, {k: result[k] for k in
                    ('summary', 'summary_html', 'model', 'requested_model', 'served_model')}, expire=CACHE_TTL)
            result['cached'] = False
            return JSONResponse(result, headers={'X-Whisper-Next-CSRF': next_token})
        return response


async def _web_summarize_uncached(request: Request, next_token):
    """Summarise a stored transcript through OmniRoute.

    The browser sends only the transcript id: the text stays on the server and
    the OmniRoute key never leaves it.
    """
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    session_id = request.cookies.get(WEBUI_SESSION_COOKIE, "")
    transcript_id = str(payload.get("transcript_id") or "")
    store = _job_store()
    try:
        job = store.get(session_id, transcript_id)
    except JobNotFound:
        job = None
    attempt = job['attempt'] if job is not None else None
    text = (_transcript_text_from_payload(job['result']) if job['status'] == 'done' else None) if job is not None else _get_webui_transcript(session_id, transcript_id)
    if not text:
        raise HTTPException(status_code=404, detail="Unknown transcript id")
    _summary_rate_limit(session_id)

    byok = payload.get("byok")
    if byok is not None and not isinstance(byok, dict):
        raise HTTPException(status_code=400, detail="Некорректные настройки провайдера")
    api_key = _omniroute_api_key() if byok is None else None
    if byok is None and not api_key:
        raise HTTPException(status_code=503, detail="Summariser is not configured")

    instruction = str(payload.get("instruction") or "").strip()
    if len(instruction) > 500:
        instruction = instruction[:500]
    system_prompt = SUMMARY_SYSTEM_PROMPT
    if instruction:
        system_prompt = f"{SUMMARY_SYSTEM_PROMPT}\n\nДополнительное указание: {instruction}"

    # The gateway lives on the LAN: never route it through a system proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    candidates = [OMNIROUTE_SUMMARY_MODEL] + [
        model for model in OMNIROUTE_SUMMARY_FALLBACK_MODELS if model != OMNIROUTE_SUMMARY_MODEL
    ]

    summary = ""
    actual_model = ""
    served_model = ""
    if byok is not None:
        formats = {"anthropic", "chat-completions", "responses"}
        if (byok.get("apiFormat") not in formats or
            not all(isinstance(byok.get(k), str) and byok[k].strip() for k in ("baseUrl", "modelId", "apiKey")) or
            len(byok["apiKey"]) > 4096 or len(byok["modelId"]) > 256 or len(byok["baseUrl"]) > 2048):
            raise HTTPException(status_code=400, detail="Укажите адрес, модель и ключ провайдера")
        config = {"apiFormat": byok["apiFormat"], "baseUrl": byok["baseUrl"],
                  "modelId": byok["modelId"], "apiKey": byok["apiKey"],
                  "maxOutputTokens": OMNIROUTE_SUMMARY_MAX_TOKENS}
        summary = await _run_byok(config, system_prompt, text[:OMNIROUTE_SUMMARY_MAX_CHARS])
        actual_model = served_model = config["modelId"]
        candidates = []
    last_error = ""
    for model in candidates:
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text[:OMNIROUTE_SUMMARY_MAX_CHARS]},
            ],
            "stream": False,
            "max_tokens": OMNIROUTE_SUMMARY_MAX_TOKENS,
        }
        upstream = urllib.request.Request(
            f"{OMNIROUTE_BASE_URL}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            raw = await _summary_open(opener, upstream)
        except urllib.error.HTTPError as error:
            last_error = f"{model}: HTTP {error.code}"
            logger.warning("Summariser %s", last_error)
            continue
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = f"{model}: {error}"
            logger.warning("Summariser unreachable via %s: %s", model, error)
            continue

        try:
            parsed = json.loads(raw)
            choice = (parsed.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            summary = (message.get("content") or "").strip()
            if not summary:
                # Reasoning models may answer with reasoning only.
                summary = (message.get("reasoning_content") or "").strip()
            actual_model = parsed.get("model") or model
            served_model = model
        except (json.JSONDecodeError, IndexError, AttributeError):
            last_error = f"{model}: unexpected response"
            continue
        if summary:
            break
        last_error = f"{model}: empty summary"

    if not summary:
        raise HTTPException(status_code=502, detail=f"Summariser failed ({last_error})")

    summary_html = SUMMARY_MARKDOWN.render(summary)
    if attempt is not None:
        try:
            store.put_summary(session_id, transcript_id, attempt, 'latest', {
                'summary': summary, 'summary_html': summary_html, 'model': actual_model or served_model,
                'instruction': instruction,
            })
        except (JobNotFound, InvalidTransition):
            raise HTTPException(status_code=409, detail='Transcript changed during summarization')
        except CapacityExceeded:
            raise HTTPException(status_code=503, detail='Could not save summary')

    # Report the model that actually did the work, not the one we asked for.
    return JSONResponse(
        {
            "summary": summary,
            "summary_html": summary_html,
            "model": actual_model or served_model,
            "requested_model": byok["modelId"] if byok is not None else OMNIROUTE_SUMMARY_MODEL,
            "served_model": served_model,
            "transcript_id": transcript_id,
            "source_chars": len(text),
        },
        headers={"X-Whisper-Next-CSRF": next_token},
    )


@app.post("/v1/audio/transcriptions")
async def openai_transcribe(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    language: Optional[str] = Form(None),
    temperature: float = Form(0.0),
    response_format: str = Form("json"),
    chunking_strategy: Optional[str] = Form(None),
    stream: bool = Form(False),
    vad_filter: bool = Form(True),
    cache: bool = Form(True),
    diarize: bool = Form(False),
    comment: Optional[str] = Form(None),
):
    """OpenAI-compatible transcription endpoint"""
    _require_openai_api_key(request)
    if isinstance(chunking_strategy, str) and chunking_strategy != "auto":
        return JSONResponse({"error": "Supported chunking_strategy: auto"}, status_code=400)
    diarize = diarize is True or response_format == "diarized_json"
    client_comment = normalize_comment(comment)

    # Map OpenAI model name to actual model
    actual_model = OPENAI_MODEL_MAP.get(model, model)

    if actual_model != OPENAI_WHISPER_INTERNAL_MODEL and actual_model not in MODEL_PRIORITY:
        return JSONResponse({"error": f"Unsupported model: {actual_model}"}, status_code=400)

    # Check file format
    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".webm"]:
        return JSONResponse({"error": "Unsupported audio format"}, status_code=400)

    # Read audio file
    audio_bytes = await file.read()
    if not audio_bytes:
        return JSONResponse({"error": "Empty audio file"}, status_code=400)

    # Create cache key for this request
    audio_hash = hash_bytes(audio_bytes)
    # Remember who sent this file. The comment stays beside the cache and never
    # changes the cache key, so the same file stays a cache hit for everyone.
    if client_comment:
        get_attribution_store().record(audio_hash, client_comment, requested_model=model)
    cache_key_model = model if actual_model == OPENAI_WHISPER_INTERNAL_MODEL else actual_model
    cache_key = (
        transcription_cache_key("openai", cache_key_model, language, audio_hash, vad_filter=vad_filter, diarize=diarize)
        if cache
        else None
    )

    # Check cache
    cached_result = _get_cached_result(cache_key) if cache_key else None
    if cached_result is not None:
        logger.info(f"[CACHE HIT] {cache_key}")
        append_cache_metric(cached_result, model, client_comment)
        cached_text = cached_result.get("text", "")
        if stream:
            async def cached_stream_generator():
                if response_format == "diarized_json":
                    for i, segment in enumerate(cached_result.get("segments", [])):
                        yield _openai_sse(diarized_segment(segment, i))
                if cached_text:
                    yield _openai_sse(_openai_delta_event(cached_text))
                yield _openai_sse(_openai_done_event(cached_text, transcription_response_payload(cached_result, response_format, client_comment)))

            return StreamingResponse(
                cached_stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "X-Content-Type-Options": "nosniff",
                },
            )
        return JSONResponse(transcription_response_payload(cached_result, response_format, client_comment))

    # Queue the request
    if not _worker_accepts_requests():
        return JSONResponse({'error': 'Model worker unavailable', 'code': 'worker_stopped'}, status_code=503)
    request_id = uuid4().hex
    loop = asyncio.get_event_loop()
    if stream:
        queue = asyncio.Queue()
        pending_streams[request_id] = queue
    else:
        future = loop.create_future()
        pending_results[request_id] = future

    app.state.request_queue.put({
        "request_id": request_id,
        "audio_bytes": audio_bytes,
        "model": actual_model,
        "requested_model": model,
        "language": language,
        "cache_key": cache_key,
        "beam_size": 5,
        "temperature": temperature,
        "stream": stream,
        "words": False,
        "vad_filter": vad_filter,
        "diarize": diarize,
        "client_comment": client_comment,
        "queued_at": time.perf_counter(),
    })

    if stream:
        async def openai_stream_generator():
            final_text = ""
            segment_index = 0
            deadline = time.monotonic() + (int(os.getenv("WHISPER_DIARIZATION_TIMEOUT_SECONDS", "86400")) if diarize else 600)
            try:
                while True:
                    if time.monotonic() >= deadline:
                        yield _openai_sse({"type": "error", "error": {"message": "Transcription timeout"}})
                        break
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=min(15, deadline - time.monotonic()))
                    except asyncio.TimeoutError:
                        yield ": processing\n\n"
                        continue
                    if item is None:
                        break
                    if "error" in item:
                        yield _openai_sse({"type": "error", "error": {"message": item["error"]}})
                        break
                    if "segment" in item:
                        segment = item["segment"]
                        text = segment.get("text", "")
                        final_text += text
                        if response_format == "diarized_json":
                            yield _openai_sse(diarized_segment(segment, segment_index))
                            segment_index += 1
                        yield _openai_sse(_openai_delta_event(text, segment.get("id")))
                    elif "result" in item:
                        result = item["result"]
                        yield _openai_sse(
                            _openai_done_event(result.get("text", final_text), transcription_response_payload(result, response_format, client_comment))
                        )
            finally:
                pending_streams.pop(request_id, None)

        return StreamingResponse(
            openai_stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Content-Type-Options": "nosniff",
            },
        )

    try:
        result = await asyncio.wait_for(future, timeout=(int(os.getenv("WHISPER_DIARIZATION_TIMEOUT_SECONDS", "86400")) if diarize else 600))
        return JSONResponse(transcription_response_payload(result, response_format, client_comment))
    except asyncio.TimeoutError:
        pending_results.pop(request_id, None)
        return JSONResponse({"error": "Transcription timeout"}, status_code=504)
    except Exception as e:
        pending_results.pop(request_id, None)
        logger.exception("OpenAI transcribe error:")
        if isinstance(e, WorkerStopped):
            return JSONResponse({'error': str(e), 'code': 'worker_stopped'}, status_code=503)
        if isinstance(e, GpuOutOfMemoryError):
            return JSONResponse(
                {"error": str(e), "code": e.code, "retryable": True},
                status_code=503,
                headers={"Retry-After": "30"},
            )
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/v1/attributions")
async def openai_attributions(request: Request, audio_hash: Optional[str] = Query(None), comment: Optional[str] = Query(None)):
    """Show which client sent which file, based on opt-in client comments."""
    _require_openai_api_key(request)
    if not audio_hash and not comment:
        return JSONResponse({"error": "Provide audio_hash or comment"}, status_code=400)
    return JSONResponse(get_attribution_store().lookup(audio_hash=audio_hash, comment=comment))


@app.get("/v1/models")
async def openai_models(request: Request):
    """OpenAI-compatible models list."""
    _require_openai_api_key(request)
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": created,
                "owned_by": "local-whisperserver",
            }
            for model_id in _get_openai_model_ids()
        ],
    }


@app.get("/models")
async def models():
    return list(MODEL_PRIORITY.keys())


@app.get("/status")
async def status():
    try:
        queue_size = app.state.request_queue.qsize()
    except (OSError, ValueError, AttributeError, NotImplementedError):
        queue_size = None
    usage = dict(app.state.model_usage) if hasattr(app.state, "model_usage") else {}
    bot_process = getattr(app.state, "bot_process", None)
    bot_status = {
        "enabled": _bot_enabled(),
        "running": bool(bot_process and bot_process.poll() is None),
        "pid": bot_process.pid if bot_process else None,
        "returncode": bot_process.poll() if bot_process else None,
    }
    return {"queue_size": queue_size, "model_usage": usage, "telegram_bot": bot_status}


@app.get("/", response_class=HTMLResponse)
async def webui(request: Request):
    _prune_webui_sessions()
    sid = request.cookies.get(WEBUI_SESSION_COOKIE, "")
    if _load_webui_session(sid):
        token = _rotate_webui_token(sid)
    else:
        sid, token = _create_webui_session()
    summary_config = json.dumps([SUMMARY_SYSTEM_PROMPT, OMNIROUTE_BASE_URL, OMNIROUTE_SUMMARY_MODEL,
                                 OMNIROUTE_SUMMARY_FALLBACK_MODELS, OMNIROUTE_SUMMARY_MAX_CHARS,
                                 OMNIROUTE_SUMMARY_MAX_TOKENS], ensure_ascii=False)
    summary_cache_version = hashlib.sha256(summary_config.encode()).hexdigest()
    html = (TEMPLATE_PATH.read_text()
            .replace("__WEBUI_CSRF_TOKEN__", json.dumps(token))
            .replace("__DEFAULT_MODEL__", json.dumps(DEFAULT_MODEL))
            .replace("__SUMMARY_CACHE_VERSION__", json.dumps(summary_cache_version)))
    response = HTMLResponse(html)
    _set_webui_session_cookie(response, request, sid)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7653)
