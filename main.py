import asyncio
import importlib.util
import multiprocessing as mp
import signal
import subprocess
import sys
import tempfile
import secrets
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Query, Form, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse, Response
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import logging
import time
import os
import hashlib
from contextlib import asynccontextmanager
from faster_whisper import WhisperModel
from parakeet_backend import ParakeetModel
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
from model_registry import MODEL_PRIORITY, MODEL_SUPERSEDES
from model_registry import select_compatible_loaded_model as _select_compatible_loaded_model

load_dotenv()

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("MODEL", "parakeet-v3")
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
}
BROKER_CPU_CAPABLE = {"tiny", "base", "small", "parakeet-v3"}


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

ALLOWED_API_KEYS = {os.getenv("API_KEY", "bad-key")}
WEBUI_SESSION_COOKIE = "whisper_web_session"
WEBUI_CSRF_HEADER = "x-whisper-csrf"
WEBUI_SESSION_TTL_SECONDS = int(os.getenv("WEBUI_SESSION_TTL_SECONDS", "7200"))
_webui_sessions = {}


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
    return sid, token


def _rotate_webui_token(sid):
    token = _new_webui_token()
    _webui_sessions[sid] = {"token": token, "expires": time.time() + WEBUI_SESSION_TTL_SECONDS}
    return token


def _same_origin_request(request: Request):
    origin = request.headers.get("origin")
    if not origin:
        return False
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme).split(",", 1)[0].strip()
    host = request.headers.get("x-forwarded-host", request.headers.get("host", "")).split(",", 1)[0].strip()
    expected = f"{scheme}://{host}"
    return origin.rstrip("/") == expected.rstrip("/")


def _consume_webui_token(request: Request):
    _prune_webui_sessions()
    sid = request.cookies.get(WEBUI_SESSION_COOKIE, "")
    supplied = request.headers.get(WEBUI_CSRF_HEADER, "")
    session = _webui_sessions.get(sid)
    if not sid or not supplied or not session or supplied != session.get("token"):
        raise HTTPException(status_code=403, detail="Invalid or expired web UI token")
    if not _same_origin_request(request):
        _webui_sessions.pop(sid, None)
        raise HTTPException(status_code=403, detail="Invalid request origin")
    return _rotate_webui_token(sid)


# Map OpenAI model names to local speech-to-text models.
OPENAI_WHISPER_ALIAS = "whisper-1"
OPENAI_WHISPER_INTERNAL_MODEL = "__openai_whisper_1__"
OPENAI_DEFAULT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "parakeet-v3")
if OPENAI_DEFAULT_MODEL not in MODEL_PRIORITY:
    logger.warning("Unknown OPENAI_DEFAULT_MODEL=%s; falling back to parakeet-v3", OPENAI_DEFAULT_MODEL)
    OPENAI_DEFAULT_MODEL = "parakeet-v3"

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
    OPENAI_WHISPER_ALIAS: OPENAI_WHISPER_INTERNAL_MODEL,
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


def transcription_cache_key(prefix: str, model: str, language: str, audio_hash: str, *, vad_filter: bool, words: bool | None = None) -> str:
    """Build a cache key that never mixes VAD-enabled and raw decoding."""
    words_part = "" if words is None else f":{words}"
    return f"{prefix}:vad-v1:{int(vad_filter)}:{model}:{language}{words_part}:{audio_hash}"


def _select_openai_whisper_model(model_cache: dict) -> str:
    """Reuse the strongest loaded model, or load Parakeet v3 by default."""
    loaded_models = [model_id for model_id in model_cache if model_id in MODEL_PRIORITY]
    if loaded_models:
        return max(loaded_models, key=MODEL_PRIORITY.get)
    return OPENAI_DEFAULT_MODEL


def _load_model(model_name: str, device: str, compute_type: str):
    """Load a registered model using its native inference backend."""
    if model_name == "parakeet-v3":
        return ParakeetModel.from_pretrained(device=device)
    return WhisperModel(model_name, device=device, compute_type=compute_type)



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
cache = Cache("whisper_cache", size_limit=10 * 1024 ** 3)


def _get_cached_result(cache_key: str):
    """Return a cached transcription result and discard legacy invalid values."""
    if cache_key not in cache:
        return None

    cached_result = cache[cache_key]
    if isinstance(cached_result, dict):
        return cached_result

    # Older cache writers could store a boolean success marker. Treat it as a
    # miss so OpenAI-compatible clients receive a real transcription response.
    logger.warning("Discarding invalid transcription cache entry: %s", cache_key)
    cache.delete(cache_key)
    return None

request_queue = None
response_queue = None
pending_results = {}
pending_streams = {}
model_usage = None

TEMPLATE_PATH = Path(__file__).parent / "templates" / "webui.html"
BOT_SCRIPT_PATH = Path(__file__).parent / "telegram_bot.py"

app = FastAPI()

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

def model_worker(request_queue, response_queue, stop_event, usage_dict):
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
                try:
                    with tempfile.NamedTemporaryFile(delete=False) as temp:
                        temp.write(audio_bytes)
                        temp.flush()
                        temp_path = temp.name
                    audio_seconds = audio_duration_seconds(temp_path)

                    if requested_model_name == OPENAI_WHISPER_INTERNAL_MODEL:
                        model_name = _select_openai_whisper_model(model_cache)
                        model_substituted = model_name != "parakeet-v3"
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
                        logger.info("Preparing broker-managed model: %s", model_name)
                        if ManagedModel is not None:
                            model_cache[model_name] = ManagedModel(
                                model_id=f"whisper-{model_name}",
                                loader=lambda mn=model_name: _load_model(
                                    mn,
                                    device="cuda" if torch.cuda.is_available() else "cpu",
                                    compute_type="float16" if torch.cuda.is_available() else "int8",
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
                        else:
                            logger.warning("cudabroker_client not available, loading model without broker")
                            device = "cuda" if torch.cuda.is_available() else "cpu"
                            compute = "float16" if device == "cuda" else "int8"
                            model_cache[model_name] = _load_model(model_name, device=device, compute_type=compute)

                    model_entry = model_cache[model_name]
                    if ManagedModel is not None and isinstance(model_entry, ManagedModel):
                        model = model_entry.acquire()
                    else:
                        model = model_entry
                    load_ms = (time.perf_counter() - load_started) * 1000 if cold_load else 0.0
                    usage_dict[model_name] = usage_dict.get(model_name, 0) + 1
                    if public_requested_model != model_name:
                        usage_dict[public_requested_model] = usage_dict.get(public_requested_model, 0) + 1

                    if ManagedModel is not None and isinstance(model_entry, ManagedModel):
                        model_entry.touch(active=True)
                    inference_started = time.perf_counter()
                    segments, info = model.transcribe(
                        temp_path,
                        beam_size=beam_size,
                        language=lang,
                        temperature=temperature,
                        word_timestamps=request.get("words", False),
                        condition_on_previous_text=False,
                        **build_vad_transcribe_options(vad_filter),
                        no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
                        log_prob_threshold=WHISPER_LOG_PROB_THRESHOLD,
                        compression_ratio_threshold=WHISPER_COMPRESSION_RATIO_THRESHOLD,
                    )
                    inference_ms = (time.perf_counter() - inference_started) * 1000
                    all_segments = []
                    all_words = []

                    for seg in segments:
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
                        all_segments.append(seg_data)
                        if request.get("words", False) and getattr(seg, "words", None):
                            words = [
                                {"start": w.start, "end": w.end, "word": w.word}
                                for w in seg.words
                            ]
                            all_words.extend(words)
                        else:
                            words = []

                        if stream:
                            payload = {"request_id": request_id, "segment": seg_data}
                            if request.get("words", False):
                                payload["words"] = words
                            response_queue.put(payload)

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
                    if request.get("words", False):
                        result["words"] = all_words

                    if cache_key:
                        cache.set(cache_key, result, expire=CACHE_TTL)

                    response_queue.put({
                        "request_id": request_id,
                        "result": result,
                        "final": True,
                    })
                    success = True

                except Exception as e:
                    error_text = str(e)
                    logger.exception("Ошибка в worker:")
                    response_queue.put({
                        "request_id": request_id,
                        "error": str(e)
                    })
                finally:
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
                    if error_text:
                        metric["error"] = error_text
                    append_metric(metric)

        except Exception as e:
            logger.exception("Фатальная ошибка в loop:")
            time.sleep(1)

async def response_listener(stop_event, response_queue):
    loop = asyncio.get_running_loop()
    while True:
        try:
            result = await loop.run_in_executor(app.state.executor, response_queue.get)
            if result is None or result.get("__shutdown__"):
                break
            logger.info(f'response_listener get from queue result {result}')
            request_id = result["request_id"]
            if request_id in pending_results:
                future = pending_results.pop(request_id)
                if "error" in result:
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

async def startup_event():
    global model_usage
    ctx = mp.get_context()
    stop_event = ctx.Event()
    manager = ctx.Manager()
    model_usage = manager.dict()
    request_queue = ctx.Queue()
    response_queue = ctx.Queue()
    process = ctx.Process(
        target=model_worker,
        args=(request_queue, response_queue, stop_event, model_usage),
    )
    process.start()
    executor = ThreadPoolExecutor()
    app.state.stop_event = stop_event
    app.state.model_process = process
    app.state.executor = executor
    app.state.request_queue = request_queue
    app.state.response_queue = response_queue
    app.state.manager = manager
    app.state.response_listener_task = asyncio.create_task(
        response_listener(stop_event, response_queue)
    )
    app.state.model_usage = model_usage
    app.state.lifecycle_stop = asyncio.Event()
    app.state.bot_process = start_bot_subprocess()
    app.state.bot_monitor_task = asyncio.create_task(monitor_bot_subprocess())

async def shutdown_event():
    logger.info("Остановка процесса...")
    if hasattr(app.state, "lifecycle_stop"):
        app.state.lifecycle_stop.set()
    if hasattr(app.state, "bot_monitor_task"):
        app.state.bot_monitor_task.cancel()
        try:
            await app.state.bot_monitor_task
        except asyncio.CancelledError:
            pass
    await stop_bot_subprocess()
    app.state.stop_event.set()
    try:
        app.state.response_queue.put({"__shutdown__": True})
    except Exception:
        pass

    if hasattr(app.state, "response_listener_task"):
        app.state.response_listener_task.cancel()
        try:
            await app.state.response_listener_task
        except asyncio.CancelledError:
            pass

    process = app.state.model_process
    process.join(timeout=10)
    if process.is_alive():
        logger.warning("Модель не завершилась вовремя, отправляем terminate.")
        process.terminate()
        process.join(timeout=5)
    else:
        logger.info("Модель завершена.")

    app.state.executor.shutdown(wait=False)

    for q in (app.state.request_queue, app.state.response_queue):
        try:
            q.close()
            q.join_thread()
        except Exception:
            pass

    if hasattr(app.state, "manager"):
        try:
            app.state.manager.shutdown()
        except Exception:
            pass

async def _transcribe_impl(
    file: UploadFile,
    model: str,
    language: Optional[str],
    beam_size: Optional[int],
    temperature: float,
    stream: bool,
    words: bool,
    vad_filter: bool,
):
    actual_model = OPENAI_MODEL_MAP.get(model, model)
    if actual_model != OPENAI_WHISPER_INTERNAL_MODEL and actual_model not in MODEL_PRIORITY:
        return JSONResponse({"error": "Unsupported model"}, status_code=400)

    audio_bytes = await file.read()
    audio_hash = hash_bytes(audio_bytes)
    cache_key = transcription_cache_key("native", model, language, audio_hash, vad_filter=vad_filter, words=words)

    cached_result = _get_cached_result(cache_key)
    if cached_result is not None and not stream:
        logger.info(f"[CACHE HIT] {cache_key}")
        append_cache_metric(cached_result, model)
        return cached_result
    if cached_result is not None and stream:
        logger.info(f"[CACHE HIT] {cache_key}")
        append_cache_metric(cached_result, model)
        async def cached():
            for segment in cached_result.get("segments", []):
                yield json.dumps({"segment": segment}) + "\n"
            yield json.dumps({"result": cached_result}) + "\n"
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # для nginx
            "X-Content-Type-Options": "nosniff",
        }
        return StreamingResponse(cached(), media_type="application/x-ndjson", headers=headers)

    request_id = id(audio_bytes)
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
        "beam_size": beam_size,
        "temperature": temperature,
        "stream": stream,
        "words": words,
        "vad_filter": vad_filter,
        "queued_at": time.perf_counter(),
    })

    if stream:
        async def generator():
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield json.dumps(item) + "\n"
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # для nginx
            "X-Content-Type-Options": "nosniff",
        }
        return StreamingResponse(generator(), media_type="application/x-ndjson", headers=headers)
    else:
        try:
            result = await asyncio.wait_for(future, timeout=600)
            return result
        except asyncio.TimeoutError:
            pending_results.pop(request_id, None)
            return JSONResponse({"error": "Timeout"}, status_code=504)
        except Exception as e:
            pending_results.pop(request_id, None)
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
):
    if api_key not in ALLOWED_API_KEYS:
        return JSONResponse({"error": "Invalid API key"}, status_code=403)
    return await _transcribe_impl(file, model, language, beam_size, temperature, stream, words, vad_filter)


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
):
    """First-party Web UI tunnel: API_KEY stays server-side; CSRF token is one-use."""
    next_token = _consume_webui_token(request)
    response = await _transcribe_impl(file, model, language, beam_size, temperature, stream, words, vad_filter)
    if not hasattr(response, "headers"):
        response = JSONResponse(response)
    response.headers["X-Whisper-Next-CSRF"] = next_token
    return response

@app.post("/v1/audio/transcriptions")
async def openai_transcribe(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    temperature: float = Form(0.0),
    response_format: str = Form("json"),
    stream: bool = Form(False),
    vad_filter: bool = Form(True),
    use_cache: bool = Form(True, alias="cache"),
):
    """OpenAI-compatible transcription endpoint"""
    _require_openai_api_key(request)

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
    cache_key_model = model if actual_model == OPENAI_WHISPER_INTERNAL_MODEL else actual_model
    cache_key = (
        transcription_cache_key("openai", cache_key_model, language, audio_hash, vad_filter=vad_filter)
        if use_cache
        else None
    )

    # Check cache
    cached_result = _get_cached_result(cache_key) if cache_key else None
    if cached_result is not None:
        logger.info(f"[CACHE HIT] {cache_key}")
        append_cache_metric(cached_result, model)
        cached_text = cached_result.get("text", "")
        if stream:
            async def cached_stream_generator():
                if cached_text:
                    yield _openai_sse(_openai_delta_event(cached_text))
                yield _openai_sse(_openai_done_event(cached_text, model_metadata(cached_result)))

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
        return JSONResponse(transcription_response_payload(cached_result))

    # Queue the request
    request_id = id(audio_bytes)
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
        "queued_at": time.perf_counter(),
    })

    if stream:
        async def openai_stream_generator():
            final_text = ""
            try:
                while True:
                    item = await asyncio.wait_for(queue.get(), timeout=600)
                    if item is None:
                        break
                    if "segment" in item:
                        segment = item["segment"]
                        text = segment.get("text", "")
                        final_text += text
                        yield _openai_sse(_openai_delta_event(text, segment.get("id")))
                    elif "result" in item:
                        result = item["result"]
                        yield _openai_sse(
                            _openai_done_event(result.get("text", final_text), model_metadata(result))
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
        result = await asyncio.wait_for(future, timeout=600)
        return JSONResponse(transcription_response_payload(result))
    except asyncio.TimeoutError:
        pending_results.pop(request_id, None)
        return JSONResponse({"error": "Transcription timeout"}, status_code=504)
    except Exception as e:
        pending_results.pop(request_id, None)
        logger.exception("OpenAI transcribe error:")
        return JSONResponse({"error": str(e)}, status_code=500)

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
    queue_size = app.state.request_queue.qsize()
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
    sid, token = _create_webui_session()
    html = TEMPLATE_PATH.read_text().replace("__WEBUI_CSRF_TOKEN__", json.dumps(token))
    response = HTMLResponse(html)
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme).split(",", 1)[0].strip()
    response.set_cookie(
        WEBUI_SESSION_COOKIE,
        sid,
        max_age=WEBUI_SESSION_TTL_SECONDS,
        httponly=True,
        secure=(scheme == "https"),
        samesite="strict",
    )
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7653)
