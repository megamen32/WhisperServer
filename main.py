import asyncio
import importlib.util
import multiprocessing as mp
import signal
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Query, Form, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import logging
import time
import os
import hashlib
from contextlib import asynccontextmanager
from faster_whisper import WhisperModel
from diskcache import Cache
from dotenv import load_dotenv
import torch

load_dotenv()

# Логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Поддерживаемые модели и их приоритет
MODEL_PRIORITY = {
    "tiny": 1,
    "base": 2,
    "small": 3,
    "medium": 4,
    "distil-large-v3": 5,
    "large-v3": 6,
    "large-v2": 7,
    "large": 8,
}

ALLOWED_API_KEYS = {os.getenv("API_KEY", "bad-key")}

# Map OpenAI model names to faster-whisper models.
# whisper-1 is resolved inside the model worker: it reuses an already loaded
# model when its priority is >= OPENAI_WHISPER_MIN_MODEL.
OPENAI_WHISPER_ALIAS = "whisper-1"
OPENAI_WHISPER_INTERNAL_MODEL = "__openai_whisper_1__"
OPENAI_WHISPER_MIN_MODEL = os.getenv("OPENAI_WHISPER_MIN_MODEL", "medium")
OPENAI_DEFAULT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "large-v3")
OPENAI_MODEL_MAP = {
    OPENAI_WHISPER_ALIAS: OPENAI_WHISPER_INTERNAL_MODEL,
}


def _model_priority(model_id):
    return MODEL_PRIORITY.get(model_id, 999)


def _min_openai_whisper_priority():
    if OPENAI_WHISPER_MIN_MODEL not in MODEL_PRIORITY:
        logger.warning(
            "Unknown OPENAI_WHISPER_MIN_MODEL=%s; falling back to medium",
            OPENAI_WHISPER_MIN_MODEL,
        )
    return MODEL_PRIORITY.get(OPENAI_WHISPER_MIN_MODEL, MODEL_PRIORITY["medium"])


def _select_openai_whisper_model(model_cache):
    """Choose a concrete faster-whisper model for the OpenAI whisper-1 alias."""
    min_priority = _min_openai_whisper_priority()
    loaded_candidates = [
        model_id
        for model_id in model_cache.keys()
        if MODEL_PRIORITY.get(model_id, -1) >= min_priority
    ]
    if loaded_candidates:
        return sorted(loaded_candidates, key=_model_priority, reverse=True)[0]

    if _model_priority(OPENAI_DEFAULT_MODEL) >= min_priority:
        return OPENAI_DEFAULT_MODEL

    # Default is smaller than the configured minimum: load the minimum model.
    return OPENAI_WHISPER_MIN_MODEL if OPENAI_WHISPER_MIN_MODEL in MODEL_PRIORITY else "medium"


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
    env.setdefault("TG_BOT_MODEL", os.getenv("MODEL", "base"))
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
                model_name = request["model"]
                audio_bytes = request["audio_bytes"]
                request_id = request["request_id"]
                lang = request.get("language")
                cache_key = request.get("cache_key")
                beam_size = request.get("beam_size", 5)
                temperature = request.get("temperature", 0.0)
                stream = request.get("stream")

                try:
                    requested_model_name = model_name
                    if requested_model_name == OPENAI_WHISPER_INTERNAL_MODEL:
                        model_name = _select_openai_whisper_model(model_cache)
                        logger.info(
                            "Resolved %s to %s (min=%s, loaded=%s)",
                            OPENAI_WHISPER_ALIAS,
                            model_name,
                            OPENAI_WHISPER_MIN_MODEL,
                            sorted(model_cache.keys(), key=_model_priority),
                        )

                    if model_name not in model_cache:
                        logger.info(f"Loading model: {model_name}")
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        compute = "float16" if device == "cuda" else "int8"
                        model_cache[model_name] = WhisperModel(model_name, device=device, compute_type=compute)
                    usage_dict[model_name] = usage_dict.get(model_name, 0) + 1
                    if requested_model_name != model_name:
                        usage_dict[requested_model_name] = usage_dict.get(requested_model_name, 0) + 1

                    model = model_cache[model_name]

                    with tempfile.NamedTemporaryFile(delete=False) as temp:
                        temp.write(audio_bytes)
                        temp.flush()
                        temp_path = temp.name

                    try:
                        segments, info = model.transcribe(
                            temp_path,
                            beam_size=beam_size,
                            language=lang,
                            temperature=temperature,
                            word_timestamps=request.get("words", False),
                            condition_on_previous_text=False,
                        )
                        all_segments = []
                        all_words = []

                        for seg in segments:
                            seg_data = {
                                "start": seg.start,
                                "end": seg.end,
                                "text": seg.text,
                            }
                            logger.info(f'{model_name} processed segment: {seg_data}')
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

                    finally:
                        os.remove(temp_path)

                except Exception as e:
                    logger.exception("Ошибка в worker:")
                    response_queue.put({
                        "request_id": request_id,
                        "error": str(e)
                    })

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

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Query("base"),
    language: Optional[str] = Query(None),
    beam_size: Optional[int] = Query(5),
    temperature: float = Query(0.0),
    stream: bool = Query(False),
    words: bool = Query(False),
    api_key: str = Query(...),
):
    if api_key not in ALLOWED_API_KEYS:
        return JSONResponse({"error": "Invalid API key"}, status_code=403)
    actual_model = OPENAI_MODEL_MAP.get(model, model)
    if actual_model != OPENAI_WHISPER_INTERNAL_MODEL and actual_model not in MODEL_PRIORITY:
        return JSONResponse({"error": "Unsupported model"}, status_code=400)

    audio_bytes = await file.read()
    audio_hash = hash_bytes(audio_bytes)
    cache_key = f"{model}:{language}:{words}:{audio_hash}"

    if cache_key in cache and not stream:
        logger.info(f"[CACHE HIT] {cache_key}")
        return cache[cache_key]
    if cache_key in cache and stream:
        logger.info(f"[CACHE HIT] {cache_key}")
        cached_result = cache[cache_key]
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
        "language": language,
        "cache_key": cache_key,
        "beam_size": beam_size,
        "temperature": temperature,
        "stream": stream,
        "words": words,
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

@app.post("/v1/audio/transcriptions")
async def openai_transcribe(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    temperature: float = Form(0.0),
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
    cache_key = f"openai:{cache_key_model}:{language}:{audio_hash}"

    # Check cache
    if cache_key in cache:
        logger.info(f"[CACHE HIT] {cache_key}")
        cached_result = cache[cache_key]
        return JSONResponse({"text": cached_result.get("text", "")})

    # Queue the request
    request_id = id(audio_bytes)
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    pending_results[request_id] = future

    app.state.request_queue.put({
        "request_id": request_id,
        "audio_bytes": audio_bytes,
        "model": actual_model,
        "language": language,
        "cache_key": cache_key,
        "beam_size": 5,
        "temperature": temperature,
        "stream": False,
        "words": False,
    })

    try:
        result = await asyncio.wait_for(future, timeout=600)
        return JSONResponse({"text": result.get("text", "")})
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
async def webui():
    api_key_value = json.dumps(os.getenv("API_KEY", "bad-key"))
    template = TEMPLATE_PATH.read_text()
    html_content = template.replace("__API_KEY__", api_key_value)
    return HTMLResponse(html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7653)
