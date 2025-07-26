import asyncio
import multiprocessing as mp
import tempfile
from fastapi import FastAPI, UploadFile, File, Query, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
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

# Cache with 10GB limit and 1 month TTL for entries
CACHE_TTL = 60 * 60 * 24 * 30
cache = Cache("whisper_cache", size_limit=10 * 1024 ** 3)

request_queue = mp.Queue()
response_queue = mp.Queue()
pending_results = {}
pending_streams = {}
model_usage = None

app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup_event()
    yield
    shutdown_event()
    app.state.response_listener_task.cancel()

app.router.lifespan_context = lifespan

def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

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
                beam_size = request.get("beam_size",5)
                stream = request.get("stream")

                try:
                    if model_name not in model_cache:
                        logger.info(f"Loading model: {model_name}")
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        compute = "float16" if device == "cuda" else "int8"
                        model_cache[model_name] = WhisperModel(model_name, device=device, compute_type=compute)
                    usage_dict[model_name] = usage_dict.get(model_name, 0) + 1

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

async def response_listener(stop_event):
    while not stop_event.is_set():
        try:
            result = await asyncio.get_event_loop().run_in_executor(app.state.executor, response_queue.get)
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
    stop_event = mp.Event()
    manager = mp.Manager()
    model_usage = manager.dict()
    process = mp.Process(target=model_worker, args=(request_queue, response_queue, stop_event, model_usage))
    process.start()
    executor = ThreadPoolExecutor()
    app.state.stop_event = stop_event
    app.state.model_process = process
    app.state.executor = executor
    app.state.response_listener_task = asyncio.create_task(response_listener(stop_event))
    app.state.model_usage = model_usage

def shutdown_event():
    logger.info("Остановка процесса...")
    app.state.stop_event.set()
    app.state.model_process.join()
    logger.info("Модель завершена.")

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Query("base"),
    language: Optional[str] = Query(None),
    beam_size: Optional[int] = Query(5),
    stream: bool = Query(False),
    words: bool = Query(False),
    api_key:str=Query(...),
):
    if api_key not in ALLOWED_API_KEYS:
        return JSONResponse({"error": "Invalid API key"}, status_code=403)
    if model not in MODEL_PRIORITY:
        return JSONResponse({"error": "Unsupported model"}, status_code=400)

    audio_bytes = await file.read()
    audio_hash = hash_bytes(audio_bytes)
    cache_key = f"{model}:{language}:{words}:{audio_hash}"

    if cache_key in cache and not stream:
        logger.info(f"[CACHE HIT] {cache_key}")
        return cache[cache_key]
    if cache_key in cache and stream:
        logger.info(f"[CACHE HIT] {cache_key}")
        async def cached():
            for segment in cache[cache_key]['segments']:
                yield json.dumps({"segment": segment}) + "\n"
        return StreamingResponse(cached(), media_type="application/json")

    request_id = id(audio_bytes)
    loop = asyncio.get_event_loop()
    if stream:
        queue = asyncio.Queue()
        pending_streams[request_id] = queue
    else:
        future = loop.create_future()
        pending_results[request_id] = future

    request_queue.put({
        "request_id": request_id,
        "audio_bytes": audio_bytes,
        "model": model,
        "language": language,
        "cache_key": cache_key,
        "beam_size": beam_size,
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
        return StreamingResponse(generator(), media_type="application/x-ndjson",
    headers=headers)
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

@app.get("/models")
async def models():
    return list(MODEL_PRIORITY.keys())


@app.get("/status")
async def status():
    queue_size = request_queue.qsize()
    usage = dict(app.state.model_usage) if hasattr(app.state, "model_usage") else {}
    return {"queue_size": queue_size, "model_usage": usage}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7653)
