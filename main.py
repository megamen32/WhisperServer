import asyncio
import multiprocessing as mp
import tempfile
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import logging
import time
import os
import hashlib
from contextlib import asynccontextmanager
from faster_whisper import WhisperModel
from diskcache import Cache

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

# Кэш
cache = Cache("whisper_cache")

# Очереди
request_queue = mp.Queue()
response_queue = mp.Queue()
pending_results = {}

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

def model_worker(request_queue, response_queue, stop_event):
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

                try:
                    if model_name not in model_cache:
                        logger.info(f"Loading model: {model_name}")
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        compute = "float16" if device == "cuda" else "int8"
                        model_cache[model_name] = WhisperModel(model_name, device=device, compute_type=compute)

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
                            word_timestamps=True,
                            condition_on_previous_text=False
                        )
                        all_segments = []
                        all_words = []

                        for seg in segments:
                            seg_data = {
                                "start": seg.start,
                                "end": seg.end,
                                "text": seg.text
                            }
                            all_segments.append(seg_data)
                            if seg.words:
                                all_words.extend([{
                                    "start": w.start,
                                    "end": w.end,
                                    "word": w.word
                                } for w in seg.words])

                        result = {
                            "text": " ".join([s["text"] for s in all_segments]),
                            "segments": all_segments,
                            "words": all_words,
                            "language": info.language,
                            "language_probability": info.language_probability
                        }

                        if cache_key:
                            cache[cache_key] = result

                        response_queue.put({
                            "request_id": request_id,
                            "result": result
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
            request_id = result["request_id"]
            if request_id in pending_results:
                future = pending_results.pop(request_id)
                if "error" in result:
                    future.set_exception(Exception(result["error"]))
                else:
                    future.set_result(result["result"])
        except asyncio.CancelledError:
            break

async def startup_event():
    stop_event = mp.Event()
    process = mp.Process(target=model_worker, args=(request_queue, response_queue, stop_event))
    process.start()
    executor = ThreadPoolExecutor()
    app.state.stop_event = stop_event
    app.state.model_process = process
    app.state.executor = executor
    app.state.response_listener_task = asyncio.create_task(response_listener(stop_event))

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
    beam_size: Optional[int] = Query(5)
):
    if model not in MODEL_PRIORITY:
        return JSONResponse({"error": "Unsupported model"}, status_code=400)

    audio_bytes = await file.read()
    audio_hash = hash_bytes(audio_bytes)
    cache_key = f"{model}:{language}:{audio_hash}"

    if cache_key in cache:
        logger.info(f"[CACHE HIT] {cache_key}")
        return cache[cache_key]

    request_id = id(audio_bytes)
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    pending_results[request_id] = future

    request_queue.put({
        "request_id": request_id,
        "audio_bytes": audio_bytes,
        "model": model,
        "language": language,
        "cache_key": cache_key,
        "beam_size": beam_size
    })

    try:
        result = await asyncio.wait_for(future, timeout=600)
        return result
    except asyncio.TimeoutError:
        pending_results.pop(request_id, None)
        return JSONResponse({"error": "Timeout"}, status_code=504)
    except Exception as e:
        pending_results.pop(request_id, None)
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    import torch
    uvicorn.run(app, host="0.0.0.0", port=8000)
