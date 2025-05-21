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
import io
from faster_whisper import WhisperModel
from contextlib import asynccontextmanager
from diskcache import Cache
import hashlib
# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Приоритет моделей: чем ниже число, тем раньше в очереди
MODEL_PRIORITY = {
    "tiny": 1,
    "base": 2,
    "small": 3,
    "medium": 4,
    "large-v2": 5,
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup_event()
    yield
    shutdown_event()
    app.state.response_listener_task.cancel()

app = FastAPI(lifespan=lifespan)

# Очереди и общие структуры
request_queue = mp.Queue()
response_queue = mp.Queue()
pending_results = {}

def model_worker(request_queue, response_queue, stop_event):
    model_cache = {}

    while not stop_event.is_set():
        try:
            # Собираем все доступные запросы
            all_requests = []
            try:
                while True:
                    all_requests.append(request_queue.get_nowait())
            except:
                pass

            if not all_requests:
                time.sleep(0.1)
                continue

            # Сортируем по приоритету модели
            all_requests.sort(key=lambda r: MODEL_PRIORITY.get(r["model"], 999))

            for request in all_requests:
                model_name = request["model"]
                audio_bytes = request["audio_bytes"]
                request_id = request["request_id"]

                try:
                    if model_name not in model_cache:
                        logger.info(f"Загрузка модели {model_name}...")
                        model_cache[model_name] = WhisperModel(model_name, compute_type="float16")

                    model = model_cache[model_name]

                    # Сохраняем временный файл
                    with tempfile.NamedTemporaryFile(delete=False) as temp:
                        temp.write(audio_bytes)
                        temp_path = temp.name

                    try:
                        segments, _ = model.transcribe(temp_path, beam_size=1)
                        text = " ".join([seg.text for seg in segments])
                        if "cache_key" in request:
                            cache[request["cache_key"]] = text
                    finally:
                        os.remove(temp_path)

                    response_queue.put({"request_id": request_id, "text": text})

                except Exception as e:
                    logger.exception("Ошибка в worker:")
                    response_queue.put({"request_id": request_id, "error": str(e)})

        except Exception as e:
            logger.exception("Критическая ошибка в worker loop:")
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
                    future.set_result(result["text"])
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
    logger.info("Завершаем модель...")
    app.state.stop_event.set()
    app.state.model_process.join()
    logger.info("Модель остановлена.")
cache = Cache("whisper_cache")
def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), model: str = Query("base")):
    if model not in MODEL_PRIORITY:
        return JSONResponse({"error": "Модель не поддерживается"}, status_code=400)

    audio_bytes = await file.read()
    audio_hash = hash_bytes(audio_bytes)
    cache_key = f"{model}:{audio_hash}"

    # Проверка кэша
    if cache_key in cache:
        logger.info(f"[CACHE HIT] model={model}, hash={audio_hash}")
        return {"text": cache[cache_key]}

    request_id = id(audio_bytes)
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    pending_results[request_id] = future

    request_queue.put({
        "request_id": request_id,
        "audio_bytes": audio_bytes,
        "model": model,
        "cache_key": cache_key,  # добавим ключ в запрос
    })

    try:
        text = await asyncio.wait_for(future, timeout=600)
        return {"text": text}
    except asyncio.TimeoutError:
        pending_results.pop(request_id, None)
        return JSONResponse({"error": "Таймаут"}, status_code=504)
    except Exception as e:
        pending_results.pop(request_id, None)
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
