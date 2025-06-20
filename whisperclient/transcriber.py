"""Utility functions for speech-to-text using Faster Whisper."""

from __future__ import annotations

import asyncio
import logging
import os
import json
import tempfile
from typing import AsyncGenerator, Generator, Optional
import threading

import aiohttp
import requests
from faster_whisper import WhisperModel
import whisperclient

_MODEL: Optional[WhisperModel] = None


def _get_model() -> WhisperModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = WhisperModel("base", device="cpu")
    return _MODEL


def _transcribe(path: str) -> str:
    model = _get_model()
    segments, _ = model.transcribe(path, beam_size=5)
    return " ".join(s.text.strip() for s in segments)


def _transcribe_stream(path: str) -> Generator[dict, None, None]:
    """Local streaming transcription yielding results in server format."""
    model = _get_model()
    segments, info = model.transcribe(
        path,
        beam_size=5,
        word_timestamps=True,
        condition_on_previous_text=False,
    )

    all_segments = []
    all_words = []
    for seg in segments:
        seg_data = {"start": seg.start, "end": seg.end, "text": seg.text}
        all_segments.append(seg_data)
        if seg.words:
            words = [
                {"start": w.start, "end": w.end, "word": w.word} for w in seg.words
            ]
            all_words.extend(words)
        else:
            words = []
        yield {"segment": seg_data, "words": words}

    result = {
        "text": " ".join(s["text"] for s in all_segments),
        "segments": all_segments,
        "words": all_words,
        "language": info.language,
        "language_probability": info.language_probability,
    }
    yield {"result": result}


async def _transcribe_stream_async(path: str) -> AsyncGenerator[dict, None]:
    """Asynchronous wrapper around :func:`_transcribe_stream`."""
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def worker() -> None:
        try:
            for item in _transcribe_stream(path):
                loop.call_soon_threadsafe(queue.put_nowait, item)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while True:
        item = await queue.get()
        if item is None:
            break
        yield item
    thread.join()


def transcribe_sync(
    file_path: str,
    language: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    url = "https://whisper.bezrabotnyi.com/transcribe"
    key = api_key or whisperclient.api_key
    model_name = model or whisperclient.model
    params = {"model": model_name, "api_key": key}
    if language:
        params["language"] = language

    try:
        with open(file_path, "rb") as f:
            files = {
                "file": (os.path.basename(file_path), f, "application/octet-stream")
            }
            response = requests.post(url, files=files, params=params, timeout=600)
            if response.ok:
                return response.json()["text"]
            else:
                logging.warning(
                    f"[Whisper сервер ответил {response.status_code}] {response.text}"
                )
    except Exception:
        logging.warning("Ошибка при обращении к удалённому whisper", exc_info=True)

    logging.info("Пробуем локальную расшифровку...")
    try:
        return _transcribe(file_path)
    except Exception:
        logging.exception("Локальный whisper тоже не сработал")
        return "[TRANSCRIPTION ERROR]"


def transcribe_stream_sync(
    file_path: str,
    language: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Generator[dict, None, None]:
    """Stream transcription results from the remote server.

    Yields dictionaries received from the server in real time. Falls back to a
    one-shot local transcription if the request fails.
    """
    url = "https://whisper.bezrabotnyi.com/transcribe"
    key = api_key or whisperclient.api_key
    model_name = model or whisperclient.model
    params = {"model": model_name, "api_key": key, "stream": "true"}
    if language:
        params["language"] = language

    try:
        with open(file_path, "rb") as f:
            files = {
                "file": (os.path.basename(file_path), f, "application/octet-stream")
            }
            with requests.post(
                url, files=files, params=params, timeout=600, stream=True
            ) as resp:
                if resp.ok:
                    for line in resp.iter_lines():
                        if line:
                            yield json.loads(line.decode("utf-8"))
                    return
                else:
                    logging.warning(
                        f"[Whisper сервер ответил {resp.status_code}] {resp.text}"
                    )
    except Exception:
        logging.warning("Ошибка при обращении к удалённому whisper", exc_info=True)

    logging.info("Пробуем локальную расшифровку...")
    try:
        for item in _transcribe_stream(file_path):
            yield item
    except Exception:
        logging.exception("Локальный whisper тоже не сработал")
        yield {"result": {"text": "[TRANSCRIPTION ERROR]"}}


async def voice_to_text(message) -> str:  # For aiogram uses
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        file_path = tmp.name

    await message.download_media(file_path)

    try:
        text = await transcribe_with_fallback(file_path)
    finally:
        try:
            os.remove(file_path)
        except OSError:
            logging.warning("Не удалось удалить временный файл", exc_info=True)

    return text


async def transcribe_with_fallback(
    file_path: str,
    language: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    url = "https://whisper.bezrabotnyi.com/transcribe"
    key = api_key or whisperclient.api_key
    model_name = model or whisperclient.model
    data = {"model": model_name, "api_key": key}
    if language:
        data["language"] = language

    try:
        async with aiohttp.ClientSession() as session:
            with open(file_path, "rb") as f:
                form = aiohttp.FormData()
                form.add_field(
                    "file",
                    f,
                    filename=os.path.basename(file_path),
                    content_type="application/octet-stream",
                )

                async with session.post(
                    url, data=form, params=data, timeout=600
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result["text"]
                    else:
                        error_text = await resp.text()
                        logging.warning(
                            f"[Whisper сервер ответил {resp.status}] {error_text}"
                        )
    except Exception:
        logging.warning("Ошибка при обращении к удалённому whisper", exc_info=True)

    logging.info("Пробуем локальную расшифровку...")
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _transcribe, file_path)
    except Exception:
        logging.exception("Локальный whisper тоже не сработал")
        return "[TRANSCRIPTION ERROR]"


async def transcribe_stream_with_fallback(
    file_path: str,
    language: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> AsyncGenerator[dict, None]:
    """Asynchronously stream transcription results with local fallback."""
    url = "https://whisper.bezrabotnyi.com/transcribe"
    key = api_key or whisperclient.api_key
    model_name = model or whisperclient.model
    params = {"model": model_name, "api_key": key, "stream": "true"}
    if language:
        params["language"] = language

    try:
        async with aiohttp.ClientSession() as session:
            with open(file_path, "rb") as f:
                form = aiohttp.FormData()
                form.add_field(
                    "file",
                    f,
                    filename=os.path.basename(file_path),
                    content_type="application/octet-stream",
                )
                async with session.post(
                    url, data=form, params=params, timeout=600
                ) as resp:
                    if resp.status == 200:
                        async for line in resp.content:
                            line = line.decode("utf-8").strip()
                            if line:
                                yield json.loads(line)
                        return
                    else:
                        error_text = await resp.text()
                        logging.warning(
                            f"[Whisper сервер ответил {resp.status}] {error_text}"
                        )
    except Exception:
        logging.warning("Ошибка при обращении к удалённому whisper", exc_info=True)

    logging.info("Пробуем локальную расшифровку...")
    try:
        async for item in _transcribe_stream_async(file_path):
            yield item
    except Exception:
        logging.exception("Локальный whisper тоже не сработал")
        yield {"result": {"text": "[TRANSCRIPTION ERROR]"}}


if __name__ == "__main__":
    t = transcribe_sync("/tmp/test.ogg")
    print(t)
