"""Utility functions for speech-to-text using Faster Whisper."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from typing import Optional

import aiohttp
import requests
from faster_whisper import WhisperModel
from dotenv import load_dotenv

load_dotenv()

_API_KEY = os.getenv("API_KEY", "bad-key")
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


def transcribe_sync(file_path: str, model: str = "large-v3", language: Optional[str] = None) -> str:
    url = "https://whisper.bezrabotnyi.com/transcribe"
    params = {"model": model, "api_key": _API_KEY}
    if language:
        params["language"] = language

    try:
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "application/octet-stream")}
            response = requests.post(url, files=files, params=params, timeout=600)
            if response.ok:
                return response.json()["text"]
            else:
                logging.warning(f"[Whisper сервер ответил {response.status_code}] {response.text}")
    except Exception:
        logging.warning("Ошибка при обращении к удалённому whisper", exc_info=True)

    logging.info("Пробуем локальную расшифровку...")
    try:
        return _transcribe(file_path)
    except Exception:
        logging.exception("Локальный whisper тоже не сработал")
        return "[TRANSCRIPTION ERROR]"


async def voice_to_text(message) -> str:#For aiogram uses
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


async def transcribe_with_fallback(file_path: str, model: str = "large-v3", language: Optional[str] = None) -> str:
    url = "https://whisper.bezrabotnyi.com/transcribe"
    data = {"model": model, "api_key": _API_KEY}
    if language:
        data["language"] = language

    try:
        async with aiohttp.ClientSession() as session:
            with open(file_path, "rb") as f:
                form = aiohttp.FormData()
                form.add_field("file", f, filename=os.path.basename(file_path), content_type="application/octet-stream")

                async with session.post(url, data=form, params=data, timeout=600) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result["text"]
                    else:
                        error_text = await resp.text()
                        logging.warning(f"[Whisper сервер ответил {resp.status}] {error_text}")
    except Exception:
        logging.warning("Ошибка при обращении к удалённому whisper", exc_info=True)

    logging.info("Пробуем локальную расшифровку...")
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _transcribe, file_path)
    except Exception:
        logging.exception("Локальный whisper тоже не сработал")
        return "[TRANSCRIPTION ERROR]"


if __name__ == '__main__':
    t = transcribe_sync("/tmp/test.ogg")
    print(t)
