import asyncio
import logging
import os
import signal
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import AsyncGenerator, Optional
from aiogram import Bot, Dispatcher, F
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.types import FSInputFile, Message

import whisperclient


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("telegram_bot")

ALLOWED_LANGS = {
    "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs", "ca",
    "cs", "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fo", "fr",
    "gl", "gu", "ha", "haw", "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it",
    "ja", "jw", "ka", "kk", "km", "kn", "ko", "la", "lb", "ln", "lo", "lt", "lv",
    "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl", "nn", "no",
    "oc", "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn",
    "so", "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl", "tr",
    "tt", "uk", "ur", "uz", "vi", "yi", "yo", "zh", "yue",
}
SUPPORTED_DOC_EXTENSIONS = {
    ".aac", ".aiff", ".avi", ".flac", ".m4a", ".m4v", ".mkv", ".mov", ".mp3",
    ".mp4", ".mpeg", ".mpg", ".oga", ".ogg", ".opus", ".wav", ".webm", ".wma",
}


OPENAI_WHISPER_ALIAS = "whisper-1"


def normalize_lang(lang: Optional[str]) -> Optional[str]:
    if not lang:
        return None
    short = lang.split("_")[0].split("-")[0].lower()
    return short if short in ALLOWED_LANGS else None


class BotSettings:
    def __init__(self) -> None:
        self.token = os.getenv("TG_BOT_TOKEN", "")
        self.api_key = os.getenv("TG_BOT_API_KEY", os.getenv("API_KEY", "bad-key"))
        self.http_proxy = os.getenv("TG_BOT_HTTP_PROXY", "http://192.168.2.1:3128")
        self.server_url = os.getenv(
            "TG_BOT_SERVER_URL",
            os.getenv("WHISPER_URL", "http://127.0.0.1:7653/transcribe"),
        )
        self.model = os.getenv("TG_BOT_MODEL", os.getenv("MODEL", OPENAI_WHISPER_ALIAS))
        self.default_lang = normalize_lang(os.getenv("TG_BOT_LANG"))
        self.concurrency = max(1, int(os.getenv("TG_BOT_CONCURRENCY", "4")))
        self.max_size_mb = max(1, int(os.getenv("TG_BOT_MAX_SIZE_MB", "40")))


settings = BotSettings()
if not settings.token:
    raise RuntimeError("TG_BOT_TOKEN is not configured")

whisperclient.api_key = settings.api_key
whisperclient.model = settings.model
whisperclient.server_url = settings.server_url

bot_session = AiohttpSession(proxy=settings.http_proxy)
bot = Bot(token=settings.token, session=bot_session)
dp = Dispatcher()
sem = asyncio.Semaphore(settings.concurrency)
chat_langs: dict[int, Optional[str]] = {}
chat_max_sizes: dict[int, int] = {}


def _chat_lang(chat_id: int) -> Optional[str]:
    return chat_langs.get(chat_id, settings.default_lang)


def _chat_max_size(chat_id: int) -> int:
    return chat_max_sizes.get(chat_id, settings.max_size_mb)


def _pick_media(message: Message):
    if message.audio:
        return message.audio
    if message.voice:
        return message.voice
    if message.video:
        return message.video
    if message.video_note:
        return message.video_note
    if message.document:
        mime_type = (message.document.mime_type or "").lower()
        file_name = message.document.file_name or ""
        suffix = Path(file_name).suffix.lower()
        if mime_type.startswith(("audio/", "video/")) or suffix in SUPPORTED_DOC_EXTENSIONS:
            return message.document
    return None


def _media_suffix(message: Message, media) -> str:
    if message.voice:
        return ".ogg"
    if message.audio:
        return Path(message.audio.file_name or "audio.mp3").suffix.lower() or ".mp3"
    if message.video:
        return Path(message.video.file_name or "video.mp4").suffix.lower() or ".mp4"
    if message.video_note:
        return ".mp4"
    if message.document:
        return Path(message.document.file_name or "upload.bin").suffix.lower() or ".bin"
    return ".bin"


async def _download_media(message: Message, media) -> Path:
    telegram_file = await bot.get_file(media.file_id)
    fd, tmp_name = tempfile.mkstemp(suffix=_media_suffix(message, media))
    os.close(fd)
    await bot.download(telegram_file, destination=tmp_name)
    return Path(tmp_name)


async def _safe_edit(message: Message, text: str) -> None:
    try:
        await message.edit_text(text)
    except TelegramBadRequest as exc:
        lowered = str(exc).lower()
        if "message is not modified" in lowered:
            return
        if "message to edit not found" in lowered:
            return
        raise


async def _send_text_file(message: Message, text: str) -> None:
    fd, tmp_name = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    path = Path(tmp_name)
    path.write_text(text, encoding="utf-8")
    try:
        await message.answer_document(FSInputFile(str(path), filename="transcript.txt"))
        with suppress(TelegramBadRequest):
            await message.delete()
    finally:
        with suppress(FileNotFoundError):
            path.unlink()


async def _publish_partial(message: Message, text: str) -> None:
    if len(text) <= 4096:
        await _safe_edit(message, text)
        return
    await _safe_edit(message, "⏳ Текст длинный, отправлю итог файлом после завершения.")


async def _publish_final(message: Message, text: str) -> None:
    if not text:
        await _safe_edit(message, "Не удалось получить текст.")
        return
    if len(text) <= 4096:
        await _safe_edit(message, text)
        return
    await _send_text_file(message, text)


async def transcribe_stream(file_path: Path, lang: Optional[str]) -> AsyncGenerator[str, None]:
    assembled: list[str] = []
    last_text = ""

    # whisper-1 is intentionally mapped to Parakeet v3 by the server.
    model = whisperclient.model

    async for chunk in whisperclient.transcribe_stream_with_fallback(
        str(file_path),
        language=lang,
        model=model,
    ):
        if "result" in chunk:
            text = (chunk["result"].get("text") or "").strip()
        else:
            segment = chunk.get("segment") or {}
            segment_text = (segment.get("text") or "").strip()
            if not segment_text:
                continue
            assembled.append(segment_text)
            text = " ".join(assembled).strip()

        if text and text != last_text:
            last_text = text
            yield text


@dp.message(CommandStart())
async def start(message: Message) -> None:
    await message.reply(
        "Пришли аудио, видео, голосовое или файл с медиа. Бот расшифрует его через этот Whisper-сервер."
    )


@dp.message(Command("help"))
async def help_command(message: Message) -> None:
    await message.reply(
        "/lang <code|auto> — язык для этого чата\n"
        "/max <mb> — лимит файла для этого чата\n"
        "Поддерживаются audio, voice, video, video_note и документы с аудио/видео."
    )


@dp.message(Command("lang"))
async def set_lang(message: Message) -> None:
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) == 1:
        current = _chat_lang(message.chat.id) or "auto"
        await message.reply(f"Текущий язык для этого чата: {current}")
        return

    requested = parts[1].strip().lower()
    if requested in {"auto", "off", "none"}:
        chat_langs.pop(message.chat.id, None)
        await message.reply("Включено автоопределение языка для этого чата.")
        return

    normalized = normalize_lang(requested)
    if not normalized:
        await message.reply("Неизвестный код языка. Пример: /lang ru")
        return

    chat_langs[message.chat.id] = normalized
    await message.reply(f"Язык для этого чата: {normalized}")


@dp.message(Command("max"))
async def set_max_size(message: Message) -> None:
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) == 1:
        await message.reply(f"Текущий лимит для этого чата: {_chat_max_size(message.chat.id)} МБ")
        return

    try:
        size_mb = int(parts[1])
    except ValueError:
        await message.reply("Использование: /max 40")
        return

    if not 1 <= size_mb <= 200:
        await message.reply("Лимит должен быть в диапазоне 1..200 МБ")
        return

    chat_max_sizes[message.chat.id] = size_mb
    await message.reply(f"Лимит для этого чата: {size_mb} МБ")


@dp.message(F.audio | F.voice | F.video | F.video_note | F.document)
async def handle_media(message: Message) -> None:
    media = _pick_media(message)
    if media is None:
        await message.reply("Нужен аудио- или видеофайл.")
        return

    async with sem:
        size_bytes = media.file_size or 0
        size_mb = size_bytes / 1_048_576
        max_size_mb = _chat_max_size(message.chat.id)
        if size_mb > max_size_mb:
            await message.reply(f"Файл слишком большой: {size_mb:.1f} МБ при лимите {max_size_mb} МБ")
            return

        temp_path = await _download_media(message, media)
        status_message = await message.reply("⏳ Расшифровка...")
        last_update = 0.0
        last_sent = ""
        full_text = ""

        try:
            async for chunk in transcribe_stream(temp_path, _chat_lang(message.chat.id)):
                full_text = chunk
                now = time.monotonic()
                if chunk != last_sent and (now - last_update >= 1 or chunk.endswith((".", "!", "?"))):
                    await _publish_partial(status_message, chunk)
                    last_sent = chunk
                    last_update = now

            if full_text != last_sent or len(full_text) > 4096:
                await _publish_final(status_message, full_text)
            elif not full_text:
                await _publish_final(status_message, "")
        except Exception as exc:
            log.exception("Telegram bot transcription failed")
            await _safe_edit(status_message, f"⚠️ Ошибка: {exc}")
        finally:
            with suppress(FileNotFoundError):
                temp_path.unlink()


async def run_bot() -> None:
    log.info(
        "Telegram bot started. Server URL: %s. Proxy: %s",
        settings.server_url,
        settings.http_proxy,
    )
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop_event.set)

    polling_task = asyncio.create_task(
        dp.start_polling(
            bot,
            allowed_updates=dp.resolve_used_update_types(),
            skip_updates=False,
        )
    )
    stop_task = asyncio.create_task(stop_event.wait())

    done, pending = await asyncio.wait(
        {polling_task, stop_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    for task in pending:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    if stop_task in done and not polling_task.done():
        polling_task.cancel()
        with suppress(asyncio.CancelledError):
            await polling_task
    elif polling_task in done:
        await polling_task

    await bot.session.close()
    log.info("Telegram bot stopped.")


if __name__ == "__main__":
    asyncio.run(run_bot())
