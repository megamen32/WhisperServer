# Architecture

## Компоненты

```text
client / OpenAI SDK / web UI / Telegram bot
        |
        v
FastAPI main.py
        |
        +--> diskcache: кеш результатов
        +--> multiprocessing queue: очередь запросов
        +--> model worker: faster-whisper model cache
        +--> CUDA broker: опциональная координация GPU/VRAM
```

## Request flow

1. FastAPI принимает файл и параметры.
2. Аудио хешируется; если результат есть в `whisper_cache`, сервер возвращает кеш.
3. Если кеша нет, запрос кладётся в multiprocessing queue.
4. Worker выбирает/загружает модель.
5. faster-whisper возвращает сегменты и финальный текст.
6. Результат сохраняется в кеш и отдаётся клиенту.

## Streaming

- OpenAI endpoint: SSE events `transcript.text.delta` и `transcript.text.done`.
- `/transcribe`: NDJSON, по одному JSON object на строку.

## Model selection

`whisper-1` — alias, который выбирает конкретную faster-whisper модель: уже загруженную подходящую, `OPENAI_DEFAULT_MODEL` или минимум `OPENAI_WHISPER_MIN_MODEL`.

## CUDA broker

Если установлен `cudabroker_client`, модель может быть обёрнута в `ManagedModel`. Это позволяет нескольким GPU-проектам договариваться о VRAM, TTL и приоритетах.

## Web UI security

HTML не содержит API key. При открытии web UI сервер выдаёт session cookie и одноразовый CSRF token. Upload endpoint принимает запрос только с тем же origin и валидным token, после успешного запроса token ротируется.
