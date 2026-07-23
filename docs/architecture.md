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
        +--> model worker: native model backend cache
        +--> CUDA broker: опциональная координация GPU/VRAM
```

## Request flow

1. FastAPI принимает файл и параметры.
2. Аудио хешируется; если результат есть в `whisper_cache`, сервер возвращает кеш.
3. Если кеша нет, запрос кладётся в multiprocessing queue.
4. Worker выбирает/загружает модель.
5. Backend возвращает сегменты и финальный текст.
6. Результат сохраняется в кеш и отдаётся клиенту.

## Streaming

- OpenAI endpoint: SSE events `transcript.text.delta` и `transcript.text.done`.
- `/transcribe`: NDJSON, по одному JSON object на строку.

## Model selection

`whisper-1` — умный alias: worker переиспользует сильнейшую уже загруженную
модель, а если cache пуст, загружает `OPENAI_DEFAULT_MODEL` (по умолчанию
`parakeet-v3`). Whisper-модели работают через faster-whisper, а `parakeet-v3`
— через NeMo.

Если в worker уже загружена более сильная совместимая Whisper-модель, запрос
более слабой модели обслуживается ей. Таблица подмен намеренно не включает
Parakeet: у него другой backend и набор гарантий по языкам/декодированию.
Ответ содержит `requested_model`, `served_model`, `model_substituted` и
`substitution_reason`.

Каждая job также записывается в `WHISPER_METRICS_JSONL`: очередь, холодная
загрузка, inference time, audio duration, RTF, peak VRAM и результат.

## CUDA broker

Если установлен `cudabroker_client`, модель может быть обёрнута в `ManagedModel`. Это позволяет нескольким GPU-проектам договариваться о VRAM, TTL и приоритетах.

## Web UI security

HTML не содержит API key. При открытии web UI сервер выдаёт session cookie и одноразовый CSRF token. Upload endpoint принимает запрос только с тем же origin и валидным token, после успешного запроса token ротируется.
