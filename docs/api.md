# API

## Authentication

Если в окружении задан `API_KEY`, OpenAI-compatible endpoint требует один из вариантов:

```http
Authorization: Bearer dev-local-key
X-API-Key: dev-local-key
```

Если `API_KEY` пустой, проверка отключается. Для публичного или сетевого запуска так делать не стоит.

## `POST /v1/audio/transcriptions`

OpenAI-compatible endpoint.

`multipart/form-data` параметры:

| Поле | Обязательно | Описание |
|---|---:|---|
| `file` | да | audio/video файл: mp3, wav, m4a, ogg, flac, mp4, webm и другие форматы, которые понимает ffmpeg/faster-whisper |
| `model` | да | `whisper-1` или локальная модель: `base`, `small`, `medium`, `large-v3` и т.д. |
| `language` | нет | ISO language code, например `ru`, `en`, `de` |
| `temperature` | нет | принимается для совместимости с OpenAI API |
| `stream` | нет | `true` включает SSE streaming |

Обычный ответ:

```json
{"text": "Распознанный текст", "language": "ru", "language_probability": 0.98}
```

Streaming ответ (`stream=true`) отдаётся как `text/event-stream`:

```text
data: {"type":"transcript.text.delta","delta":"Привет"}

data: {"type":"transcript.text.done","text":"Привет"}
```

Curl:

```bash
curl -sS http://127.0.0.1:7653/v1/audio/transcriptions \
  -H "Authorization: Bearer dev-local-key" \
  -F "file=@audio.mp3" \
  -F "model=whisper-1"
```

OpenAI SDK:

```python
from openai import OpenAI
client = OpenAI(api_key="dev-local-key", base_url="http://127.0.0.1:7653/v1")
with open("audio.mp3", "rb") as f:
    result = client.audio.transcriptions.create(model="whisper-1", file=f)
print(result.text)
```

## `GET /v1/models`

Возвращает OpenAI-compatible список моделей. Первым идёт `whisper-1`, дальше локальные faster-whisper модели.

## `POST /transcribe`

Простой endpoint проекта.

| Поле | Описание |
|---|---|
| `file` | файл |
| `model` | модель, по умолчанию `MODEL` из окружения или `base` |
| `language` | язык |
| `stream` | `true` для NDJSON streaming |
| `words` | `true` для word-level timestamps |

```bash
curl -sS -F "file=@audio.mp3" "http://127.0.0.1:7653/transcribe?model=base&language=ru"
curl -N -F "file=@audio.mp3" "http://127.0.0.1:7653/transcribe?model=base&stream=true"
```

## Service endpoints

| Endpoint | Описание |
|---|---|
| `GET /models` | локальные модели |
| `GET /status` | размер очереди, cache stats, loaded models |
| `GET /` | web UI |
| `POST /web/transcribe` | upload из web UI; защищён cookie + одноразовым CSRF token |
