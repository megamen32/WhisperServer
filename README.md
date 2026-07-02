# WhisperServer

Локальный сервер транскрибации на FastAPI и faster-whisper. Делает из аудио текст, совместим с OpenAI v1 Audio Transcriptions, умеет стриминг, очередь, кеш, web-интерфейс, Telegram-бота и клиентскую библиотеку с fallback на локальную расшифровку.

Проект вырос из простой идеи: нажал кнопку, сказал обычной речью, получил нормальный многоязычный текст с пунктуацией и вставил его туда, где стоял курсор. Без обязательного облака, без корпоративных фаерволов, без привязки к одному провайдеру.

## Что умеет

- **OpenAI-compatible API**: `POST /v1/audio/transcriptions`, `GET /v1/models`.
- **Обычный API**: `POST /transcribe` для прямого использования без OpenAI SDK.
- **Streaming**: Server-Sent Events для OpenAI endpoint и NDJSON для локального endpoint.
- **Автовыбор модели**: `whisper-1` разворачивается в локальную faster-whisper модель, по умолчанию `large-v3`.
- **Очередь и приоритеты**: более лёгкие модели получают более высокий приоритет, тяжёлые не блокируют весь сервер.
- **Lazy loading + TTL**: модели грузятся по требованию и могут выгружаться после простоя через CUDA broker.
- **Кеширование**: повторная отправка того же файла возвращает результат из `diskcache`.
- **Web UI**: простая страница для ручной загрузки аудио.
- **Telegram bot**: можно отправить голосовое, аудио или видео и получить текст.
- **Python client**: библиотека `whisperclient` сначала пробует удалённый сервер, при ошибке может упасть в локальный faster-whisper CLI.
- **CUDA broker integration**: сервер может договариваться с другими GPU-проектами о VRAM и приоритетах.

## Быстрый старт

```bash
git clone https://github.com/megamen32/WhisperServer.git
cd WhisperServer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python main.py
```

Сервер поднимается на `http://127.0.0.1:7653`.

```bash
curl -fsS http://127.0.0.1:7653/status
```

## Пример через OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(api_key="dev-local-key", base_url="http://127.0.0.1:7653/v1")

with open("audio.mp3", "rb") as f:
    result = client.audio.transcriptions.create(model="whisper-1", file=f, language="ru")

print(result.text)
```

Curl:

```bash
curl -sS http://127.0.0.1:7653/v1/audio/transcriptions \
  -H "Authorization: Bearer dev-local-key" \
  -F "file=@audio.mp3" \
  -F "model=whisper-1" \
  -F "language=ru"
```

Streaming:

```bash
curl -N http://127.0.0.1:7653/v1/audio/transcriptions \
  -H "Authorization: Bearer dev-local-key" \
  -F "file=@audio.mp3" \
  -F "model=whisper-1" \
  -F "stream=true"
```

## API

| Endpoint | Назначение |
|---|---|
| `POST /v1/audio/transcriptions` | OpenAI-compatible transcription endpoint |
| `GET /v1/models` | список OpenAI-compatible model ids |
| `POST /transcribe` | простой endpoint проекта |
| `POST /web/transcribe` | endpoint для web UI с CSRF/session защитой |
| `GET /models` | список локальных моделей |
| `GET /status` | очередь, кеш и загруженные модели |
| `GET /` | web UI |

Подробнее: [docs/api.md](docs/api.md).

## Модели

Поддерживаются `tiny`, `base`, `small`, `medium`, `distil-large-v3`, `large-v3`, `large-v2`, `large` и OpenAI alias `whisper-1`.

По умолчанию `whisper-1` выбирает `OPENAI_DEFAULT_MODEL=large-v3`, но если уже загружена подходящая модель не ниже `OPENAI_WHISPER_MIN_MODEL`, сервер переиспользует её.

## Конфигурация

```bash
API_KEY=dev-local-key
MODEL=base
OPENAI_DEFAULT_MODEL=large-v3
TG_BOT_ENABLED=false
```

Все переменные описаны в [docs/configuration.md](docs/configuration.md). Не коммитьте `.env`; используйте `.env.example` как шаблон.

## Клиентская библиотека

```python
import whisperclient
from whisperclient import transcribe_sync, transcribe_stream_sync

whisperclient.api_key = "dev-local-key"
whisperclient.model = "large-v3"
whisperclient.whisper_url = "http://127.0.0.1:7653/transcribe"

print(transcribe_sync("voice.ogg"))
for event in transcribe_stream_sync("voice.ogg"):
    print(event)
```

Подробнее: [docs/client.md](docs/client.md).

## Telegram bot

Если `TG_BOT_ENABLED=1` и задан `TG_BOT_TOKEN`, `main.py` запускает `telegram_bot.py` отдельным subprocess и перезапускает его при падении.

## Deployment

Пример systemd unit лежит в `deploy/whisperserver.service`.

```bash
sudo cp deploy/whisperserver.service /etc/systemd/system/whisperserver.service
sudo systemctl daemon-reload
sudo systemctl enable --now whisperserver
sudo systemctl status whisperserver --no-pager
```

## Документация

- [docs/quickstart.md](docs/quickstart.md) — запуск за несколько минут.
- [docs/api.md](docs/api.md) — endpoints, параметры и примеры.
- [docs/configuration.md](docs/configuration.md) — переменные окружения.
- [docs/client.md](docs/client.md) — Python client и fallback.
- [docs/architecture.md](docs/architecture.md) — очередь, worker, кеш, CUDA broker.
- [docs/development.md](docs/development.md) — тесты и разработка.
- [docs/security.md](docs/security.md) — ключи, web UI, чистка истории.
- [docs/article-context.md](docs/article-context.md) — короткий контекст для статьи.

## Разработка

```bash
pip install -r requirements-dev.txt
pytest -q
python -m py_compile main.py telegram_bot.py whisperclient/*.py tests/*.py
```

## License

MIT.
