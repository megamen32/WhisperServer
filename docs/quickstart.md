# Quickstart

## 1. Установка

```bash
git clone https://github.com/megamen32/WhisperServer.git
cd WhisperServer
uv sync --locked --python 3.10 --extra diarization --no-install-project
.venv/bin/python scripts/runtime.py seal
cp .env.example .env
```

Для разработки дополнительно:

```bash
uv sync --locked --python 3.10 --extra diarization --extra dev --no-install-project
.venv/bin/python scripts/runtime.py seal
```

## 2. Настройка `.env`

```bash
API_KEY=dev-local-key
MODEL=gigaam-v3
OPENAI_DEFAULT_MODEL=gigaam-v3
TG_BOT_ENABLED=false
```

Если сервер будет доступен из сети, замените `API_KEY` на длинное случайное значение.

## 3. Запуск

```bash
.venv/bin/python scripts/runtime.py run
```

Сервер слушает `0.0.0.0:7653`; локально он доступен как `http://127.0.0.1:7653`.

## 4. Проверка

```bash
curl -fsS http://127.0.0.1:7653/status
curl -fsS http://127.0.0.1:7653/v1/models
```

## 5. Первая транскрибация

```bash
curl -sS http://127.0.0.1:7653/v1/audio/transcriptions \
  -H "Authorization: Bearer dev-local-key" \
  -F "file=@audio.mp3" \
  -F "model=whisper-1" \
  -F "language=ru"
```

## 6. Web UI

Откройте `http://127.0.0.1:7653/`, выберите файл и модель. Web UI не раскрывает `API_KEY` в HTML и использует одноразовый CSRF token для upload-запросов.
