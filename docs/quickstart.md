# Quickstart

## 1. Установка

```bash
git clone https://github.com/megamen32/WhisperServer.git
cd WhisperServer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Для разработки дополнительно:

```bash
pip install -r requirements-dev.txt
```

## 2. Настройка `.env`

```bash
API_KEY=dev-local-key
MODEL=base
OPENAI_DEFAULT_MODEL=large-v3
TG_BOT_ENABLED=false
```

Если сервер будет доступен из сети, замените `API_KEY` на длинное случайное значение.

## 3. Запуск

```bash
python main.py
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
