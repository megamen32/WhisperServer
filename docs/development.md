# Development

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
```

## Run

```bash
python main.py
```

## Tests

```bash
pytest -q
pytest tests/test_openai_endpoint_simple.py -q
pytest tests/test_openai_v1_streaming.py -q
```

## Syntax check

```bash
python -m py_compile main.py telegram_bot.py whisperclient/*.py tests/*.py
```

## Project layout

```text
main.py                         FastAPI app, queue, model worker, endpoints
telegram_bot.py                 Telegram integration
whisperclient/                  reusable Python client and local CLI fallback
templates/webui.html            web UI
tests/                          endpoint and streaming tests
deploy/whisperserver.service    systemd unit example
docs/                           maintained documentation
```

## Style notes

- Не добавляйте реальные ключи в тесты, README и history.
- Для тестов используйте фиктивные значения вроде `WHISPER_TEST_API_KEY`.
- `.env`, cache, venv, dist и egg-info должны оставаться untracked.
- При изменении API обновляйте `docs/api.md` и README examples.
