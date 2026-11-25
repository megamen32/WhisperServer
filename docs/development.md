# Development

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
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

Black-box тесты требуют запущенный сервер, доступную модель и `espeak`/`espeak-ng` для TTS:

```bash
BLACKBOX_TESTS=1 \
BLACKBOX_BASE_URL=http://127.0.0.1:7653 \
BLACKBOX_API_KEY=dev-local-key \
pytest -m integration tests/test_blackbox_audio.py -q
```

Можно выбрать другую модель через `BLACKBOX_MODEL`. Без `BLACKBOX_TESTS=1` этот набор пропускается.

Реальный браузерный smoke-тест Web UI запускается отдельно: он открывает Chromium через
`playwright-cli`, выбирает `gigaam-v3`, загружает TTS-файл и проверяет результат транскрибации:

```bash
BROWSER_TESTS=1 \
BROWSER_BASE_URL=http://127.0.0.1:7653 \
pytest -m integration tests/test_webui_browser.py -q
```

Для визуального запуска добавьте `BROWSER_HEADED=1`. Тест требует `npx`, `espeak` или
`espeak-ng` и заранее запущенный WhisperServer.

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


## Streaming jobs and production deployment

The Web UI subscribes to `GET /web/jobs/{id}/events` while a job is queued or
processing. The SSE stream carries the same session-owned snapshot as
`GET /web/jobs/{id}`, including partial `result.segments` and `progress`.
The browser reconnects after a connection loss and uses ordinary GET requests
as a fallback. Upload byte progress is independent of transcription progress.
A completed job replaces the partial transcript with its final result.

Production on server 88 runs the canonical checkout
`/home/roomhacker/PycharmProjects/WhisperServer` on `main` using its
`.venv-new/bin/python`. `deploy/whisperserver.service` is the source for the
service command. Release directories and stacked `ExecStart` overrides must
not select a second application version. Keep runtime settings and stored jobs
outside Git. Before restarting, check for active jobs and preserve the job store.

The `Whisper checks` GitHub workflow runs CPU-only persistence/upload tests
and the streaming UI regression in real headless Chrome. Chrome is required;
a missing browser fails the workflow. Model loading, the API dispatcher,
cancellation, and actual GPU transcription are verified in the existing server
environment:

```bash
.venv-new/bin/python -m pytest -q -m "not integration"
```

After deployment, test an uncached audio job through the public Web UI and
confirm that partial text and progress arrive before the terminal result.
