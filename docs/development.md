# Development

## Environment

```bash
uv sync --locked --python 3.10 --extra diarization --extra dev --no-install-project
.venv/bin/python scripts/runtime.py seal
cp .env.example .env
```

## Run

```bash
.venv/bin/python scripts/runtime.py run
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
processing. The SSE stream carries bounded, durable segment deltas with a resume cursor.
The browser reconnects after a connection loss and uses the full
`GET /web/jobs/{id}` snapshot as a fallback. History requests carry metadata only.
See [the job events contract](webui-job-events.md) for replay, upload durability
and retention behavior. Upload byte progress is independent of transcription progress.
A completed job replaces the partial transcript with its final result.

Production on server 88 runs the canonical checkout
`/home/roomhacker/PycharmProjects/WhisperServer` on `main` using its
`.venv/bin/python`. `deploy/whisperserver.service` is the source for the
service command. Release directories and stacked `ExecStart` overrides must
not select a second application version. Keep runtime settings and stored jobs
outside Git. Before restarting, check for active jobs and preserve the job store.

The `Whisper checks` GitHub workflow runs CPU-only persistence/upload tests
and the streaming UI regression in real headless Chrome. Chrome is required;
a missing browser fails the workflow. Model loading, the API dispatcher,
cancellation, and actual GPU transcription are verified in the existing server
environment:

```bash
.venv/bin/python -m pytest -q -m "not integration"
```

After deployment, test an uncached audio job through the public Web UI and
confirm that partial text and progress arrive before the terminal result.


## Reproducible native runtime

Use `uv sync --locked --python 3.10 --extra diarization --extra dev --no-install-project`
to build a clean candidate environment. `UV_PROJECT_ENVIRONMENT` can point to an
empty directory on persistent storage. The active checkout uses only `.venv`.
Do not install `nvidia-cudnn-cu12` alongside the cuDNN wheel selected by PyTorch:
both distributions own the same `nvidia/cudnn/lib` files and installation order
can mix incompatible sublibraries even when `uv pip check` passes.

After the source is committed, run the candidate interpreter with
`scripts/runtime.py seal`. This verifies NVIDIA shared libraries against wheel
SHA256 records and records the source commit, lock digest and complete package
inventory in the environment. `scripts/runtime.py check` is the cheap activation
gate; a changed lock, package set, source revision, or native-file size invalidates
it. The launcher constructs CUDA paths only from this environment and exposes
`WHISPER_BUILD_REVISION`, `WHISPER_LOCK_SHA256`, and `WHISPER_RUNTIME_ID` to the app.

Before activation, test actual GPU inference with an uncached audio fixture,
including diarization and a Whisper backend. During activation, stop the service
before changing the canonical checkout or the `.venv` link, fast-forward `main`
to the tested commit, install `deploy/whisperserver.service`, and start the whole
service. Never update HTML/source underneath a running process. Keep one rollback
archive outside the checkout. Confirm the public consumer and `/health` after
activation, and compare `/status` revision/runtime identity to the candidate.

Runtime model references are pinned in the GigaAM and diarization adapters.
An intentional model or dependency update requires resealing and the GPU gate.

See [worker progress deadlines and job quarantine](worker-recovery.md) for phased
timeouts, fault attribution, retained inputs and explicit retry behavior.
