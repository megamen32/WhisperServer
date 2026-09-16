# Russian-first deployment

Install `deploy/gigaam-default.env` as
`/home/roomhacker/services/whisperserver-defaults.env` and
`deploy/gigaam-default.conf` into the Whisper systemd drop-in directory
and restart Whisper to select `gigaam-v3` for Telegram, the web UI and native
requests without a model. The OpenAI-compatible endpoint also uses the server
default when the model field is omitted. `OPENAI_DEFAULT_MODEL=gigaam-v3`
also routes existing `whisper-1` clients to GigaAM on both API endpoints.
Explicit `large-v3` still selects Whisper. Alias cache keys include the
overridden backend, so old Whisper transcriptions cannot mask the switch.

The extra EnvironmentFile is intentional: systemd EnvironmentFile entries
override Environment assignments, including those from later drop-ins.

To roll back, set `MODEL=large-v3`, `OPENAI_DEFAULT_MODEL=large-v3`, and
`TG_BOT_MODEL=whisper-1` in this env file
and restart Whisper. Do not change the existing broker or credential settings.

The worker caches model entries, acquiring models on demand. Broker-managed
inference holds an active lease through the complete segment iterator, then
marks the model idle. Worker-local model/iterator references are dropped before
the idle transition so eviction can release GPU memory. Models can stay warm
together when the GPU broker admits them; no unconditional unload/reload occurs
on each model switch. Memory pressure remains owned by the GPU broker.

Regression tests: `tests/test_default_model.py`, `tests/test_worker_model_reuse.py`.
