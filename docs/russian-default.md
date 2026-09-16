# Russian-first deployment

Install `deploy/gigaam-default.conf` into the Whisper systemd drop-in directory
and restart Whisper to select `gigaam-v3` for Telegram, the web UI and native
requests without a model. The OpenAI-compatible endpoint also uses the server
default when the model field is omitted. Explicit `whisper-1` and `large-v3`
requests retain Whisper semantics; unrelated clients need not change.

To roll back, set `MODEL=large-v3` and `TG_BOT_MODEL=whisper-1` in this drop-in
and restart Whisper. Do not change the existing broker or credential settings.

The worker caches model entries, acquiring models on demand. Broker-managed
inference holds an active lease through the complete segment iterator, then
marks the model idle. Worker-local model/iterator references are dropped before
the idle transition so eviction can release GPU memory. Models can stay warm
together when the GPU broker admits them; no unconditional unload/reload occurs
on each model switch. Memory pressure remains owned by the GPU broker.

Regression tests: `tests/test_default_model.py`, `tests/test_worker_model_reuse.py`.
