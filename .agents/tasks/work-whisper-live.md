# Whisper live recovery

Description: Recover the live WhisperServer transcription path when the GPU driver/broker reports no available GPU.
Severity: P1
Started: 2026-07-29 00:15 MSK (UTC+3)
Executor: L (host agent)
Harness: Codex desktop

Workflow:

1. [x] Explorer — inspect roadmap, service state, journal, graph, and runtime dependencies.
2. [x] Diagnosis — confirm queue growth, zero GPU capacity, and indefinite broker acquire behavior.
3. [x] Repair — patch the CPU path and add regression coverage.
4. [x] Verification — run targeted tests, restart service, and perform authenticated end-to-end transcription.
5. [x] Handoff — record result and roadmap status (no ROADMAP.md exists).

Acceptance:

- The worker does not call the broker lease path when CUDA is unavailable.
- Targeted tests pass in `.venv`.
- Live `/status` shows a draining/empty queue and model usage after a real transcription.
- The authenticated OpenAI-compatible transcription endpoint returns HTTP 200.

Result:

- Patched `main.py` so `ManagedModel` is used only when CUDA is available; CPU execution loads directly with `int8`.
- Added `CPU_FALLBACK_MODEL` (default `small`) and automatic `whisper-1` fallback when CUDA is unavailable.
- Added regression coverage for broker bypass and alias fallback.
- `graphify update .`: 461 nodes, 674 edges, 30 communities.
- `.venv/bin/pytest -q tests/test_model_registry.py tests/test_openai_transcription.py tests/test_blackbox_audio.py`: 19 passed, 3 skipped.
- Live service restarted successfully as PID 3033498. Broker still reports GPU total/free 0 MB, but the service is healthy.
- Authenticated `POST /v1/audio/transcriptions` with a real 1-second WAV and `model=whisper-1`: HTTP 200 in 5 seconds, `served_model=small`, queue returned to 0.
- No `ROADMAP.md` exists; no roadmap status change was possible.

Final status: P0 CONFIRMED — transcription is working end-to-end through the live service, with CPU fallback active while the host GPU driver remains unavailable.
