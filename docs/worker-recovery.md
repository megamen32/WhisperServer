# Worker progress deadlines and job quarantine

## Two recovery boundaries

The HTTP process watches the model worker without importing its model backends.
The worker publishes the current execution boundary through one synchronous
Manager dictionary assignment, before entering native code. A queue feeder
cannot be used for this marker: a process can exit before its last queued
message is sent.

The shared dictionary has one key, `snapshot`. Its value contains the worker
`generation`, `request_id`, optional `job_key=[job_id, attempt]`, `phase`,
`unit`, real `progress` and a durable `checkpoint` watermark. The scheduler
publishes `phase=idle` after every yielded or finished quantum. It never sends
synthetic heartbeat progress.

The existing response queue still carries transcription results. After worker
exit, the supervisor closes the parent's unused response writer and drains
complete terminal replies before advancing the generation. Late old-generation
messages remain fenced. Only requests still pending after that cutoff receive
`WorkerStopped`.

## Progress deadlines

Each phase has its own maximum interval without real progress:

| Phase | Default seconds | Environment override |
| --- | ---: | --- |
| Audio preparation | 300 | `WHISPER_WATCHDOG_PREPARING_SECONDS` |
| CPU imports and model weights | 900 | `WHISPER_WATCHDOG_PREPARING_CPU_SECONDS` |
| GPU broker admission | 900 | `WHISPER_WATCHDOG_WAITING_GPU_SECONDS` |
| Transfer/loading on GPU | 300 | `WHISPER_WATCHDOG_LOADING_GPU_SECONDS` |
| ASR inference | 600 | `WHISPER_WATCHDOG_INFERENCE_SECONDS` |
| Speaker diarization | 3600 | `WHISPER_WATCHDOG_DIARIZATION_SECONDS` |

A changed request, phase, unit, real progress value or durable checkpoint starts
a new interval. Repeated unchanged snapshots and unrelated heartbeat fields
do not extend it. Idle workers have no execution deadline. Stale-generation
snapshots cannot clear or reset a current deadline. Limits must be positive
finite numbers.

The startup readiness timeout remains separate. A phase timeout closes
admission, terminates the stuck worker and uses the normal drained-generation
recovery path. Global rolling restart limits remain in force, so repeated
failures cannot create an unbounded restart storm.

`/health` and `/status` expose `worker.execution.phase`,
`stalled_seconds` and `timeout_seconds`. They do not expose another owner's
job identity. Readiness changes immediately when the supervisor decides the
worker is stuck.

## Isolating one repeatedly failing job

The active snapshot identifies which request was executing at exit or timeout.
Only that request receives fault metadata. Other requests interrupted by the
same worker loss retry normally without a per-job penalty.

`JobStore.interrupt_for_retry(..., failure=..., max_failures=...)` stores a
small recovery record in the additive `job_recovery` table. No existing
`jobs` columns change. Recovery metadata is returned as `job.recovery`:

- `failures`: consecutive attributed failures without new durable progress;
- `checkpoint`: the largest durable watermark observed at a failure;
- `phase`, `unit`, `reason`: the most recent failing boundary;
- `quarantined`: whether automatic attempts have stopped.

A watermark greater than the previous maximum starts a new streak at one.
Replaying an existing prefix or temporarily reporting a lower watermark cannot
reset the streak. Preliminary text and native progress callbacks do not count
as durable advancement. GigaAM uses stable milestones for durable PCM, speaker
turns and plan preparation, followed by committed final ASR units.

`WHISPER_JOB_CRASH_LIMIT` defaults to 3. At the limit, the current job becomes
terminal `error`; its source, saved checkpoints and current partial result
remain available. Queued neighbours can run after worker recovery. The UI error
explains that a manual retry is available.

An explicit retry or an explicit changed diarization setting clears quarantine.
The Web UI action sends an owner-scoped, CSRF-protected
`POST /web/jobs/{job_id}/retry`. It accepts a terminal job only while the complete
source audio is retained and the worker can accept work. A successful response
is the new queued attempt with the recovery counter and error cleared. Repeated
requests while that attempt is queued or processing return the existing attempt.
The upload `/start` route remains idempotent and does not release quarantine.
Completed audio checkpoints retain their content/configuration keys across the
manual retry, so unchanged work can be reused.

Automatic settings reconciliation and application startup recovery do not revive
a quarantined job. Cancellation wins over charging a failure and completes the
existing cancellation cleanup. Deleting the terminal job cascades its recovery
record. Owner reassignment preserves it.

## Verification record

The initial two regressions were reproduced on the unchanged d42c303 source
through the actual web job runner and worker supervisor: three identical worker
losses still left the job queued, and an alive native worker ignored its phase
deadline. Both failed before the change and passed afterward with the same
inputs (2 tests).

Focused tests additionally cover persistence across store reopening, advancing
and replayed checkpoints, innocent queued requests, explicit retry, cancellation,
automatic settings reconciliation, phase-specific budgets, stale generations,
per-acquire callbacks and releasing model references before a cancelled GPU
lease. The existing real partial-frame IPC test and bounded terminal-drain tests
remain part of the recovery regression suite.

Run the focused suite with:

```bash
.venv/bin/python -m pytest -q tests/test_job_recovery_policy.py tests/test_observed_models.py tests/test_runtime_lifecycle.py tests/test_worker_ipc_recovery.py tests/test_webui_jobs.py tests/test_webui_job_api.py tests/test_diarization.py
```
