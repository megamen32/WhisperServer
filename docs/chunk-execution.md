# Resumable GigaAM execution

The public job remains one recording. Internally, the default GigaAM path runs
one completed audio unit at a time and returns the worker to a FIFO cooperative
scheduler. This is execution scheduling, independent of browser/SSE delivery.

## Boundaries and quality

The default VAD path uses the same speech regions and 20-second maximum chunks
as the previous GigaAM adapter. With diarization enabled, chunk boundaries stay
inside the existing globally computed speaker turns. Original audio offsets are
preserved. Chunk PCM uses the same clipping, conversion and minimum decoder
padding as the original adapter.

Global Community-1 diarization remains one complete clustering pass. A hook can
report its internal progress and observe cancellation, but the scheduler does not
claim to preempt that pass every 20 seconds. Other backends retain their original
decode contract. Raw GigaAM requests without VAD/diarization retain their direct
short-file behavior.

A GigaAM continuation holds CPU metadata and a disk-backed PCM plan. Every model
inference scope is closed before the continuation yields. New arrivals are
admitted before that continuation is appended to the end of the runnable queue;
older continuations retain FIFO order.

Web jobs are admitted to this scheduler concurrently. Their input uses the
existing durable upload-spool path and SHA-256 hash, rather than keeping an extra
recording-sized buffer in the process queue. The worker never removes that
source path: the job owner removes it after a terminal cancellation acknowledgement
or explicit deletion.

## Completed-unit identity

`audio_checkpoints.py` uses a separate SQLite database under the existing job
directory, configurable with `WHISPER_CHECKPOINT_DB`. Only complete units are
committed, before they are published to clients.

The signature includes source SHA-256, actual model revision, adapter version,
language, beam/retry policy, temperature, word/VAD settings, exact chunk bounds
and speaker-turn context. GigaAM is pinned to the immutable production revision
`7655ad717f8122257385bb4b2f373db3697e8680`. Complete-result cache keys also include
the pinned GigaAM revision and adapter version.

The job attempt controls publication and cancellation, not reuse identity. A
new attempt may reuse completed units only when its computation signature matches.
Every unit is emitted at most once during an attempt. A failed or partially
decoded unit is never committed. OOM retries affect only the unfinished GigaAM
unit; already committed earlier units remain valid. Legacy backend retry/reset
behavior is preserved.

When Community-1 exports an immutable model revision and adapter version, its
completed global speaker-turn result is cached separately by source/VAD/context.
No partially finished clustering result is reused.

## Recovery integration

Full application startup recovers queued/processing jobs from the persistent
registry. Worker-only loss is distinguished from ordinary model errors:
`JobStore.interrupt_for_retry` starts a new queued attempt, and
`_defer_or_launch_web_job` starts it only after the replacement worker's readiness
marker. These job-store and lifecycle pieces are integrated with the worker
change. A cancellation must not be converted into a new runnable attempt.

`resumed_chunks`, `computed_chunks`, `chunks_total` and `diarization_reused`
are emitted in final results and metrics. They distinguish reuse from complete
recomputation without exposing source hashes or cache keys.

## Storage limits and cleanup

Completed units have a seven-day TTL, a 100,000-unit limit and a 256 MiB logical
payload quota. A transactional usage counter makes quota accounting independent
of transcript length; evictions remove the oldest units. SQLite may retain free
pages for later reuse, so physical file size includes ordinary database/index
overhead.

PCM plans use `WHISPER_PCM_DIR`, defaulting to `pcm` inside the job directory.
A plan is limited to 2 GiB, matching the existing diarization PCM limit. Active
plans share an 8 GiB limit and require 1 GiB of free-disk headroom. Reaching the
concurrent storage limit returns an explicit storage-limit error; a new attempt
can run after another job releases its plan. Completed, cancelled and normally stopped continuations delete
their plans in `finally`. Worker startup removes private plan directories whose
owning PID no longer exists. Finished-unit cache entries survive those cleanups.

## Focused verification

The primary regression submits a short recording after the first chunk of a
long one. Its final result must arrive before the long recording's final result.
Restart tests stop a worker after a committed chunk, start a fresh worker against
the same database and assert that only unfinished units are decoded. Additional
checks cover parameter/revision invalidation, cancellation before replay, model
scope closure, persisted diarization reuse, bounded storage and concurrent Web
UI admission.


## Cold GigaAM preparation and broker leases

GigaAM constructs its configuration and loads CPU weights before requesting a
GPU lease. The broker-managed loader only moves that prepared model to CUDA.
The same preparation runs again after an eviction or release; a still-cached
CUDA model is reused directly. Failed admission or transfer discards the
unpublished model and releases any granted lease. The broker heartbeat timeout
and inference lease contract stay unchanged.
