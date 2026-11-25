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
A plan is limited to 2 GiB, matching the existing diarization PCM limit. Prepared sources share an 8 GiB limit and require 1 GiB of free-disk headroom. Reaching the
concurrent storage limit returns an explicit storage-limit error; a new attempt
can run after another job releases its plan. Completed, cancelled and normally
stopped continuations close their source loans and remove per-request chunk
workspaces in `finally`. Worker startup removes private workspaces whose owning
PID no longer exists. Completed PCM/context and ASR-unit entries survive these
cleanups within their retention limits.

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

## Early web text and final identity

For a web GigaAM job with diarization, the worker can decode one preliminary
chunk before global Community-1 clustering. It searches at most the first
60 seconds for speech and decodes at most 20 seconds. The provisional text has
no speaker labels. A request without speech in that bounded prefix simply
continues to the final path.

Progress and segments carry `provisional: true` while the draft is visible,
including subsequent loading and diarization progress. At the start of final
ASR, `reset: true, provisional: false` starts a new presentation epoch in the
same job attempt. The final result contains only the original globally
attributed ASR units. Preliminary decoding does not change final unit signatures,
speaker-turn bounds, decoder options or the native API streaming contract.

The worker yields after the preliminary chunk and after global diarization,
with inference scopes closed. A completed cached global turn plan skips the
preliminary stage and restores the final path directly. Cancellation is checked
before/after audio preparation, before model admission, after CPU weights and
GPU transfer, after each decode and before publication. GigaAM model hooks are
bound only for the current acquire call.

`WHISPER_EARLY_TEXT_ENABLED=0` disables the bounded preview for controlled
comparison. Final results and metrics expose `preliminary_segments` and
`preliminary_inference_ms` when applicable. The preview adds one bounded ASR
operation; global speaker clustering remains an indivisible quality-preserving
pass and final decoding may revisit that short region.

## Reusable audio preparation

`prepared_audio.py` stores complete source PCM under
`WHISPER_PCM_DIR/cache-v1`. Its source identity includes SHA-256, PCM format and
decoder package versions. Independent context plans include VAD settings,
speaker turns, bounds/adapter version and preview limits. Changing one context
reuses the same PCM but recomputes its boundaries. The final path uses exactly
the prior GigaAM boundaries and sample conversion.

Publication writes and fsyncs a private stage before an atomic rename. Context
JSON is separately fsynced and replaced atomically. A shared source flock is
held until the consumer closes its memmap. Eviction takes a nonblocking exclusive
lock, so active plans cannot be removed. Process death releases these references.
A later exclusive borrower removes abandoned build stages before rebuilding;
quota accounting also includes orphan staging data.

Sources share an 8 GiB disk quota, a 1,000-entry limit and a 24-hour idle TTL
(`WHISPER_PCM_CACHE_TTL_SECONDS`). Each source PCM is limited to 2 GiB, and at
most 128 context plans are retained per source. The worker prunes expired idle
entries at startup and every minute. Per-request chunk WAV workspaces are
removed at completion/cancellation; completed PCM and context plans remain
available across worker recovery until eviction. `pcm_reused` and
`audio_plan_reused` in final results/metrics make this reuse observable.

Explicit source deletion can call `PreparedAudioCache(root).evict_source(hash)`.
This best-effort operation deletes unborrowed variants of that source and returns
the number removed. It neither waits for active consumers nor touches upload
originals. Existing global diarization retains its original FFmpeg/VAD
preparation and reuses its completed speaker-turn result separately.

## Runtime execution attribution

The worker replaces one shared execution snapshot before each expensive phase
and clears it whenever a scheduler quantum yields or completes. Phases distinguish
preparation, CPU model loading, broker admission, GPU transfer, ASR and global
diarization. Units identify the current operation without transcript content.

Only durable progress advances the recovery watermark: source PCM is 1,
completed global turns are 2, final audio context is 3, then each committed or
replayed final ASR unit advances it. Preliminary ASR does not advance this
watermark. Finalization is attributed even for an empty/silent audio plan.
