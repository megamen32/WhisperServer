# Durable Web UI job events

## Read and stream contract

`GET /web/jobs` returns only job metadata. It omits transcript and summary payloads,
so opening the history sidebar does not read every completed transcript.
`GET /web/jobs/{id}` remains the full, session-owned snapshot and is the
fallback after a lost stream. Both reads use read transactions.

`GET /web/jobs/{id}/events?cursor=attempt:epoch:sequence` sends SSE messages:

```json
{
  "type": "delta",
  "job": {"id": "opaque-id", "attempt": 1, "status": "processing", "progress": {}},
  "segments": [{"start": 0, "end": 5, "text": "example"}],
  "cursor": "1:0:1",
  "reset": false,
  "has_more": false
}
```

Each SSE event has `id: attempt:epoch:sequence`. A browser reconnect's
`Last-Event-ID` takes precedence over the initial query cursor. Replay is
bounded to 128 segments per packet. The stream drains all replay pages before
closing a terminal job. An absent or obsolete cursor produces `reset: true`;
an OOM restart within one attempt increases the epoch. Clients replace that
generation, then append each unseen sequence only once.

Progress-only events have no new segments. Small JSON heartbeat messages are
sent every 15 seconds when no result changes. In-process notifications wake
subscribers immediately; a five-second durable-state read also observes changes
from another process. Healthy browser streams do not repeatedly fetch full
transcript snapshots. The final delta triggers one full GET to reconcile final
speaker text, metrics and saved summaries. All routes keep session ownership;
unknown and other-session jobs both return 404.

Segments are individual SQLite rows. A segment and its cursor are committed in
the same transaction. The hot job row stores small progress state, not the
growing transcript prefix. Final text is stored once for the existing result
and summarization APIs. Existing databases are accepted without rebuilding
completed results or summaries. Old partial snapshots are imported into
segment rows only when the next write arrives. A fresh attempt never replays
the previous attempt's text.

## Upload responsiveness and recovery

Normal browser mutations retain their serialized CSRF lane. GET requests and
explicit cancellation bypass that lane. Cancellation obtains its own token
from the existing same-origin session endpoint; it does not overwrite the
upload lane's token. Tokens are still single-use.

File writes, flushes and fsync run outside the event loop. Each upload has a
cross-process file lock. Only the short capacity reservation transaction is
shared: the complete declared file size is reserved atomically before writing,
so two simultaneous uploads cannot exceed the spool limit. A stalled fsync for
one file does not block another file or the cancellation request.

The acknowledged offset is committed after file fsync, including the directory
fsync for a new file. GET never advertises an uncommitted suffix. Retrying an acknowledged byte range verifies its contents. After a crash,
a partial unacknowledged suffix is replaced from the last committed offset,
then flushed and committed again. A mismatching replay or changed declared size is rejected. Cancellation
becomes visible immediately; if disk work is already in progress, cleanup
waits for its file lock in a background thread. The write task stays alive
after HTTP disconnection and checks cancellation or deletion again before
finishing, so late writes cannot leave orphan source files.

## Retention and migration

There is no age-based deletion. Existing source audio, jobs and completed
summaries are retained until an owner explicitly deletes a terminal job.
Explicit cancellation keeps its existing cleanup behavior. Retrying or changing
diarization replaces the result of the current job attempt, as before; this API
does not expose an archive of previous attempts.

The upload ledger initially accounts for all existing source files and never
expires them. A full spool rejects additional capacity instead of deleting old
audio. Operators can inspect `UploadSpool.usage()` for reserved and committed
bytes. Back up the SQLite job registry and its source spool together before
activating an upgraded service.

A lost worker requeues the current attempt through readiness coordination.
The original upload stays available, and completed compute checkpoints can be
reused by the worker. A user cancellation still wins over this automatic retry.

## Focused verification

```bash
python -m pytest tests/test_webui_deltas.py tests/test_webui_delta_api.py \
  tests/test_webui_job_progress.py tests/test_webui_streaming_browser.py -q
```

These checks cover legacy database reads, bounded replay, cursor ownership,
reset generations, SQLite read/write separation, quota races, cancellation
during fsync, independent browser control requests, partial DOM reuse and final
reconciliation. They do not load ASR models. Use an uncached public Web UI job
for the separate deployed-model check.
