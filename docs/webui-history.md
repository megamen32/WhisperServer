# Durable history access and explicit source removal

The browser session and the history owner are separate identities. Session
cookies remain HttpOnly, same-origin, and expiring. New sessions receive a random
durable owner that is independent of the cookie value. Jobs, transcripts, and
summary caches are scoped to that owner. A new visitor cannot list or access
another owner's jobs, SSE events, source files, or summaries.

## Recovery key

The sidebar's **Сохранить доступ к истории** section creates a recovery key on
explicit request. The key contains 256 random bits. The server stores only its
SHA-256 digest in `jobs.sqlite`; the raw key is returned once in a no-store
response. The browser shows it in a masked field, supports copying to a password
manager, and clears the field on acknowledgement or page exit. It does not put
the key in IndexedDB, localStorage, sessionStorage, URLs, or logs.

Possession of the key grants access to the owner's complete history. Treat it as
a password. The UI can replace a key after explicit confirmation; replacement
invalidates the previous key without signing out already authorized devices.

The same sidebar accepts a saved key after cookies are cleared or on another
device. Recovery creates a new session cookie and CSRF token. The previous
session used for recovery is invalidated. Existing sessions belonging to the
recovered owner remain authorized.

If the current browser owns completed, failed, or cancelled jobs, recovery
requires explicit confirmation before transferring that history to the
recovered owner. The transfer keeps job IDs, text, summaries, attempts, source
files, and stream cursors. It happens in the same SQLite transaction as the
ownership change. The previous current owner is retired, its recovery key is
invalidated, and all its sessions are revoked. The confirmation explains that
other devices using that old identity must reconnect with the recovered
owner's key. Requests which captured the retired owner before the transfer
cannot create new jobs under it.

Recovery is refused while the current owner has uploading, queued, processing,
or cancellation-pending jobs. Finish or cancel them first. This avoids moving a
job while its worker or upload is using the previous owner.

## Existing installations and expiry

No job rows, result payloads, summaries, or source files are rebuilt or deleted.
A valid existing session is lazily bound to a durable owner with the original
owner value, preserving its history. New owner/session mapping tables are added
alongside the existing job and stream schema.

Session expiry remains sliding where the existing session endpoints refresh it.
Expiry or cookie deletion does not delete transcript data or source audio. A
previously saved recovery key restores access under a fresh session. If the
cookie had already been lost or expired before a key was created, the public
API cannot infer which old history belongs to the visitor. There is no
anonymous enumeration or owner-ID claim endpoint.

## API

All mutations use the existing same-origin check and single-use CSRF token
pool. Identity request bodies are capped at 4 KiB. Key responses and identity
metadata are `Cache-Control: no-store`.

- `GET /web/identity`: `has_recovery_key` and current `job_count`, without owner
  IDs or key hashes.
- `POST /web/identity/recovery-key`: `{}` creates a key once; `{"replace": true}`
  explicitly replaces it. Returns `recovery_key` once. An existing key without
  replacement returns 409 with `detail.code = "key_exists"`.
- `POST /web/identity/recover`: `{"recovery_key": "...", "merge_current": false}`.
  A required history transfer returns 409 with `detail.code = "merge_required"`
  and the current job count. Resubmit with `merge_current: true` only after user
  confirmation. Active current jobs return `detail.code = "active_jobs"`.
  Success returns `ok`, `merged_jobs`, and the new session's CSRF token, sets the
  new cookie, and supplies the same token in `X-Whisper-Next-CSRF`. Invalid keys
  return a generic error without echoing the submitted value.

## Delete only the source audio

The completed job's history menu offers **Удалить только аудио**. Its
confirmation states that text and summaries stay available and that changing
speaker labels or rerunning recognition requires another upload.

`DELETE /web/jobs/{id}/source` is owner- and CSRF-protected. It only accepts a
completed job. It marks the source unavailable, removes the durable upload
spool with its per-file lock, and keeps the job, text, segments, saved summaries,
and stream cursor. The response and later snapshots expose
`source_available: false`. Until unlink succeeds, `source_removal_pending: true`
keeps an explicit retry action in the menu. Repeated removal is idempotent, including
retrying a previous filesystem failure. Changing diarization or retrying the
job then fails without clearing existing text or summaries.

There is no automatic expiration or deletion of user audio. This action removes
the original upload for that job. It does not remove the job from history or
erase unrelated reusable transcription cache entries.
