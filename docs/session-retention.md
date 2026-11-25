# Browser identity retention

A new anonymous session triggers at most one expiry-cleanup pass per minute.
Each pass removes at most 500 rows from each of the session, CSRF, owner mapping
and empty-owner tables. Session expiry remains authoritative in
`sessions.sqlite`; the job registry does not duplicate it.

The cleanup uses one SQLite connection with the authentication database
attached. Both write reservations are acquired before reading expiry, so a
renewal cannot race the deletion decision. It runs after session creation and
owner binding have committed, never while another session write transaction is
open. Lock contention has a 250 ms budget; a busy pass is deferred.

An owner is removed only when it has no jobs, recovery key or remaining session
mappings. Owners with any history or an issued recovery key remain durable.
Revoked mappings and retired-owner tombstones remain until the corresponding
authentication sessions expire. Recovery invalidates the old session's CSRF
tokens immediately but retains its expiry record for that cleanup.

Both production job-creation paths require an existing active owner inside the
same write transaction that creates the job. A delayed request from an owner
removed during expiry cleanup receives HTTP 403 and cannot create orphaned
history. Direct legacy JobStore callers can still explicitly create their own
owner-scoped fixtures without an authenticated browser identity.

Verification: the HTTP session-expiry and deleted-owner scenarios failed on the
previous source and passed after the change. Tests retain owners with jobs and
keys, exercise key recovery after auth expiry, keep revocation until expiry,
bound each batch, limit cleanup frequency, and pause a real HTTP request body
across expiry cleanup in both idempotent and ordinary creation paths.
