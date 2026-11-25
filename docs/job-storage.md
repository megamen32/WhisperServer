# Job registry storage

The job registry uses SQLite WAL so a consistent read snapshot does not block a streamed segment commit. Snapshot reads collect raw job, segment and summary rows in one short transaction, then decode JSON and assemble text after closing that transaction. Delta reads retain the existing attempt, epoch and sequence contract.

Each connection requests an automatic checkpoint after 256 WAL pages and a 16 MiB retained journal limit after checkpoint. An active long reader can temporarily prevent truncation, so application reads must stay short. The JSON assembly phase does not hold a database transaction.

Back up an active database with SQLite's online backup API. Copying only jobs.sqlite while a writer is running can omit data that is still in the WAL. A stopped service may also have WAL/SHM files until its final connection closes; use the backup API for deployment snapshots as well.

The concurrency regression holds a reader while a real segment write commits. A second case pauses JSON decoding after the raw snapshot, writes a new segment, then completes a truncating checkpoint and verifies that the original reader still returns its consistent earlier text.

The response listener persists progress outside the API event loop, awaiting each write before consuming the next worker response. Notifications return to the event loop through its thread-safe dispatch. A terminal response therefore cannot overtake its earlier segments.
