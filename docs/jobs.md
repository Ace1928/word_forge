# Durable jobs

Word Forge includes a small SQLite-backed job queue for work that must survive
application restarts, such as rebuilding a derived index. It is local-first and
requires no message broker.

## The lifecycle in plain language

1. The application enqueues a JSON payload with an idempotency key.
2. A worker claims the oldest available job for a limited lease.
3. Long-running work sends heartbeats before the lease expires.
4. The lease owner completes the job or records a failed attempt.
5. A failed or abandoned job becomes available again until `max_attempts` is
   reached.

An idempotency key names one logical operation. Reusing the key with the same
job kind and payload is safe and returns the existing job. Reusing it for
different work raises `IdempotencyConflictError`.

## Example

```python
from word_forge import JobQueue, Lexicon

lexicon = Lexicon()
queue = lexicon.jobs

job = queue.enqueue(
    "vector.rebuild",
    {"lexeme_uid": "018f2a86-example"},
    idempotency_key="vector.rebuild:018f2a86-example:revision-7",
    max_attempts=3,
)

claimed = queue.claim("index-worker-1", lease_seconds=60)
if claimed is not None:
    try:
        # Rebuild the requested derived artifact here.
        queue.complete(claimed.id, "index-worker-1")
    except Exception as exc:
        queue.fail(claimed.id, "index-worker-1", error=str(exc))
```

The queue rejects non-JSON payloads, invalid job kinds, empty worker names, and
unbounded lease or retry values before changing the database.

## Lease rules

- A claim increments the attempt counter.
- Only the current owner can heartbeat, complete, or fail a running job.
- An expired lease is reclaimed automatically by a later claim.
- An expired final attempt is marked failed rather than retried forever.
- Pending work may be cancelled; completed work remains immutable history.

Keep the work itself idempotent whenever possible. A lease prevents concurrent
ownership, but a process can crash after an external side effect and before it
records completion.
