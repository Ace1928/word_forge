"""Durable, idempotent background jobs backed by the lexicon database.

The queue uses short SQLite transactions and expiring leases. A worker crash
therefore cannot leave a job permanently stuck in ``running`` state, and an
idempotency key prevents duplicate work after caller retries.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Literal

from word_forge.database.database_manager import DatabaseError, DBManager

JobState = Literal["pending", "running", "completed", "failed", "cancelled"]
_KIND_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,100}$")


class JobError(DatabaseError):
    """Base error for invalid durable-job state transitions."""


class JobLeaseError(JobError):
    """Raised when a worker tries to mutate a job it does not lease."""


class IdempotencyConflictError(JobError):
    """Raised when one idempotency key is reused for different work."""


@dataclass(frozen=True, slots=True)
class Job:
    """Immutable snapshot of one durable background job."""

    id: str
    kind: str
    payload: dict[str, Any]
    state: JobState
    idempotency_key: str
    attempts: int
    max_attempts: int
    lease_owner: str | None
    lease_expires_at: float | None
    last_error: str
    created_at: float
    updated_at: float
    completed_at: float | None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def _canonical_payload(payload: dict[str, Any]) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("job payload must contain only JSON-compatible values") from exc


def _job_from_row(row: Any) -> Job:
    payload = json.loads(str(row["payload_json"]))
    if not isinstance(payload, dict):
        raise JobError(f"Stored payload for job {row['id']} is not a JSON object")
    return Job(
        id=str(row["id"]),
        kind=str(row["kind"]),
        payload=payload,
        state=str(row["state"]),  # type: ignore[arg-type]
        idempotency_key=str(row["idempotency_key"]),
        attempts=int(row["attempts"]),
        max_attempts=int(row["max_attempts"]),
        lease_owner=str(row["lease_owner"]) if row["lease_owner"] is not None else None,
        lease_expires_at=(
            float(row["lease_expires_at"]) if row["lease_expires_at"] is not None else None
        ),
        last_error=str(row["last_error"] or ""),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        completed_at=float(row["completed_at"]) if row["completed_at"] is not None else None,
    )


class JobQueue:
    """Enqueue and lease reliable local background work."""

    def __init__(self, database: DBManager) -> None:
        self._database = database

    def enqueue(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        idempotency_key: str,
        max_attempts: int = 3,
    ) -> Job:
        """Create a job or return the identical job from a retried request."""

        normalized_kind = str(kind).strip()
        if not _KIND_PATTERN.fullmatch(normalized_kind):
            raise ValueError(
                "job kind must use 1-100 letters, digits, dots, colons, underscores, or dashes"
            )
        normalized_key = str(idempotency_key).strip()
        if not normalized_key or len(normalized_key) > 200:
            raise ValueError("idempotency_key must contain 1-200 characters")
        if not 1 <= max_attempts <= 100:
            raise ValueError("max_attempts must be between 1 and 100")
        payload_json = _canonical_payload(payload)
        identifier = str(uuid.uuid4())
        timestamp = time.time()
        with self._database.transaction() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO jobs(
                    id, kind, payload_json, state, idempotency_key,
                    attempts, max_attempts, last_error, created_at, updated_at
                ) VALUES (?, ?, ?, 'pending', ?, 0, ?, '', ?, ?)
                """,
                (
                    identifier,
                    normalized_kind,
                    payload_json,
                    normalized_key,
                    max_attempts,
                    timestamp,
                    timestamp,
                ),
            )
            row = connection.execute(
                "SELECT * FROM jobs WHERE idempotency_key = ?", (normalized_key,)
            ).fetchone()
            if row is None:
                raise JobError("Failed to fetch job after enqueue")
            if str(row["kind"]) != normalized_kind or str(row["payload_json"]) != payload_json:
                raise IdempotencyConflictError(
                    f"Idempotency key {normalized_key!r} already identifies different work"
                )
            return _job_from_row(row)

    def get(self, job_id: str) -> Job | None:
        """Return one job snapshot, or ``None`` if it does not exist."""

        rows = self._database.execute_query("SELECT * FROM jobs WHERE id = ?", (job_id,))
        return _job_from_row(rows[0]) if rows else None

    def claim(self, worker: str, *, lease_seconds: float = 60.0) -> Job | None:
        """Atomically lease the oldest claimable job to ``worker``."""

        owner = str(worker).strip()
        if not owner or len(owner) > 200:
            raise ValueError("worker must contain 1-200 characters")
        if not 1.0 <= lease_seconds <= 86_400.0:
            raise ValueError("lease_seconds must be between 1 and 86400")
        now = time.time()
        expires = now + lease_seconds
        with self._database.transaction() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET state = 'failed', lease_owner = NULL, lease_expires_at = NULL,
                    last_error = CASE WHEN last_error = '' THEN 'lease expired' ELSE last_error END,
                    updated_at = ?
                WHERE state = 'running' AND lease_expires_at <= ?
                  AND attempts >= max_attempts
                """,
                (now, now),
            )
            connection.execute(
                """
                UPDATE jobs
                SET state = 'pending', lease_owner = NULL, lease_expires_at = NULL,
                    updated_at = ?
                WHERE state = 'running' AND lease_expires_at <= ?
                  AND attempts < max_attempts
                """,
                (now, now),
            )
            row = connection.execute(
                """
                SELECT id FROM jobs
                WHERE state = 'pending' AND attempts < max_attempts
                ORDER BY created_at, id
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            identifier = str(row["id"])
            cursor = connection.execute(
                """
                UPDATE jobs
                SET state = 'running', lease_owner = ?, lease_expires_at = ?,
                    attempts = attempts + 1, updated_at = ?
                WHERE id = ? AND state = 'pending'
                """,
                (owner, expires, now, identifier),
            )
            if cursor.rowcount != 1:
                return None
            claimed = connection.execute(
                "SELECT * FROM jobs WHERE id = ?", (identifier,)
            ).fetchone()
            if claimed is None:
                raise JobError("Claimed job disappeared")
            return _job_from_row(claimed)

    def heartbeat(self, job_id: str, worker: str, *, lease_seconds: float = 60.0) -> Job:
        """Extend a live lease held by the same worker."""

        if not 1.0 <= lease_seconds <= 86_400.0:
            raise ValueError("lease_seconds must be between 1 and 86400")
        now = time.time()
        with self._database.transaction() as connection:
            self._require_lease(connection, job_id, worker, now=now)
            connection.execute(
                "UPDATE jobs SET lease_expires_at = ?, updated_at = ? WHERE id = ?",
                (now + lease_seconds, now, job_id),
            )
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise JobError("Job disappeared during heartbeat")
            return _job_from_row(row)

    def complete(self, job_id: str, worker: str) -> Job:
        """Mark a job completed if ``worker`` still owns its lease."""

        now = time.time()
        with self._database.transaction() as connection:
            self._require_lease(connection, job_id, worker, now=now)
            connection.execute(
                """
                UPDATE jobs SET state = 'completed', lease_owner = NULL,
                    lease_expires_at = NULL, updated_at = ?, completed_at = ?
                WHERE id = ?
                """,
                (now, now, job_id),
            )
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise JobError("Job disappeared during completion")
            return _job_from_row(row)

    def fail(self, job_id: str, worker: str, error: str) -> Job:
        """Release a failed attempt for retry, or exhaust it permanently."""

        message = str(error).strip()[:20_000]
        now = time.time()
        with self._database.transaction() as connection:
            row = self._require_lease(connection, job_id, worker, now=now)
            state = "failed" if int(row["attempts"]) >= int(row["max_attempts"]) else "pending"
            connection.execute(
                """
                UPDATE jobs SET state = ?, lease_owner = NULL, lease_expires_at = NULL,
                    last_error = ?, updated_at = ? WHERE id = ?
                """,
                (state, message, now, job_id),
            )
            updated = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if updated is None:
                raise JobError("Job disappeared during failure handling")
            return _job_from_row(updated)

    def cancel(self, job_id: str) -> Job | None:
        """Cancel pending or running work; completed jobs remain immutable."""

        now = time.time()
        with self._database.transaction() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                return None
            if str(row["state"]) in {"completed", "failed", "cancelled"}:
                return _job_from_row(row)
            connection.execute(
                """
                UPDATE jobs SET state = 'cancelled', lease_owner = NULL,
                    lease_expires_at = NULL, updated_at = ?, completed_at = ?
                WHERE id = ?
                """,
                (now, now, job_id),
            )
            updated = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if updated is None:
                raise JobError("Job disappeared during cancellation")
            return _job_from_row(updated)

    @staticmethod
    def _require_lease(
        connection: Any,
        job_id: str,
        worker: str,
        *,
        now: float,
    ) -> Any:
        row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise JobLeaseError(f"Job does not exist: {job_id}")
        if (
            str(row["state"]) != "running"
            or str(row["lease_owner"] or "") != str(worker)
            or row["lease_expires_at"] is None
            or float(row["lease_expires_at"]) <= now
        ):
            raise JobLeaseError(f"Worker {worker!r} does not hold a live lease for job {job_id}")
        return row


__all__ = [
    "IdempotencyConflictError",
    "Job",
    "JobError",
    "JobLeaseError",
    "JobQueue",
    "JobState",
]
