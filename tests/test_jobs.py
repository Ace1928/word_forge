"""Tests for idempotent, leased, crash-recoverable background jobs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from word_forge import Lexicon
from word_forge.database.database_manager import DBManager
from word_forge.jobs import IdempotencyConflictError, JobLeaseError, JobQueue


@pytest.fixture
def queue(tmp_path: Path):
    database = DBManager(tmp_path / "jobs.sqlite")
    try:
        yield JobQueue(database), database
    finally:
        database.close()


def test_lexicon_exposes_shared_durable_queue(tmp_path: Path) -> None:
    with Lexicon(tmp_path / "lexicon.sqlite") as lexicon:
        job = lexicon.jobs.enqueue(
            "graph.refresh",
            {"reason": "test"},
            idempotency_key="graph.refresh:test",
        )
        assert lexicon.jobs.get(job.id) == job


def test_enqueue_is_idempotent_and_rejects_key_reuse(queue) -> None:
    jobs, _ = queue
    first = jobs.enqueue("vector.embed", {"entry": "one"}, idempotency_key="entry-one")
    repeated = jobs.enqueue("vector.embed", {"entry": "one"}, idempotency_key="entry-one")
    assert repeated.id == first.id
    assert repeated.state == "pending"
    with pytest.raises(IdempotencyConflictError, match="different work"):
        jobs.enqueue("vector.embed", {"entry": "two"}, idempotency_key="entry-one")


def test_claim_heartbeat_complete_and_owner_checks(queue) -> None:
    jobs, _ = queue
    created = jobs.enqueue("graph.refresh", {}, idempotency_key="graph-1")
    claimed = jobs.claim("worker-a", lease_seconds=30)
    assert claimed is not None
    assert claimed.id == created.id
    assert claimed.state == "running"
    assert claimed.attempts == 1
    with pytest.raises(JobLeaseError, match="does not hold"):
        jobs.complete(created.id, "worker-b")
    heartbeat = jobs.heartbeat(created.id, "worker-a", lease_seconds=60)
    assert heartbeat.lease_expires_at is not None
    completed = jobs.complete(created.id, "worker-a")
    assert completed.state == "completed"
    assert completed.completed_at is not None
    assert jobs.claim("worker-b") is None


def test_failures_retry_until_attempt_budget_is_exhausted(queue) -> None:
    jobs, _ = queue
    created = jobs.enqueue("source.fetch", {}, idempotency_key="fetch-1", max_attempts=2)
    first = jobs.claim("worker")
    assert first is not None
    retriable = jobs.fail(created.id, "worker", "temporary")
    assert retriable.state == "pending"
    second = jobs.claim("worker")
    assert second is not None
    assert second.attempts == 2
    exhausted = jobs.fail(created.id, "worker", "permanent")
    assert exhausted.state == "failed"
    assert exhausted.last_error == "permanent"
    assert jobs.claim("worker") is None


def test_expired_lease_is_recovered_after_worker_crash(queue) -> None:
    jobs, database = queue
    created = jobs.enqueue("index.refresh", {}, idempotency_key="index-1")
    claimed = jobs.claim("crashed-worker")
    assert claimed is not None
    with database.transaction() as connection:
        connection.execute(
            "UPDATE jobs SET lease_expires_at = 0 WHERE id = ?",
            (created.id,),
        )
    recovered = jobs.claim("replacement-worker")
    assert recovered is not None
    assert recovered.id == created.id
    assert recovered.lease_owner == "replacement-worker"
    assert recovered.attempts == 2


def test_concurrent_claimers_cannot_lease_the_same_job(queue) -> None:
    jobs, _ = queue
    created = jobs.enqueue("exclusive.work", {}, idempotency_key="exclusive-1")
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(jobs.claim, ("worker-a", "worker-b")))
    claimed = [job for job in results if job is not None]
    assert len(claimed) == 1
    assert claimed[0].id == created.id
