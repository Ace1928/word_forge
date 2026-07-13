"""Safe backup, restore, and integrity operations for Word Forge databases."""

from __future__ import annotations

import os
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    """A machine-readable health snapshot for one lexicon database."""

    database: str
    ok: bool
    integrity: str
    schema_version: int
    foreign_key_violations: int
    pending_invalidations: int
    pending_jobs: int
    checked_at: float

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def _connect_existing(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"Database does not exist: {path}")
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA query_only = ON")
    return connection


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def inspect_database(path: str | Path) -> IntegrityReport:
    """Run SQLite integrity and lifecycle checks without changing the database."""

    database_path = Path(path).expanduser().resolve()
    connection = _connect_existing(database_path)
    try:
        integrity_row = connection.execute("PRAGMA integrity_check").fetchone()
        integrity = str(integrity_row[0]) if integrity_row else "no result"
        foreign_key_violations = len(connection.execute("PRAGMA foreign_key_check").fetchall())
        schema_row = connection.execute("PRAGMA user_version").fetchone()
        schema_version = int(schema_row[0]) if schema_row else 0
        pending_invalidations = (
            int(
                connection.execute(
                    "SELECT COUNT(*) FROM invalidation_queue WHERE state = 'pending'"
                ).fetchone()[0]
            )
            if _table_exists(connection, "invalidation_queue")
            else 0
        )
        pending_jobs = (
            int(
                connection.execute(
                    "SELECT COUNT(*) FROM jobs WHERE state IN ('pending', 'running')"
                ).fetchone()[0]
            )
            if _table_exists(connection, "jobs")
            else 0
        )
    finally:
        connection.close()
    return IntegrityReport(
        database=str(database_path),
        ok=integrity.casefold() == "ok" and foreign_key_violations == 0,
        integrity=integrity,
        schema_version=schema_version,
        foreign_key_violations=foreign_key_violations,
        pending_invalidations=pending_invalidations,
        pending_jobs=pending_jobs,
        checked_at=time.time(),
    )


def _atomic_sqlite_copy(source: Path, destination: Path) -> Path:
    """Copy a live SQLite database to an integrity-checked temporary file."""

    if not source.is_file():
        raise FileNotFoundError(f"Database does not exist: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    source_connection: sqlite3.Connection | None = None
    destination_connection: sqlite3.Connection | None = None
    try:
        source_connection = sqlite3.connect(source)
        source_connection.execute("PRAGMA query_only = ON")
        destination_connection = sqlite3.connect(temporary)
        source_connection.backup(destination_connection)
        destination_connection.commit()
        check = destination_connection.execute("PRAGMA integrity_check").fetchone()
        if check is None or str(check[0]).casefold() != "ok":
            raise RuntimeError(
                f"Backup integrity check failed: {check[0] if check else 'no result'}"
            )
    finally:
        if destination_connection is not None:
            destination_connection.close()
        if source_connection is not None:
            source_connection.close()
    try:
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            # Windows does not support opening directories this way. The file
            # itself has already been flushed and atomically replaced.
            directory_fd = None
        try:
            if directory_fd is not None:
                os.fsync(directory_fd)
        finally:
            if directory_fd is not None:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def backup_database(
    source: str | Path,
    destination: str | Path,
    *,
    force: bool = False,
) -> Path:
    """Create a snapshot, requiring explicit permission to replace a file."""

    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if source_path == destination_path:
        raise ValueError("backup destination must differ from the source database")
    if destination_path.exists() and not force:
        raise FileExistsError(
            f"Backup destination already exists: {destination_path}; pass force=True to replace it"
        )
    return _atomic_sqlite_copy(source_path, destination_path)


def restore_database(
    backup: str | Path,
    destination: str | Path,
    *,
    force: bool = False,
) -> Path:
    """Atomically restore a validated snapshot, requiring opt-in to overwrite."""

    backup_path = Path(backup).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if backup_path == destination_path:
        raise ValueError("restore source must differ from the destination database")
    if destination_path.exists() and not force:
        raise FileExistsError(
            f"Destination already exists: {destination_path}; pass force=True to replace it"
        )
    report = inspect_database(backup_path)
    if not report.ok:
        raise RuntimeError(
            f"Refusing to restore an unhealthy backup: {report.integrity}; "
            f"foreign-key violations={report.foreign_key_violations}"
        )
    return _atomic_sqlite_copy(backup_path, destination_path)


__all__ = [
    "IntegrityReport",
    "backup_database",
    "inspect_database",
    "restore_database",
]
