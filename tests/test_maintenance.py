"""Tests for integrity, online backup, and guarded restore operations."""

from __future__ import annotations

from pathlib import Path

import pytest

from word_forge import Lexicon
from word_forge.database.database_manager import TermNotFoundError
from word_forge.maintenance import backup_database, inspect_database, restore_database


def test_online_backup_is_consistent_while_lexicon_is_open(tmp_path: Path) -> None:
    database = tmp_path / "live.sqlite"
    backup = tmp_path / "backups" / "snapshot.sqlite"
    with Lexicon(database) as lexicon:
        lexicon.add("before", definition="Included in snapshot")
        assert backup_database(database, backup) == backup
        lexicon.add("after", definition="Added after snapshot")

    report = inspect_database(backup)
    assert report.ok is True
    assert report.schema_version >= 1
    with Lexicon(backup) as snapshot:
        assert snapshot.get("before").definition == "Included in snapshot"
        with pytest.raises(TermNotFoundError, match="after"):
            snapshot.get("after")


def test_restore_requires_force_and_preserves_snapshot_content(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite"
    backup = tmp_path / "snapshot.sqlite"
    destination = tmp_path / "destination.sqlite"
    with Lexicon(source) as lexicon:
        lexicon.add("restored", definition="From backup")
    backup_database(source, backup)
    with Lexicon(destination) as lexicon:
        lexicon.add("existing", definition="Must not be replaced accidentally")

    with pytest.raises(FileExistsError, match="force=True"):
        restore_database(backup, destination)
    assert restore_database(backup, destination, force=True) == destination
    with Lexicon(destination) as lexicon:
        assert lexicon.get("restored").definition == "From backup"


def test_copy_rejects_same_path_and_missing_source(tmp_path: Path) -> None:
    database = tmp_path / "database.sqlite"
    with Lexicon(database):
        pass
    with pytest.raises(ValueError, match="must differ"):
        backup_database(database, database)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        backup_database(tmp_path / "missing.sqlite", tmp_path / "backup.sqlite")


def test_backup_requires_force_to_replace_existing_snapshot(tmp_path: Path) -> None:
    database = tmp_path / "database.sqlite"
    snapshot = tmp_path / "snapshot.sqlite"
    with Lexicon(database) as lexicon:
        lexicon.add("durable")
    backup_database(database, snapshot)

    with pytest.raises(FileExistsError, match="force=True"):
        backup_database(database, snapshot)
    assert backup_database(database, snapshot, force=True) == snapshot
