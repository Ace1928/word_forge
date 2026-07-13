"""Regression tests for durable Word Forge database evolution."""

from __future__ import annotations

import sqlite3

import pytest

from word_forge import Lexicon
from word_forge.database import migrations
from word_forge.database.database_manager import (
    SQL_CREATE_LEXEMES_TABLE,
    SQL_CREATE_RELATIONSHIPS_TABLE,
    SQL_CREATE_TRANSLATIONS_TABLE,
    SQL_CREATE_WORDS_TABLE,
    DBManager,
)


def _legacy_connection(path) -> sqlite3.Connection:
    """Create the four core tables used by pre-migration databases."""

    connection = sqlite3.connect(path)
    for statement in (
        SQL_CREATE_WORDS_TABLE,
        SQL_CREATE_RELATIONSHIPS_TABLE,
        SQL_CREATE_LEXEMES_TABLE,
        SQL_CREATE_TRANSLATIONS_TABLE,
    ):
        connection.execute(statement)
    return connection


def test_fresh_database_reaches_latest_version_idempotently(tmp_path) -> None:
    database = DBManager(tmp_path / "fresh.sqlite")
    assert database.schema_version == migrations.LATEST_SCHEMA_VERSION
    database.create_tables()
    assert database.schema_version == migrations.LATEST_SCHEMA_VERSION
    assert database.table_exists("change_log")
    assert database.table_exists("invalidation_queue")
    assert database.table_exists("jobs")
    database.close()


def test_legacy_data_is_backfilled_without_changing_row_ids(tmp_path) -> None:
    connection = _legacy_connection(tmp_path / "legacy.sqlite")
    decomposed = "Cafe\u0301"
    connection.execute(
        """
        INSERT INTO lexemes(lemma, lang, last_refreshed)
        VALUES (?, 'fr', 123.0)
        """,
        (decomposed,),
    )
    connection.execute(
        """
        INSERT INTO words(term, definition, part_of_speech, usage_examples, last_refreshed)
        VALUES ('legacy', 'Existing English data.', 'adjective', '', 100.0)
        """
    )
    connection.commit()
    original_id = int(connection.execute("SELECT id FROM lexemes").fetchone()[0])
    assert migrations.apply_migrations(connection) == migrations.LATEST_SCHEMA_VERSION

    row = connection.execute(
        """
        SELECT id, uid, normalized_lemma, script, revision, created_at
        FROM lexemes
        """
    ).fetchone()
    assert row is not None
    assert int(row[0]) == original_id
    assert len(str(row[1])) == 36
    assert str(row[2]) == "caf\u00e9"
    assert str(row[3]) == "Latn"
    assert int(row[4]) == 1
    assert float(row[5]) == 123.0
    english = connection.execute(
        "SELECT lemma, lang, gloss, uid FROM lexemes WHERE lemma = 'legacy'"
    ).fetchone()
    assert english is not None
    assert english[:3] == ("legacy", "en", "Existing English data.")
    assert len(str(english[3])) == 36
    connection.close()


def test_failed_migration_rolls_back_schema_and_version(tmp_path, monkeypatch) -> None:
    connection = _legacy_connection(tmp_path / "interrupted.sqlite")

    def fail_after_ddl(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE should_roll_back(id INTEGER PRIMARY KEY)")
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(
        migrations,
        "_MIGRATIONS",
        (migrations.Migration(1, "simulated-failure", fail_after_ddl),),
    )
    with pytest.raises(migrations.MigrationError, match="previous schema was preserved"):
        migrations.apply_migrations(connection)

    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    table = connection.execute(
        "SELECT name FROM sqlite_master WHERE name = 'should_roll_back'"
    ).fetchone()
    assert version == 0
    assert table is None
    connection.close()


def test_newer_database_is_rejected_instead_of_downgraded(tmp_path) -> None:
    connection = _legacy_connection(tmp_path / "future.sqlite")
    connection.execute("PRAGMA user_version = 999")
    with pytest.raises(migrations.MigrationError, match="newer than this Word Forge"):
        migrations.apply_migrations(connection)
    connection.close()


def test_writes_keep_identity_and_audit_invalidation_atomic(tmp_path) -> None:
    database_path = tmp_path / "audited.sqlite"
    with Lexicon(database_path) as lexicon:
        first = lexicon.add("ontology", definition="A formal account.", source="editor")
        second = lexicon.add("ontology", definition="A theory of being.", source="editor")
    assert first.term == second.term

    connection = sqlite3.connect(database_path)
    identity = connection.execute(
        "SELECT uid, revision FROM lexemes WHERE lemma = 'ontology' AND lang = 'en'"
    ).fetchone()
    assert identity is not None
    assert len(str(identity[0])) == 36
    assert int(identity[1]) == 2
    revisions = connection.execute(
        """
        SELECT revision, operation FROM change_log
        WHERE entity_type = 'lexeme' AND entity_id = ?
        ORDER BY revision
        """,
        (identity[0],),
    ).fetchall()
    assert revisions == [(1, "create"), (2, "update")]
    invalidations = connection.execute(
        """
        SELECT kind, source_revision FROM invalidation_queue
        WHERE entity_id = ? ORDER BY source_revision, kind
        """,
        (identity[0],),
    ).fetchall()
    assert invalidations == [
        ("graph", 1),
        ("search", 1),
        ("vector", 1),
        ("graph", 2),
        ("search", 2),
        ("vector", 2),
    ]
    connection.close()


def test_live_views_acknowledge_only_their_derived_work(tmp_path) -> None:
    database_path = tmp_path / "derived.sqlite"
    with Lexicon(database_path) as lexicon:
        lexicon.add("graph", definition="Connected data")
        assert [entry.term for entry in lexicon.find("connected")] == ["graph"]
        lexicon.network()

    connection = sqlite3.connect(database_path)
    states = dict(
        connection.execute(
            """
            SELECT kind, state FROM invalidation_queue
            WHERE entity_id = (SELECT uid FROM lexemes WHERE lemma = 'graph')
            ORDER BY kind
            """
        ).fetchall()
    )
    assert states == {
        "graph": "completed",
        "search": "completed",
        "vector": "pending",
    }
    connection.close()
