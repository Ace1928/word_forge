"""Integration tests for the versioned multilingual lexical schema."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from word_forge.database.database_manager import DBManager, SchemaError
from word_forge.database.schema import CURRENT_SCHEMA_VERSION
from word_forge.parser.linguistics import Phoneme, Pronunciation, segment_graphemes

LEGACY_SCHEMA = """
CREATE TABLE words (
    id INTEGER PRIMARY KEY,
    term TEXT UNIQUE NOT NULL,
    definition TEXT,
    part_of_speech TEXT,
    usage_examples TEXT,
    last_refreshed REAL NOT NULL
);
CREATE TABLE relationships (
    id INTEGER PRIMARY KEY,
    word_id INTEGER NOT NULL,
    related_term TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    FOREIGN KEY(word_id) REFERENCES words(id),
    UNIQUE(word_id, related_term, relationship_type)
);
CREATE TABLE emotional_relationships (
    id INTEGER PRIMARY KEY,
    word_id INTEGER NOT NULL,
    related_term TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    valence REAL NOT NULL,
    arousal REAL NOT NULL,
    last_updated REAL NOT NULL,
    FOREIGN KEY(word_id) REFERENCES words(id),
    UNIQUE(word_id, related_term, relationship_type)
);
"""


def _create_legacy_database(path: Path) -> None:
    """Create a populated schema-v1 database without using current code."""

    with sqlite3.connect(path) as connection:
        connection.executescript(LEGACY_SCHEMA)
        connection.executemany(
            """
            INSERT INTO words (
                id, term, definition, part_of_speech, usage_examples,
                last_refreshed
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (1, "Straße", "a road", "noun", "Die Straße", 10.0),
                (2, "STRASSE", "a street", "noun", "Auf der Strasse", 20.0),
                (3, "Ziel", "a target", "noun", "Das Ziel", 30.0),
            ],
        )
        connection.execute("""
            INSERT INTO relationships (
                id, word_id, related_term, relationship_type
            ) VALUES (1, 2, 'Ziel', 'related')
            """)
        connection.execute("""
            INSERT INTO emotional_relationships (
                id, word_id, related_term, relationship_type,
                valence, arousal, last_updated
            ) VALUES (1, 1, 'trust', 'trust_associated', 0.8, 0.4, 40.0)
            """)


def test_fresh_schema_is_versioned_and_complete(tmp_path: Path) -> None:
    database = DBManager(db_path=tmp_path / "fresh.sqlite")

    assert database.schema_version == CURRENT_SCHEMA_VERSION
    assert database.last_migration_report is not None
    assert database.last_migration_report.migrated is False

    with database.get_connection() as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        relationship_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(relationships)")
        }

    assert {
        "words",
        "relationships",
        "emotional_relationships",
        "graphemes",
        "pronunciations",
        "phonemes",
        "graph_metadata",
    } <= tables
    assert {
        "related_normalized_term",
        "related_language",
        "source",
        "confidence",
    } <= relationship_columns


def test_word_and_relationship_identity_is_language_aware(tmp_path: Path) -> None:
    database = DBManager(db_path=tmp_path / "languages.sqlite")

    german_id = database.insert_or_update_word("Straße", language="de")
    normalized_match_id = database.insert_or_update_word("STRASSE", language="de")
    english_id = database.insert_or_update_word("STRASSE", language="en")
    english_chat_id = database.insert_or_update_word("chat", language="en")
    french_chat_id = database.insert_or_update_word("chat", language="fr")

    assert normalized_match_id == german_id
    assert english_id != german_id
    assert french_chat_id != english_chat_id

    assert database.insert_relationship(
        "chat",
        "chat",
        "translation",
        base_language="en",
        related_language="fr",
        source="test",
        confidence=0.95,
    )
    assert not database.insert_relationship(
        "chat",
        "CHAT",
        "translation",
        base_language="en",
        related_language="fr",
        source="test",
        confidence=0.95,
    )

    relationship = database.get_relationships(str(english_chat_id))[0]
    assert relationship["related_term"] == "chat"
    assert relationship["related_normalized_term"] == "chat"
    assert relationship["related_language"] == "fr"
    assert relationship["confidence"] == pytest.approx(0.95)


def test_graphemes_pronunciations_and_phonemes_round_trip(tmp_path: Path) -> None:
    database = DBManager(db_path=tmp_path / "forms.sqlite")
    word_id = database.insert_or_update_word("café", language="en")
    graphemes = segment_graphemes("café")
    pronunciation = Pronunciation(
        notation="ipa",
        phonemes=(
            Phoneme(0, "k", "k", None, False),
            Phoneme(1, "æ", "æ", 1, True),
            Phoneme(2, "f", "f", None, False),
            Phoneme(3, "eɪ", "eɪ", 0, True),
        ),
        language="en-AU",
        dialect="en-AU",
        source="test-lexicon",
        confidence=0.8,
    )

    assert database.replace_graphemes(word_id, graphemes) == len(graphemes)
    assert database.replace_pronunciations(word_id, (pronunciation,)) == 1

    entry = database.get_word_entry("CAFÉ", "en")
    assert [item["text"] for item in entry["graphemes"]] == list("café")
    assert entry["graphemes"][-1]["codepoints"] == ["U+00E9"]
    assert entry["pronunciations"][0]["text"] == "k æ f eɪ"
    assert entry["pronunciations"][0]["stress_pattern"] == [1, 0]
    assert entry["pronunciations"][0]["phonemes"][1]["stress"] == 1


def test_deleting_word_cascades_through_derived_forms(tmp_path: Path) -> None:
    database = DBManager(db_path=tmp_path / "cascade.sqlite")
    word_id = database.insert_or_update_word("alpha")
    database.insert_relationship("alpha", "beta", "related")
    database.replace_graphemes(word_id, segment_graphemes("alpha"))
    database.replace_pronunciations(
        word_id,
        (
            Pronunciation(
                notation="ipa",
                phonemes=(Phoneme(0, "a", "a", 1, True),),
                language="en",
                dialect=None,
                source="test",
            ),
        ),
    )

    with database.get_connection() as connection:
        connection.execute("DELETE FROM words WHERE id = ?", (word_id,))
        connection.commit()
        counts = {
            table: int(
                connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )
            for table in ("relationships", "graphemes", "pronunciations", "phonemes")
        }

    assert counts == {
        "relationships": 0,
        "graphemes": 0,
        "pronunciations": 0,
        "phonemes": 0,
    }


def test_legacy_migration_preserves_data_and_merges_unicode_collisions(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy.sqlite"
    _create_legacy_database(db_path)

    database = DBManager(db_path=db_path)

    assert database.last_migration_report is not None
    assert database.last_migration_report.migrated is True
    assert database.last_migration_report.previous_version == 0
    assert database.last_migration_report.merged_word_collisions == 1
    assert database.schema_version == CURRENT_SCHEMA_VERSION

    entry = database.get_word_entry("strasse")
    assert entry["id_int"] == 1
    assert entry["definition"] == "a road | a street"
    assert entry["usage_examples"] == ["Die Straße", "Auf der Strasse"]
    assert entry["relationships"][0]["related_term"] == "Ziel"
    assert entry["relationships"][0]["related_normalized_term"] == "ziel"
    assert entry["relationships"][0]["source"] == "legacy"

    with database.get_connection() as connection:
        emotional = connection.execute("""
            SELECT word_id, related_term, related_language, valence, arousal
            FROM emotional_relationships
            """).fetchone()
        legacy_tables = connection.execute("""
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name LIKE '%_legacy_v1'
            """).fetchall()
        violations = connection.execute("PRAGMA foreign_key_check").fetchall()

    assert tuple(emotional) == (1, "trust", "en", 0.8, 0.4)
    assert legacy_tables == []
    assert violations == []


def test_failed_legacy_migration_rolls_back_without_renaming_tables(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "invalid-legacy.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(LEGACY_SCHEMA)
        connection.execute("""
            INSERT INTO words (
                id, term, definition, part_of_speech, usage_examples,
                last_refreshed
            ) VALUES (1, '   ', '', '', '', 0.0)
            """)

    with pytest.raises(SchemaError, match="Legacy word 1 has no term"):
        DBManager(db_path=db_path)

    with sqlite3.connect(db_path) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])

    assert "words" in tables
    assert "words_legacy_v1" not in tables
    assert version == 0


def test_newer_schema_version_is_rejected_without_mutation(tmp_path: Path) -> None:
    db_path = tmp_path / "future.sqlite"
    future_version = CURRENT_SCHEMA_VERSION + 1
    with sqlite3.connect(db_path) as connection:
        connection.execute(f"PRAGMA user_version = {future_version}")

    with pytest.raises(SchemaError, match="newer than supported"):
        DBManager(db_path=db_path)

    with sqlite3.connect(db_path) as connection:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()

    assert version == future_version
    assert tables == []
