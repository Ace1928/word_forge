"""Integration tests for normalized multi-sense lexical persistence."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from word_forge.database.database_manager import DBManager
from word_forge.database.lexical_repository import (
    LexicalRepository,
    LexicalRepositoryError,
)
from word_forge.database.schema import (
    CURRENT_SCHEMA_VERSION,
    SQL_CREATE_EMOTIONAL_RELATIONSHIPS_TABLE,
    SQL_CREATE_GRAPH_METADATA_TABLE,
    SQL_CREATE_RELATIONSHIPS_TABLE,
    SQL_CREATE_WORDS_TABLE,
)
from word_forge.lexicon import (
    LexicalEntryRecord,
    LexicalExampleRecord,
    LexicalFormRecord,
    LexicalGlossRecord,
    LexicalPronunciationRecord,
    LexicalRelationRecord,
    LexicalSenseRecord,
    SourceSnapshot,
    canonical_json,
)


def _snapshot() -> SourceSnapshot:
    return SourceSnapshot(
        source_id="test-lexicon",
        source_version="2026.07",
        source_url="https://example.test/lexicon.jsonl",
        license_name="Test Data License",
        license_url="https://example.test/license",
        attribution="Example Lexicon contributors",
        importer_version="word-forge-test/1",
        retrieved_at=1_752_470_400.0,
        artifact_sha256="a" * 64,
        artifact_bytes=2048,
        metadata_json=canonical_json({"edition": "integration"}),
    )


def _noun_entry() -> LexicalEntryRecord:
    return LexicalEntryRecord(
        source_entry_id="bank:English:noun:1",
        lemma="bank",
        language="en",
        part_of_speech="noun",
        lexical_category="common-noun",
        etymology="From Middle English banke.",
        tags=("countable",),
        metadata_json=canonical_json({"homograph": 1}),
        forms=(
            LexicalFormRecord(
                form="banks",
                language="en",
                position=0,
                source_form_id="plural",
                features=("number=plural",),
            ),
        ),
        senses=(
            LexicalSenseRecord(
                source_sense_id="financial-institution",
                position=0,
                concept_id="test:finance",
                tags=("finance",),
                glosses=(
                    LexicalGlossRecord(
                        "An institution that safeguards and lends money.", "en"
                    ),
                    LexicalGlossRecord("institution financière", "fr", kind="gloss"),
                ),
                examples=(
                    LexicalExampleRecord(
                        "She deposited the cheque at the bank.",
                        "en",
                        source_example_id="example-1",
                        translation="Elle a déposé le chèque à la banque.",
                        translation_language="fr",
                        reference="Test corpus",
                    ),
                ),
            ),
            LexicalSenseRecord(
                source_sense_id="river-edge",
                position=1,
                glosses=(
                    LexicalGlossRecord("The land alongside a body of water.", "en"),
                ),
            ),
        ),
        pronunciations=(
            LexicalPronunciationRecord(
                transcription="bæŋk",
                notation="ipa",
                language="en",
                position=0,
                source_record_id="ipa-1",
                form_source_id="plural",
                dialect="en-AU",
                tags=("standard",),
                audio_url="https://example.test/audio/bank.ogg",
                confidence=0.9,
            ),
        ),
        relations=(
            LexicalRelationRecord(
                relationship_type="synonym",
                target_term="financial institution",
                target_language="en",
                position=0,
                source_sense_id="financial-institution",
            ),
            LexicalRelationRecord(
                relationship_type="translation",
                target_term="banque",
                target_language="fr",
                position=1,
                source_sense_id="financial-institution",
                target_source_entry_id="banque:French:noun:1",
            ),
        ),
    )


def _verb_entry() -> LexicalEntryRecord:
    return LexicalEntryRecord(
        source_entry_id="bank:English:verb:1",
        lemma="bank",
        language="en",
        part_of_speech="verb",
        senses=(
            LexicalSenseRecord(
                source_sense_id="tilt-aircraft",
                position=0,
                glosses=(
                    LexicalGlossRecord("To tilt an aircraft while turning.", "en"),
                ),
            ),
        ),
    )


def test_snapshot_registration_is_idempotent_and_auditable(tmp_path: Path) -> None:
    database = DBManager(tmp_path / "snapshot.sqlite")
    repository = LexicalRepository(database)
    snapshot = _snapshot()

    first_id = repository.register_snapshot(snapshot)
    second_id = repository.register_snapshot(
        replace(snapshot, attribution="Updated attribution")
    )

    assert second_id == first_id
    row = database.execute_query(
        "SELECT * FROM source_snapshots WHERE id = ?", (first_id,)
    )[0]
    assert row["artifact_sha256"] == "a" * 64
    assert row["artifact_bytes"] == 2048
    assert row["attribution"] == "Updated attribution"


def test_multiple_entries_senses_and_forms_round_trip_idempotently(
    tmp_path: Path,
) -> None:
    database = DBManager(tmp_path / "lexicon.sqlite")
    repository = LexicalRepository(database)
    snapshot_id = repository.register_snapshot(_snapshot())
    noun = _noun_entry()
    verb = _verb_entry()

    first = repository.upsert_entries(snapshot_id, (noun, verb))

    assert first.attempted == 2
    assert first.inserted == 2
    assert first.updated == 0
    assert first.senses == 3
    assert first.glosses == 4
    assert first.examples == 1
    assert first.pronunciations == 1
    assert first.relations == 2
    assert database.execute_scalar("SELECT COUNT(*) FROM words") == 1
    assert database.execute_scalar("SELECT COUNT(*) FROM lexical_entries") == 2
    assert len(repository.find_entry_ids("BANK", "EN")) == 2
    assert repository.get_entry_by_source(snapshot_id, noun.source_entry_id) == noun
    assert repository.get_entry_by_source(snapshot_id, verb.source_entry_id) == verb

    second = repository.upsert_entries(snapshot_id, (noun, verb))

    assert second.inserted == 0
    assert second.updated == 2
    assert database.execute_scalar("SELECT COUNT(*) FROM lexical_entries") == 2
    assert database.execute_scalar("SELECT COUNT(*) FROM lexical_senses") == 3
    assert database.execute_scalar("SELECT COUNT(*) FROM lexical_glosses") == 4
    assert database.execute_scalar("SELECT COUNT(*) FROM lexical_examples") == 1
    assert database.execute_scalar("SELECT COUNT(*) FROM lexical_relations") == 2
    assert database.execute_scalar("SELECT COUNT(*) FROM relationships") == 2

    facade = database.get_word_entry("bank", "en")
    assert facade["definition"] == "To tilt an aircraft while turning."
    assert facade["graphemes"]
    assert {item["related_language"] for item in facade["relationships"]} == {
        "en",
        "fr",
    }


def test_reimport_replaces_only_the_selected_entry_children(tmp_path: Path) -> None:
    database = DBManager(tmp_path / "replace.sqlite")
    repository = LexicalRepository(database)
    snapshot_id = repository.register_snapshot(_snapshot())
    noun = _noun_entry()
    verb = _verb_entry()
    repository.upsert_entries(snapshot_id, (noun, verb))

    updated_noun = replace(
        noun,
        senses=(noun.senses[0],),
        relations=(noun.relations[0],),
    )
    report = repository.upsert_entries(snapshot_id, (updated_noun,))

    assert report.updated == 1
    assert (
        repository.get_entry_by_source(snapshot_id, noun.source_entry_id)
        == updated_noun
    )
    assert repository.get_entry_by_source(snapshot_id, verb.source_entry_id) == verb
    assert database.execute_scalar("SELECT COUNT(*) FROM lexical_senses") == 2
    assert database.execute_scalar("SELECT COUNT(*) FROM lexical_relations") == 1


def test_missing_snapshot_rolls_back_entire_batch(tmp_path: Path) -> None:
    database = DBManager(tmp_path / "rollback.sqlite")
    repository = LexicalRepository(database)

    with pytest.raises(LexicalRepositoryError, match="does not exist"):
        repository.upsert_entries(999, (_noun_entry(), _verb_entry()))

    assert database.execute_scalar("SELECT COUNT(*) FROM words") == 0
    assert database.execute_scalar("SELECT COUNT(*) FROM lexical_entries") == 0


def test_duplicate_batch_identity_is_rejected_before_writes(tmp_path: Path) -> None:
    database = DBManager(tmp_path / "duplicate.sqlite")
    repository = LexicalRepository(database)
    snapshot_id = repository.register_snapshot(_snapshot())
    entry = _noun_entry()

    with pytest.raises(ValueError, match="unique within a batch"):
        repository.upsert_entries(snapshot_id, (entry, entry))

    assert database.execute_scalar("SELECT COUNT(*) FROM words") == 0


def test_schema_v2_upgrades_additively_without_touching_words(tmp_path: Path) -> None:
    path = tmp_path / "v2.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(SQL_CREATE_WORDS_TABLE)
        connection.execute(SQL_CREATE_RELATIONSHIPS_TABLE)
        connection.execute(SQL_CREATE_EMOTIONAL_RELATIONSHIPS_TABLE)
        connection.execute(SQL_CREATE_GRAPH_METADATA_TABLE)
        connection.execute("""
            INSERT INTO words (
                term, normalized_term, language, script, definition,
                part_of_speech, usage_examples, source, is_stub,
                last_refreshed
            ) VALUES ('café', 'café', 'fr', 'Latn', 'boisson', 'noun',
                      '', 'v2-test', 0, 1.0)
            """)
        connection.execute("PRAGMA user_version = 2")

    database = DBManager(path)

    assert database.schema_version == CURRENT_SCHEMA_VERSION == 3
    assert database.last_migration_report is not None
    assert database.last_migration_report.migrated is True
    assert database.get_word_entry("CAFÉ", "fr")["definition"] == "boisson"
    assert database.table_exists("source_snapshots")
    assert database.table_exists("lexical_entries")


def test_record_validation_matches_database_uniqueness() -> None:
    duplicate_form = LexicalFormRecord(
        form="Straße", language="de", position=0, features=("plural",)
    )
    normalized_duplicate = LexicalFormRecord(
        form="STRASSE", language="de", position=1, features=("plural",)
    )

    with pytest.raises(ValueError, match="forms must be unique"):
        LexicalEntryRecord(
            source_entry_id="duplicate",
            lemma="Straße",
            language="de",
            forms=(duplicate_form, normalized_duplicate),
        )

    with pytest.raises(ValueError, match="64 hexadecimal"):
        replace(_snapshot(), artifact_sha256="not-a-digest")
