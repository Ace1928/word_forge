"""Integration coverage for governed streaming Kaikki imports."""

from __future__ import annotations

import bz2
import hashlib
import json
from pathlib import Path

import pytest

from word_forge.database.database_manager import DBManager
from word_forge.database.lexical_repository import LexicalRepository
from word_forge.sources.kaikki import (
    KaikkiImporter,
    KaikkiImportError,
    KaikkiLicenseError,
    inspect_artifact,
    parse_kaikki_record,
)


def _records() -> list[dict[str, object]]:
    return [
        {
            "id": "en-bank-noun-1",
            "word": "bank",
            "lang": "English",
            "lang_code": "en",
            "pos": "noun",
            "pos_title": "Noun",
            "etymology_text": "From Middle English banke.",
            "tags": ["countable"],
            "forms": [{"form": "banks", "tags": ["plural"]}],
            "sounds": [
                {
                    "ipa": "/bæŋk/",
                    "tags": ["UK"],
                    "ogg_url": "https://example.test/bank.ogg",
                }
            ],
            "senses": [
                {
                    "id": "en-bank-finance",
                    "glosses": ["An institution that safeguards and lends money."],
                    "raw_glosses": ["A financial institution."],
                    "tags": ["finance"],
                    "examples": [
                        {
                            "text": "She deposited a cheque at the bank.",
                            "ref": "Example corpus",
                        }
                    ],
                    "synonyms": [{"word": "financial institution"}],
                    "translations": [{"word": "banque", "code": "fr"}],
                },
                {
                    "id": "en-bank-river",
                    "glosses": ["The land alongside a body of water."],
                },
            ],
            "synonyms": ["depository"],
            "translations": [{"word": "Bank", "code": "de"}],
        },
        {
            "word": "bank",
            "lang": "English",
            "lang_code": "en",
            "pos": "verb",
            "senses": [
                {
                    "glosses": ["To tilt an aircraft while turning."],
                    "synonyms": ["incline"],
                }
            ],
        },
        {
            "id": "fr-chat-noun-1",
            "word": "chat",
            "lang": "French",
            "lang_code": "fr",
            "pos": "noun",
            "senses": [{"glosses": ["Mammifère félin domestique."]}],
        },
    ]


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _importer(
    database: DBManager, *, languages: tuple[str, ...] = (), batch_size: int = 2
) -> KaikkiImporter:
    return KaikkiImporter(
        LexicalRepository(database),
        source_version="2026-07-06",
        source_url="https://kaikki.org/dictionary/rawdata.html",
        accept_source_license=True,
        batch_size=batch_size,
        languages=languages,
    )


def test_record_parser_preserves_nested_lexical_structure() -> None:
    entry = parse_kaikki_record(_records()[0], line_number=1)

    assert entry.source_entry_id == "en-bank-noun-1"
    assert entry.language == "en"
    assert entry.part_of_speech == "noun"
    assert entry.forms[0].form == "banks"
    assert entry.forms[0].features == ("plural",)
    assert len(entry.senses) == 2
    assert entry.senses[0].glosses[1].kind == "raw"
    assert entry.senses[0].examples[0].reference == "Example corpus"
    assert entry.pronunciations[0].transcription == "/bæŋk/"
    assert entry.pronunciations[0].audio_url.endswith("bank.ogg")
    assert {
        (relation.relationship_type, relation.target_term, relation.target_language)
        for relation in entry.relations
    } >= {
        ("synonym", "financial institution", "en"),
        ("translation", "banque", "fr"),
        ("translation", "Bank", "de"),
    }


def test_share_alike_data_requires_explicit_acknowledgement(tmp_path: Path) -> None:
    database = DBManager(tmp_path / "license.sqlite")

    with pytest.raises(KaikkiLicenseError, match="attribution/share-alike"):
        KaikkiImporter(
            LexicalRepository(database),
            source_version="2026-07-06",
            source_url="https://kaikki.org/dictionary/rawdata.html",
            accept_source_license=False,
        )


def test_artifact_inspection_hashes_exact_source_bytes(tmp_path: Path) -> None:
    path = tmp_path / "sample.jsonl"
    _write_jsonl(path, _records())

    identity = inspect_artifact(path, chunk_bytes=17)

    assert identity.byte_size == path.stat().st_size
    assert identity.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert identity.path == path.resolve()


def test_artifact_inspection_verifies_an_expected_digest(tmp_path: Path) -> None:
    path = tmp_path / "verified.jsonl"
    _write_jsonl(path, _records()[:1])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    assert inspect_artifact(path, expected_sha256=digest.upper()).sha256 == digest
    with pytest.raises(KaikkiImportError, match="SHA-256 mismatch"):
        inspect_artifact(path, expected_sha256="0" * 64)


def test_streaming_import_filters_batches_checkpoints_and_resumes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample.jsonl"
    checkpoint = tmp_path / "state" / "kaikki.json"
    _write_jsonl(path, _records())
    database = DBManager(tmp_path / "kaikki.sqlite")
    importer = _importer(database, languages=("en",), batch_size=1)
    artifact = inspect_artifact(path)

    report = importer.import_artifact(artifact, checkpoint_path=checkpoint)

    assert report.lines_read == 3
    assert report.parsed_entries == 2
    assert report.skipped_entries == 1
    assert report.batches == 2
    assert report.write_report.inserted == 2
    assert report.write_report.senses == 3
    assert database.execute_scalar("SELECT COUNT(*) FROM lexical_entries") == 2
    assert database.execute_scalar("SELECT COUNT(*) FROM source_snapshots") == 1
    assert database.execute_scalar("SELECT COUNT(*) FROM lexical_relations") >= 5
    checkpoint_data = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert checkpoint_data["schema_version"] == 2
    assert checkpoint_data["next_line"] == 4
    assert checkpoint_data["artifact_sha256"] == artifact.sha256
    assert len(checkpoint_data["configuration_sha256"]) == 64

    resumed = importer.import_artifact(artifact, checkpoint_path=checkpoint)

    assert resumed.first_line == 4
    assert resumed.lines_read == 0
    assert resumed.write_report.attempted == 0
    assert database.execute_scalar("SELECT COUNT(*) FROM lexical_entries") == 2


def test_checkpoint_rejects_changed_filters_before_source_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "filtered.jsonl"
    checkpoint = tmp_path / "filtered.checkpoint.json"
    _write_jsonl(path, _records())
    database = DBManager(tmp_path / "filtered.sqlite")
    artifact = inspect_artifact(path)
    _importer(database, languages=("en",), batch_size=1).import_artifact(
        artifact,
        checkpoint_path=checkpoint,
        max_entries=1,
    )
    retrieved_at = database.execute_scalar(
        "SELECT retrieved_at FROM source_snapshots LIMIT 1"
    )

    with pytest.raises(KaikkiImportError, match="configuration does not match"):
        _importer(database, languages=("fr",), batch_size=1).import_artifact(
            artifact,
            checkpoint_path=checkpoint,
        )

    assert (
        database.execute_scalar("SELECT retrieved_at FROM source_snapshots LIMIT 1")
        == retrieved_at
    )


def test_checkpoint_cannot_skip_entries_in_another_database(tmp_path: Path) -> None:
    path = tmp_path / "database-bound.jsonl"
    checkpoint = tmp_path / "database-bound.checkpoint.json"
    _write_jsonl(path, _records()[:1])
    artifact = inspect_artifact(path)
    first_database = DBManager(tmp_path / "first.sqlite")
    _importer(first_database).import_artifact(artifact, checkpoint_path=checkpoint)
    second_database = DBManager(tmp_path / "second.sqlite")

    with pytest.raises(KaikkiImportError, match="configuration does not match"):
        _importer(second_database).import_artifact(
            artifact,
            checkpoint_path=checkpoint,
        )

    assert second_database.execute_scalar("SELECT COUNT(*) FROM source_snapshots") == 0


def test_replaying_without_checkpoint_updates_in_place(tmp_path: Path) -> None:
    path = tmp_path / "sample.jsonl"
    _write_jsonl(path, _records()[:2])
    database = DBManager(tmp_path / "replay.sqlite")
    importer = _importer(database)
    artifact = inspect_artifact(path)

    first = importer.import_artifact(artifact)
    second = importer.import_artifact(artifact)

    assert first.write_report.inserted == 2
    assert second.write_report.updated == 2
    assert database.execute_scalar("SELECT COUNT(*) FROM lexical_entries") == 2
    assert database.execute_scalar("SELECT COUNT(*) FROM source_snapshots") == 1


def test_bzip2_input_is_streamed_without_extraction(tmp_path: Path) -> None:
    path = tmp_path / "sample.jsonl.bz2"
    payload = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in _records()
    )
    with bz2.open(path, "wt", encoding="utf-8") as handle:
        handle.write(payload)
    database = DBManager(tmp_path / "compressed.sqlite")

    report = _importer(database, batch_size=2).import_artifact(
        inspect_artifact(path), max_entries=1
    )

    assert report.parsed_entries == 1
    assert report.write_report.inserted == 1
    metadata = database.execute_scalar("SELECT metadata_json FROM source_snapshots")
    assert json.loads(metadata)["compression"] == "bzip2"


def test_malformed_line_reports_last_committed_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    valid = json.dumps(_records()[0], ensure_ascii=False)
    path.write_text(valid + "\n{not-json}\n", encoding="utf-8")
    checkpoint = tmp_path / "broken.checkpoint.json"
    database = DBManager(tmp_path / "broken.sqlite")

    with pytest.raises(KaikkiImportError) as caught:
        _importer(database, batch_size=1).import_artifact(
            inspect_artifact(path), checkpoint_path=checkpoint
        )

    assert caught.value.line_number == 2
    assert caught.value.committed_through == 1
    assert database.execute_scalar("SELECT COUNT(*) FROM lexical_entries") == 1
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["next_line"] == 2


def test_changed_artifact_is_rejected_before_snapshot_registration(
    tmp_path: Path,
) -> None:
    path = tmp_path / "changed.jsonl"
    _write_jsonl(path, _records()[:1])
    identity = inspect_artifact(path)
    path.write_text("{}\n", encoding="utf-8")
    database = DBManager(tmp_path / "changed.sqlite")

    with pytest.raises(KaikkiImportError, match="changed after hashing"):
        _importer(database).import_artifact(identity)

    assert database.execute_scalar("SELECT COUNT(*) FROM source_snapshots") == 0
