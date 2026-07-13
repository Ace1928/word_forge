from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path

from word_forge.multilingual.runtime import run_multilingual_ingest


def test_run_multilingual_ingest_writes_status_and_reports(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    source_path = repo_root / "data" / "wiktextract_sample.jsonl"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        json.dumps(
            {
                "word": "bonjour",
                "lang": "fr",
                "pos": "interj",
                "senses": [{"glosses": ["hello"], "examples": [{"text": "bonjour le monde"}]}],
                "translations": [{"lang": "en", "word": "hello"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"

    result = run_multilingual_ingest(
        repo_root=repo_root,
        source_path=source_path,
        source_type="wiktextract",
        db_path=db_path,
    )

    assert result["status"]["status"] == "completed"
    assert result["report"]["deltas"]["word_delta"] >= 2
    assert result["report"]["deltas"]["relationship_delta"] >= 2
    assert result["report"]["deltas"]["lexeme_delta"] == 1
    assert result["report"]["deltas"]["translation_delta"] == 1
    assert result["report"]["deltas"]["base_aligned_delta"] == 1
    from word_forge.database.database_manager import DBManager

    db = DBManager(db_path=db_path)
    assert db.get_word_id("hello") > 0
    assert db.get_word_id("bonjour") > 0
    entry = db.get_word_entry("bonjour")
    assert entry["part_of_speech"] == "interj"
    assert entry["usage_examples"]
    assert (repo_root / "data" / "runtime" / "word_forge_multilingual_ingest_status.json").exists()
    assert (
        repo_root / "data" / "runtime" / "word_forge_multilingual_ingest_history.jsonl"
    ).exists()
    assert Path(result["artifacts"]["latest_json"]).exists()
    assert Path(result["artifacts"]["latest_markdown"]).exists()


def test_run_multilingual_ingest_skips_unchanged_source(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    source_path = repo_root / "data" / "kaikki_sample.jsonl"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        json.dumps(
            {
                "word": "hola",
                "lang": "es",
                "pos": "interj",
                "senses": [{"glosses": ["hello"]}],
                "translations": [{"lang": "en", "word": "hello"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"

    first = run_multilingual_ingest(
        repo_root=repo_root,
        source_path=source_path,
        source_type="kaikki",
        db_path=db_path,
    )
    second = run_multilingual_ingest(
        repo_root=repo_root,
        source_path=source_path,
        source_type="kaikki",
        db_path=db_path,
    )

    assert first["status"]["status"] == "completed"
    assert second["status"]["status"] == "skipped"
    assert second["status"]["phase"] == "unchanged"


def test_run_multilingual_ingest_supports_gzip_and_language_filters(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    source_path = repo_root / "data" / "kaikki_sample.jsonl.gz"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "word": "hola",
            "lang": "Español",
            "lang_code": "es",
            "pos": "interj",
            "senses": [{"glosses": ["hello"], "examples": [{"text": "hola mundo"}]}],
            "translations": [
                {"lang": "English", "lang_code": "en", "word": "hello"},
                {"lang": "French", "lang_code": "fr", "word": "bonjour"},
            ],
        },
        {
            "word": "bonjour",
            "lang": "Français",
            "lang_code": "fr",
            "pos": "interj",
            "senses": [{"glosses": ["hello"]}],
            "translations": [{"lang": "English", "lang_code": "en", "word": "hello"}],
        },
    ]
    with gzip.open(source_path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"
    result = run_multilingual_ingest(
        repo_root=repo_root,
        source_path=source_path,
        source_type="kaikki",
        db_path=db_path,
        include_languages=["es"],
    )

    assert result["status"]["status"] == "completed"
    assert result["report"]["after"]["lexeme_count"] == 1
    assert result["report"]["after"]["translation_count"] == 2
    assert result["report"]["after"]["top_translation_languages"][0]["lang"] == "en"
    assert result["report"]["after"]["word_count"] >= 3
    assert result["report"]["deltas"]["usage_word_delta"] >= 1

    from word_forge.database.database_manager import DBManager

    db = DBManager(db_path=db_path)
    assert db.get_word_id("hola") > 0
    assert db.get_word_id("hello") > 0
    assert db.get_word_id("bonjour") > 0


def test_run_multilingual_ingest_avoids_expected_miss_error_spam(tmp_path: Path, caplog) -> None:
    repo_root = tmp_path / "repo"
    source_path = repo_root / "data" / "kaikki_sample.jsonl"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        json.dumps(
            {
                "word": "hallo",
                "lang": "Deutsch",
                "lang_code": "de",
                "pos": "interj",
                "senses": [{"glosses": ["hello"]}],
                "translations": [
                    {"lang": "English", "lang_code": "en", "word": "hello"},
                    {"lang": "French", "lang_code": "fr", "word": "bonjour"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"

    with caplog.at_level(logging.ERROR):
        result = run_multilingual_ingest(
            repo_root=repo_root,
            source_path=source_path,
            source_type="kaikki",
            db_path=db_path,
        )

    assert result["status"]["status"] == "completed"
    error_messages = [record.getMessage() for record in caplog.records]
    assert not any(
        "get_word_entry raised TermNotFoundError" in message for message in error_messages
    )
    assert not any("get_word_id raised TermNotFoundError" in message for message in error_messages)
