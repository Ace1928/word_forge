from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import word_forge.multilingual.runtime as multilingual_runtime
from word_forge.multilingual.runtime import run_multilingual_ingest, run_multilingual_repair


def test_run_multilingual_repair_backfills_definition_and_pos(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    source_path = repo_root / "data" / "kaikki_sample.jsonl"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        json.dumps(
            {
                "word": "casa",
                "lang": "Português",
                "lang_code": "pt",
                "pos": "noun",
                "senses": [{"glosses": ["house"]}],
                "translations": [{"lang": "English", "lang_code": "en", "word": "house"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"

    run_multilingual_ingest(
        repo_root=repo_root,
        source_path=source_path,
        source_type="kaikki",
        db_path=db_path,
        force=True,
        include_languages=["pt"],
    )

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "INSERT INTO words (term, definition, part_of_speech, usage_examples, last_refreshed) VALUES (?, ?, ?, '', strftime('%s','now'))",
            ("thinword", "en translation linked to 'seed'.", ""),
        )
        thinword_id = conn.execute("SELECT id FROM words WHERE term = 'thinword'").fetchone()[0]
        conn.execute(
            "INSERT OR IGNORE INTO relationships (word_id, related_term, relationship_type) VALUES (?, ?, 'translation')",
            (thinword_id, "seed"),
        )
        conn.commit()

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "UPDATE words SET definition = ?, part_of_speech = '' WHERE term = 'house'",
            ("en translation linked to 'casa'.",),
        )
        conn.commit()

    result = run_multilingual_repair(repo_root=repo_root, db_path=db_path, apply=True)
    assert result["status"]["status"] == "completed"
    assert result["report"]["repaired_count"] >= 1

    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute(
            "SELECT definition, part_of_speech FROM words WHERE term = 'house'"
        ).fetchone()
    assert row is not None
    assert row[0]
    assert row[0] != "en translation linked to 'casa'."
    assert row[1] == "noun"


def test_run_multilingual_repair_prioritizes_dense_placeholder_entries(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    source_path = repo_root / "data" / "kaikki_sample.jsonl"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        json.dumps(
            {
                "word": "casa",
                "lang": "Português",
                "lang_code": "pt",
                "pos": "noun",
                "senses": [{"glosses": ["house"]}],
                "translations": [{"lang": "English", "lang_code": "en", "word": "house"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"

    run_multilingual_ingest(
        repo_root=repo_root,
        source_path=source_path,
        source_type="kaikki",
        db_path=db_path,
        force=True,
        include_languages=["pt"],
    )

    with sqlite3.connect(str(db_path)) as conn:
        house_id = conn.execute("SELECT id FROM words WHERE term = 'house'").fetchone()[0]
        conn.execute(
            "UPDATE words SET definition = ?, part_of_speech = '' WHERE term = 'house'",
            ("en translation linked to 'casa'.",),
        )
        conn.execute(
            "INSERT INTO words (term, definition, part_of_speech, usage_examples, last_refreshed) VALUES (?, ?, ?, '', strftime('%s','now'))",
            ("thinword", "en translation linked to 'seed'.", ""),
        )
        thinword_id = conn.execute("SELECT id FROM words WHERE term = 'thinword'").fetchone()[0]
        for related in ["casa", "home", "dwelling", "habitation"]:
            conn.execute(
                "INSERT OR IGNORE INTO relationships (word_id, related_term, relationship_type) VALUES (?, ?, 'translation')",
                (house_id, related),
            )
        conn.execute(
            "INSERT OR IGNORE INTO relationships (word_id, related_term, relationship_type) VALUES (?, ?, 'translation')",
            (thinword_id, "seed"),
        )
        conn.commit()

    result = run_multilingual_repair(repo_root=repo_root, db_path=db_path, apply=False, limit=1)
    assert result["report"]["candidate_count"] == 1
    assert result["report"]["repaired_preview"][0]["term"] == "house"


def test_run_multilingual_repair_marks_interrupted_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    source_path = repo_root / "data" / "kaikki_sample.jsonl"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        json.dumps(
            {
                "word": "casa",
                "lang": "Português",
                "lang_code": "pt",
                "pos": "noun",
                "senses": [{"glosses": ["house"]}],
                "translations": [{"lang": "English", "lang_code": "en", "word": "house"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"

    run_multilingual_ingest(
        repo_root=repo_root,
        source_path=source_path,
        source_type="kaikki",
        db_path=db_path,
        force=True,
        include_languages=["pt"],
    )

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "INSERT INTO words (term, definition, part_of_speech, usage_examples, last_refreshed) VALUES (?, ?, ?, '', strftime('%s','now'))",
            ("thinword", "en translation linked to 'seed'.", ""),
        )
        thinword_id = conn.execute("SELECT id FROM words WHERE term = 'thinword'").fetchone()[0]
        conn.execute(
            "INSERT OR IGNORE INTO relationships (word_id, related_term, relationship_type) VALUES (?, ?, 'translation')",
            (thinword_id, "seed"),
        )
        conn.commit()

    def _interrupt(_conn: sqlite3.Connection, _row: sqlite3.Row):
        raise KeyboardInterrupt("test interrupt")

    monkeypatch.setattr(multilingual_runtime, "_repair_translation_word_entry", _interrupt)

    with pytest.raises(KeyboardInterrupt):
        run_multilingual_repair(repo_root=repo_root, db_path=db_path, apply=False)

    status_path = repo_root / "data" / "runtime" / "word_forge_multilingual_repair_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "interrupted"
    assert status["phase"] == "interrupted"


def test_run_multilingual_repair_emits_running_progress_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    source_path = repo_root / "data" / "kaikki_sample.jsonl"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "word": "casa",
                        "lang": "Português",
                        "lang_code": "pt",
                        "pos": "noun",
                        "senses": [{"glosses": ["house"]}],
                        "translations": [{"lang": "English", "lang_code": "en", "word": "house"}],
                    }
                ),
                json.dumps(
                    {
                        "word": "hogar",
                        "lang": "Español",
                        "lang_code": "es",
                        "pos": "noun",
                        "senses": [{"glosses": ["home"]}],
                        "translations": [{"lang": "English", "lang_code": "en", "word": "home"}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"

    run_multilingual_ingest(
        repo_root=repo_root,
        source_path=source_path,
        source_type="kaikki",
        db_path=db_path,
        force=True,
        include_languages=["pt", "es"],
    )

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "UPDATE words SET definition = ?, part_of_speech = '' WHERE term = 'house'",
            ("en translation linked to 'casa'.",),
        )
        conn.execute(
            "UPDATE words SET definition = ?, part_of_speech = '' WHERE term = 'home'",
            ("en translation linked to 'hogar'.",),
        )
        conn.commit()

    captured_statuses: list[dict[str, object]] = []
    original_write_json = multilingual_runtime.write_json

    def _capture_write_json(path: Path, payload: dict[str, object]) -> None:
        if path.name == "word_forge_multilingual_repair_status.json":
            captured_statuses.append(dict(payload))
        original_write_json(path, payload)

    monkeypatch.setattr(multilingual_runtime, "write_json", _capture_write_json)

    result = run_multilingual_repair(repo_root=repo_root, db_path=db_path, apply=False, limit=2)
    assert result["status"]["status"] == "completed"
    assert any(
        row.get("status") == "running" and int(row.get("processed_count") or 0) >= 1
        for row in captured_statuses
    )
