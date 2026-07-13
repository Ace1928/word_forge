from __future__ import annotations

import gzip
import json
import sqlite3
from pathlib import Path

from word_forge.multilingual.runtime import (
    build_translation_language_aliases,
    run_multilingual_ingest,
    run_multilingual_normalize,
)


def test_build_translation_language_aliases_prefers_lang_code(tmp_path: Path) -> None:
    source_path = tmp_path / "pt-extract.jsonl.gz"
    rows = [
        {
            "word": "casa",
            "lang": "Português",
            "lang_code": "pt",
            "translations": [{"lang": "inglês", "lang_code": "en", "word": "house"}],
        },
        {
            "word": "livro",
            "lang": "Português",
            "lang_code": "pt",
            "translations": [{"lang": "inglês", "lang_code": "en", "word": "book"}],
        },
        {
            "word": "livro",
            "lang": "Português",
            "lang_code": "pt",
            "translations": [{"lang": "francês", "lang_code": "fr", "word": "livre"}],
        },
    ]
    with gzip.open(source_path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    alias_map = build_translation_language_aliases([source_path], per_source_limit=100)
    assert alias_map["inglês"]["code"] == "en"
    assert alias_map["francês"]["code"] == "fr"
    assert "en" not in alias_map


def test_run_multilingual_normalize_updates_target_lang_codes(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    source_path = (
        repo_root / "word_forge" / "data" / "multilingual" / "kaikki" / "pt-extract.jsonl.gz"
    )
    source_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "word": "casa",
            "lang": "Português",
            "lang_code": "pt",
            "pos": "noun",
            "senses": [{"glosses": ["house"]}],
            "translations": [{"lang": "inglês", "lang_code": "en", "word": "house"}],
        }
    ]
    with gzip.open(source_path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

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
        conn.execute("UPDATE translations SET target_lang = 'inglês' WHERE target_lang = 'en'")
        conn.commit()

    result = run_multilingual_normalize(
        repo_root=repo_root, db_path=db_path, per_source_limit=100, apply=True
    )
    assert result["status"]["status"] == "completed"
    assert result["report"]["result"]["updated_count"] >= 1
    with sqlite3.connect(str(db_path)) as conn:
        langs = {row[0] for row in conn.execute("SELECT DISTINCT target_lang FROM translations")}
    assert "en" in langs
    assert "inglês" not in langs
