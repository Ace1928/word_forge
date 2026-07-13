from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import word_forge.completion_runtime as completion_runtime
from word_forge.completion_runtime import (
    run_word_forge_completion_audit,
    run_word_forge_completion_daemon,
)
from word_forge.database.database_manager import DBManager


def test_run_word_forge_completion_audit_emits_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"
    vector_db_path = repo_root / "data" / "word_forge_fasttext.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    vector_db_path.parent.mkdir(parents=True, exist_ok=True)

    manager = DBManager(db_path=db_path)
    manager.create_tables()
    manager.close()

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "INSERT INTO words (term, definition, part_of_speech, usage_examples, last_refreshed) VALUES (?, ?, ?, ?, strftime('%s','now'))",
            ("archive", "a place where records are kept", "noun", "The archive is searchable."),
        )
        word_id = conn.execute("SELECT id FROM words WHERE term = 'archive'").fetchone()[0]
        conn.execute(
            "INSERT INTO phonetics (word_id, ipa, arpabet, stress_pattern, source, is_exception, last_updated) VALUES (?, ?, ?, ?, ?, 0, strftime('%s','now'))",
            (word_id, "ˈɑɹ.kaɪv", "AA1 R K AY2 V", "10", "test"),
        )
        conn.execute(
            "INSERT INTO relationships (word_id, related_term, relationship_type) VALUES (?, ?, 'translation')",
            (word_id, "archivo"),
        )
        conn.execute(
            "INSERT INTO lexemes (lemma, lang, part_of_speech, gloss, base_term, source, last_refreshed) VALUES (?, ?, ?, ?, ?, ?, strftime('%s','now'))",
            ("archivo", "es", "noun", "archive", "archive", "test"),
        )
        lexeme_id = conn.execute(
            "SELECT id FROM lexemes WHERE lemma = 'archivo' AND lang = 'es'"
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO translations (lexeme_id, target_lang, target_term, relation, source, last_refreshed) VALUES (?, 'en', 'archive', 'translation', 'test', strftime('%s','now'))",
            (lexeme_id,),
        )
        conn.execute(
            "INSERT INTO phrases (text, phoneme_template, prosody_priors, affect_link, is_recurring, novelty_score, last_seen) VALUES (?, ?, '{}', ?, 1, 0.1, strftime('%s','now'))",
            ("public archive", "P AH B L IH K AA R K AY V", "curious"),
        )
        conn.commit()

    with sqlite3.connect(str(vector_db_path)) as conn:
        conn.execute("CREATE TABLE aligned_vectors (lang TEXT, term TEXT, updated_at TEXT)")
        conn.execute(
            "INSERT INTO aligned_vectors (lang, term, updated_at) VALUES ('es', 'archivo', 'now')"
        )
        conn.commit()

    monkeypatch.setattr(
        completion_runtime,
        "_semantic_vector_snapshot",
        lambda repo_root: {
            "backend": "sqlite-hnsw",
            "path": str(repo_root / "semantic-index"),
            "exists": True,
            "file_count": 1,
            "size_bytes": 1234,
            "vector_count": 3,
            "word_vector_count": 1,
            "definition_vector_count": 1,
            "example_vector_count": 1,
        },
    )

    result = run_word_forge_completion_audit(
        repo_root=repo_root, db_path=db_path, vector_db_path=vector_db_path
    )
    assert result["status"]["status"] == "completed"
    assert result["report"]["contract"] == "eidos.word_forge.completion.audit.v1"
    assert result["report"]["lexicon"]["word_count"] == 1
    assert result["report"]["dimensions"]["definitions"]["ratio"] == 1.0
    assert result["report"]["language_coverage"][0]["lang"] == "es"
    assert Path(result["artifacts"]["latest_json"]).exists()


def test_run_word_forge_completion_daemon_one_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"
    vector_db_path = repo_root / "data" / "word_forge_fasttext.sqlite"
    rawdata_html_path = repo_root / "docs" / "research" / "bundle.html"
    download_dir = repo_root / "word_forge" / "data" / "multilingual" / "kaikki"
    download_dir.mkdir(parents=True, exist_ok=True)
    rawdata_html_path.parent.mkdir(parents=True, exist_ok=True)
    rawdata_html_path.write_text("<html></html>", encoding="utf-8")

    audits = [
        {
            "completion_ratio": 0.2,
            "lexicon": {"word_count": 10, "defined_word_count": 5},
            "multilingual": {"word_count": 10, "language_count": 2, "lexeme_count": 20},
            "phrases": {"phrase_count": 0},
            "semantic_index": {"word_vector_count": 3},
            "aligned_vectors": {"aligned_vector_count": 4},
            "dimensions": {
                "definitions": {"ratio": 0.5},
                "part_of_speech": {"ratio": 0.4},
                "usage_examples": {"ratio": 0.3},
                "phonetics": {"ratio": 0.1},
                "emotion": {"ratio": 0.1},
                "derivation": {"ratio": 0.2},
                "translations": {"ratio": 0.4},
                "semantic_word_vectors": {"ratio": 0.3},
                "aligned_vectors": {"ratio": 0.2},
            },
            "coverage_focus": [
                {
                    "dimension": "phonetics",
                    "ratio": 0.1,
                    "gap": 0.9,
                    "recommended_family": "revisit",
                    "preferred_mode": "semantic",
                }
            ],
            "next_actions": ["phonetics"],
        },
        {
            "completion_ratio": 0.6,
            "lexicon": {"word_count": 16, "defined_word_count": 12},
            "multilingual": {"word_count": 16, "language_count": 3, "lexeme_count": 30},
            "phrases": {"phrase_count": 2},
            "semantic_index": {"word_vector_count": 12},
            "aligned_vectors": {"aligned_vector_count": 18},
            "dimensions": {
                "definitions": {"ratio": 0.75},
                "part_of_speech": {"ratio": 0.7},
                "usage_examples": {"ratio": 0.55},
                "phonetics": {"ratio": 0.4},
                "emotion": {"ratio": 0.3},
                "derivation": {"ratio": 0.45},
                "translations": {"ratio": 0.65},
                "semantic_word_vectors": {"ratio": 0.75},
                "aligned_vectors": {"ratio": 0.6},
            },
            "coverage_focus": [
                {
                    "dimension": "emotion",
                    "ratio": 0.3,
                    "gap": 0.7,
                    "recommended_family": "emotion",
                    "preferred_mode": "semantic",
                }
            ],
            "next_actions": ["emotion"],
        },
    ]
    state = {"index": 0}

    def fake_build(*, repo_root: Path, db_path: Path, vector_db_path: Path) -> dict[str, object]:
        idx = min(state["index"], len(audits) - 1)
        payload = audits[idx]
        state["index"] += 1
        return payload

    monkeypatch.setattr(completion_runtime, "build_word_forge_completion_audit", fake_build)
    monkeypatch.setattr(
        completion_runtime,
        "run_word_forge_completion_audit",
        lambda **kwargs: {
            "status": {"status": "completed"},
            "report": audits[-1],
            "artifacts": {
                "latest_json": str(repo_root / "reports" / "word_forge_completion" / "latest.json")
            },
        },
    )
    monkeypatch.setattr(
        completion_runtime,
        "run_multilingual_batch_ingest",
        lambda **kwargs: {
            "status": {
                "status": "completed",
                "phase": "completed",
                "download_count": 2,
                "ingest_count": 2,
            }
        },
    )
    monkeypatch.setattr(
        completion_runtime,
        "run_multilingual_repair",
        lambda **kwargs: {
            "status": {
                "status": "completed",
                "phase": "completed",
                "candidate_count": 10,
                "repaired_count": 9,
            }
        },
    )
    monkeypatch.setattr(
        completion_runtime,
        "run_word_forge_growth",
        lambda **kwargs: {
            "status": {
                "status": "completed",
                "phase": "completed",
                "lexicon_term_delta": 6,
                "semantic_vector_delta": 8,
                "relationship_delta": 12,
            }
        },
    )
    monkeypatch.setattr(completion_runtime, "run_emotion_annotation", lambda **kwargs: True)
    monkeypatch.setattr(completion_runtime, "run_vector_index", lambda **kwargs: True)

    result = run_word_forge_completion_daemon(
        repo_root=repo_root,
        db_path=db_path,
        vector_db_path=vector_db_path,
        rawdata_html_path=rawdata_html_path,
        download_dir=download_dir,
        cycles=1,
        enable_multilingual=True,
        enable_repair=True,
        enable_growth=True,
        enable_emotion=True,
        enable_vector_index=True,
    )
    assert result["status"] == "completed"
    latest = completion_runtime.read_json(
        repo_root / "data" / "runtime" / "word_forge_completion_daemon_status.json"
    )
    assert latest["cycle_count"] == 1
    assert latest["progress"]["current"]["current_word_count"] == 16
    operations = latest["current_cycle"]["operations"]
    assert [row["family"] for row in operations] == [
        "multilingual_batch",
        "multilingual_repair",
        "growth",
        "emotion_annotation",
        "semantic_vector_index",
    ]
