from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from word_forge.database.database_manager import DBManager
from word_forge.forge import run_emotion_annotation, run_vector_index
from word_forge.growth_runtime import _resolve_growth_focus, run_word_forge_growth
from word_forge.multilingual.runtime import (
    append_history,
    db_counts,
    now_iso,
    read_json,
    run_multilingual_batch_ingest,
    run_multilingual_repair,
    write_json,
)

REPORT_DIR_NAME = "word_forge_completion"
STATUS_FILE_NAME = "word_forge_completion_status.json"
HISTORY_FILE_NAME = "word_forge_completion_history.jsonl"
DAEMON_STATUS_FILE_NAME = "word_forge_completion_daemon_status.json"
DAEMON_HISTORY_FILE_NAME = "word_forge_completion_daemon_history.jsonl"
DAEMON_STOP_FILE_NAME = "word_forge_completion_daemon_stop.json"


def _report_dir(repo_root: Path) -> Path:
    return repo_root / "reports" / REPORT_DIR_NAME


def _status_path(repo_root: Path) -> Path:
    return repo_root / "data" / "runtime" / STATUS_FILE_NAME


def _history_path(repo_root: Path) -> Path:
    return repo_root / "data" / "runtime" / HISTORY_FILE_NAME


def _daemon_status_path(repo_root: Path) -> Path:
    return repo_root / "data" / "runtime" / DAEMON_STATUS_FILE_NAME


def _daemon_history_path(repo_root: Path) -> Path:
    return repo_root / "data" / "runtime" / DAEMON_HISTORY_FILE_NAME


def _daemon_stop_path(repo_root: Path) -> Path:
    return repo_root / "data" / "runtime" / DAEMON_STOP_FILE_NAME


def _safe_scalar(conn: sqlite3.Connection, query: str, params: tuple[Any, ...] = ()) -> int:
    try:
        row = conn.execute(query, params).fetchone()
    except Exception:
        return 0
    if not row:
        return 0
    value = row[0] if isinstance(row, (tuple, list, sqlite3.Row)) else row
    try:
        return int(value or 0)
    except Exception:
        return 0


def _safe_text(conn: sqlite3.Connection, query: str, params: tuple[Any, ...] = ()) -> str:
    try:
        row = conn.execute(query, params).fetchone()
    except Exception:
        return ""
    if not row:
        return ""
    value = row[0] if isinstance(row, (tuple, list, sqlite3.Row)) else row
    return str(value or "")


def _sqlite_tables(db_path: Path) -> set[str]:
    if not db_path.exists():
        return set()
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {str(row[0]) for row in rows}


def _phrase_snapshot(db_path: Path) -> dict[str, int]:
    if not db_path.exists():
        return {
            "phrase_count": 0,
            "recurring_phrase_count": 0,
            "phrase_with_phoneme_template_count": 0,
            "phrase_with_affect_link_count": 0,
            "phrase_component_count": 0,
            "phrase_realization_count": 0,
        }
    tables = _sqlite_tables(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        return {
            "phrase_count": _safe_scalar(conn, "SELECT COUNT(*) FROM phrases")
            if "phrases" in tables
            else 0,
            "recurring_phrase_count": _safe_scalar(
                conn, "SELECT COUNT(*) FROM phrases WHERE COALESCE(is_recurring, 0) != 0"
            )
            if "phrases" in tables
            else 0,
            "phrase_with_phoneme_template_count": _safe_scalar(
                conn,
                "SELECT COUNT(*) FROM phrases WHERE COALESCE(trim(phoneme_template), '') <> ''",
            )
            if "phrases" in tables
            else 0,
            "phrase_with_affect_link_count": _safe_scalar(
                conn, "SELECT COUNT(*) FROM phrases WHERE COALESCE(trim(affect_link), '') <> ''"
            )
            if "phrases" in tables
            else 0,
            "phrase_component_count": _safe_scalar(conn, "SELECT COUNT(*) FROM phrase_components")
            if "phrase_components" in tables
            else 0,
            "phrase_realization_count": _safe_scalar(
                conn, "SELECT COUNT(*) FROM phrase_realizations"
            )
            if "phrase_realizations" in tables
            else 0,
        }


def _lexicon_snapshot(db_path: Path) -> dict[str, int]:
    if not db_path.exists():
        return {
            "word_count": 0,
            "defined_word_count": 0,
            "pos_word_count": 0,
            "usage_example_word_count": 0,
            "phonetic_word_count": 0,
            "emotion_word_count": 0,
            "derivation_word_count": 0,
            "relationship_count": 0,
            "translation_word_count": 0,
        }
    tables = _sqlite_tables(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        return {
            "word_count": _safe_scalar(conn, "SELECT COUNT(*) FROM words")
            if "words" in tables
            else 0,
            "defined_word_count": _safe_scalar(
                conn, "SELECT COUNT(*) FROM words WHERE COALESCE(trim(definition), '') <> ''"
            )
            if "words" in tables
            else 0,
            "pos_word_count": _safe_scalar(
                conn, "SELECT COUNT(*) FROM words WHERE COALESCE(trim(part_of_speech), '') <> ''"
            )
            if "words" in tables
            else 0,
            "usage_example_word_count": _safe_scalar(
                conn, "SELECT COUNT(*) FROM words WHERE COALESCE(trim(usage_examples), '') <> ''"
            )
            if "words" in tables
            else 0,
            "phonetic_word_count": _safe_scalar(
                conn, "SELECT COUNT(DISTINCT word_id) FROM phonetics"
            )
            if "phonetics" in tables
            else 0,
            "emotion_word_count": _safe_scalar(conn, "SELECT COUNT(*) FROM word_emotion")
            if "word_emotion" in tables
            else 0,
            "derivation_word_count": _safe_scalar(
                conn,
                "SELECT COUNT(DISTINCT word_id) FROM relationships WHERE lower(relationship_type) IN ('derived_from', 'etymological_source')",
            )
            if "relationships" in tables
            else 0,
            "relationship_count": _safe_scalar(conn, "SELECT COUNT(*) FROM relationships")
            if "relationships" in tables
            else 0,
            "translation_word_count": _safe_scalar(
                conn,
                "SELECT COUNT(DISTINCT word_id) FROM relationships WHERE lower(relationship_type) = 'translation'",
            )
            if "relationships" in tables
            else 0,
        }


def _semantic_vector_snapshot(repo_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "backend": "unavailable",
        "path": "",
        "exists": False,
        "file_count": 0,
        "size_bytes": 0,
        "vector_count": 0,
        "word_vector_count": 0,
        "definition_vector_count": 0,
        "example_vector_count": 0,
    }
    try:
        from word_forge.config import config
        from word_forge.vectorizer.vector_store import SQLITE_DB_FILENAME

        index_path = Path(str(config.vectorizer.index_path)).resolve()
        sqlite_path = index_path / SQLITE_DB_FILENAME
        files = (
            [entry for entry in index_path.rglob("*") if entry.is_file()]
            if index_path.exists()
            else []
        )
        result.update(
            {
                "path": str(index_path),
                "exists": index_path.exists(),
                "file_count": len(files),
                "size_bytes": sum(int(entry.stat().st_size) for entry in files),
            }
        )
        if sqlite_path.exists():
            with sqlite3.connect(str(sqlite_path)) as conn:
                result["vector_count"] = _safe_scalar(conn, "SELECT COUNT(*) FROM vectors")
                result["word_vector_count"] = _safe_scalar(
                    conn, "SELECT COUNT(*) FROM vectors WHERE substr(id, 1, 2) = 'w_'"
                )
                result["definition_vector_count"] = _safe_scalar(
                    conn, "SELECT COUNT(*) FROM vectors WHERE substr(id, 1, 2) = 'd_'"
                )
                result["example_vector_count"] = _safe_scalar(
                    conn, "SELECT COUNT(*) FROM vectors WHERE substr(id, 1, 2) = 'e_'"
                )
                result["backend"] = (
                    _safe_text(conn, "SELECT value FROM collection_meta WHERE key = 'backend'")
                    or "sqlite-persistent"
                )
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _aligned_vector_snapshot(vector_db_path: Path) -> dict[str, Any]:
    result = {
        "path": str(vector_db_path),
        "exists": vector_db_path.exists(),
        "aligned_vector_count": 0,
        "translation_candidate_count": 0,
        "language_count": 0,
    }
    if not vector_db_path.exists():
        return result
    tables = _sqlite_tables(vector_db_path)
    with sqlite3.connect(str(vector_db_path)) as conn:
        if "aligned_vectors" in tables:
            result["aligned_vector_count"] = _safe_scalar(
                conn, "SELECT COUNT(*) FROM aligned_vectors"
            )
            result["language_count"] = _safe_scalar(
                conn, "SELECT COUNT(DISTINCT lang) FROM aligned_vectors"
            )
        if "translation_candidates" in tables:
            result["translation_candidate_count"] = _safe_scalar(
                conn, "SELECT COUNT(*) FROM translation_candidates"
            )
    return result


def _language_coverage(db_path: Path, *, limit: int = 16) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    counts = db_counts(db_path)
    top_languages = counts.get("top_languages") if isinstance(counts, dict) else []
    top_translation_languages = (
        counts.get("top_translation_languages") if isinstance(counts, dict) else []
    )
    translation_lookup = {
        str(row.get("lang") or ""): int(row.get("count") or 0)
        for row in (top_translation_languages or [])
        if isinstance(row, dict) and row.get("lang")
    }
    coverage: list[dict[str, Any]] = []
    for row in (top_languages or [])[: max(1, int(limit))]:
        if not isinstance(row, dict) or not row.get("lang"):
            continue
        lang = str(row.get("lang") or "")
        lexeme_count = int(row.get("count") or 0)
        coverage.append(
            {
                "lang": lang,
                "lexeme_count": lexeme_count,
                "translation_count": int(translation_lookup.get(lang) or 0),
                "linked_word_count": None,
                "definition_ratio": None,
                "pos_ratio": None,
                "phonetic_ratio": None,
                "translation_ratio": None,
            }
        )
    return coverage


def _ratio(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(max(0.0, min(1.0, float(count) / float(total))), 6)


def build_word_forge_completion_audit(
    *,
    repo_root: Path,
    db_path: Path,
    vector_db_path: Path,
) -> dict[str, Any]:
    multilingual = db_counts(db_path)
    lexicon = _lexicon_snapshot(db_path)
    phrases = _phrase_snapshot(db_path)
    semantic = _semantic_vector_snapshot(repo_root)
    aligned = _aligned_vector_snapshot(vector_db_path)
    word_count = int(lexicon.get("word_count") or multilingual.get("word_count") or 0)
    weak_dimensions = _resolve_growth_focus(lexicon)[:6]
    vector_word_count = min(word_count, int(semantic.get("word_vector_count") or 0))
    phrase_count = int(phrases.get("phrase_count") or 0)
    phrase_phonetic_count = int(phrases.get("phrase_with_phoneme_template_count") or 0)
    phrase_affect_count = int(phrases.get("phrase_with_affect_link_count") or 0)
    dimensions = {
        "definitions": {
            "count": int(lexicon.get("defined_word_count") or 0),
            "total": word_count,
            "ratio": _ratio(int(lexicon.get("defined_word_count") or 0), word_count),
        },
        "part_of_speech": {
            "count": int(lexicon.get("pos_word_count") or 0),
            "total": word_count,
            "ratio": _ratio(int(lexicon.get("pos_word_count") or 0), word_count),
        },
        "usage_examples": {
            "count": int(lexicon.get("usage_example_word_count") or 0),
            "total": word_count,
            "ratio": _ratio(int(lexicon.get("usage_example_word_count") or 0), word_count),
        },
        "phonetics": {
            "count": int(lexicon.get("phonetic_word_count") or 0),
            "total": word_count,
            "ratio": _ratio(int(lexicon.get("phonetic_word_count") or 0), word_count),
        },
        "emotion": {
            "count": int(lexicon.get("emotion_word_count") or 0),
            "total": word_count,
            "ratio": _ratio(int(lexicon.get("emotion_word_count") or 0), word_count),
        },
        "derivation": {
            "count": int(lexicon.get("derivation_word_count") or 0),
            "total": word_count,
            "ratio": _ratio(int(lexicon.get("derivation_word_count") or 0), word_count),
        },
        "translations": {
            "count": int(lexicon.get("translation_word_count") or 0),
            "total": word_count,
            "ratio": _ratio(int(lexicon.get("translation_word_count") or 0), word_count),
        },
        "semantic_word_vectors": {
            "count": vector_word_count,
            "total": word_count,
            "ratio": _ratio(vector_word_count, word_count),
        },
        "aligned_vectors": {
            "count": int(aligned.get("aligned_vector_count") or 0),
            "total": max(1, int(multilingual.get("lexeme_count") or 0)),
            "ratio": _ratio(
                int(aligned.get("aligned_vector_count") or 0),
                int(multilingual.get("lexeme_count") or 0),
            ),
        },
        "phrases": {
            "count": phrase_count,
            "total": phrase_count,
            "ratio": 1.0 if phrase_count else 0.0,
        },
        "phrase_phonetics": {
            "count": phrase_phonetic_count,
            "total": phrase_count,
            "ratio": _ratio(phrase_phonetic_count, phrase_count),
        },
        "phrase_affect": {
            "count": phrase_affect_count,
            "total": phrase_count,
            "ratio": _ratio(phrase_affect_count, phrase_count),
        },
    }
    tracked = [row.get("ratio") for row in dimensions.values() if isinstance(row, dict)]
    completion_ratio = round(
        sum(float(value or 0.0) for value in tracked) / float(len(tracked or [1])), 6
    )
    coverage_focus = [
        {
            "dimension": str(item.get("dimension") or ""),
            "ratio": float(item.get("ratio") or 0.0),
            "gap": float(item.get("gap") or 0.0),
            "recommended_family": str(item.get("recommended_family") or ""),
            "preferred_mode": str(item.get("preferred_mode") or "community"),
        }
        for item in weak_dimensions
        if isinstance(item, dict)
    ]
    language_coverage = _language_coverage(db_path)
    return {
        "contract": "eidos.word_forge.completion.audit.v1",
        "generated_at": now_iso(),
        "repo_root": str(repo_root),
        "db_path": str(db_path),
        "vector_db_path": str(vector_db_path),
        "completion_ratio": completion_ratio,
        "multilingual": multilingual,
        "lexicon": lexicon,
        "phrases": phrases,
        "semantic_index": semantic,
        "aligned_vectors": aligned,
        "dimensions": dimensions,
        "coverage_focus": coverage_focus,
        "language_coverage": language_coverage,
        "next_actions": [
            item.get("dimension") for item in coverage_focus[:4] if item.get("dimension")
        ],
    }


def _write_completion_report(repo_root: Path, report: dict[str, Any]) -> dict[str, str]:
    report_dir = _report_dir(repo_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime_safe_filename(report.get("generated_at") or now_iso())
    latest_json = report_dir / "latest.json"
    timestamped_json = report_dir / f"{timestamp}.json"
    write_json(latest_json, report)
    write_json(timestamped_json, report)
    return {
        "latest_json": str(latest_json),
        "timestamped_json": str(timestamped_json),
    }


def datetime_safe_filename(value: Any) -> str:
    text = str(value or now_iso())
    return text.replace(":", "").replace("+", "_plus_")


def run_word_forge_completion_audit(
    *,
    repo_root: Path,
    db_path: Path,
    vector_db_path: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    status_path = _status_path(repo_root)
    history_path = _history_path(repo_root)
    status_payload: dict[str, Any] = {
        "contract": "eidos.word_forge.completion.status.v1",
        "status": "running",
        "phase": "audit",
        "started_at": now_iso(),
        "repo_root": str(repo_root),
        "db_path": str(db_path),
        "vector_db_path": str(vector_db_path),
    }
    write_json(status_path, status_payload)
    try:
        report = build_word_forge_completion_audit(
            repo_root=repo_root, db_path=db_path, vector_db_path=vector_db_path
        )
        artifacts = _write_completion_report(repo_root, report)
        status_payload.update(
            {
                "status": "completed",
                "phase": "completed",
                "finished_at": now_iso(),
                "report_path": artifacts["latest_json"],
                "completion_ratio": float(report.get("completion_ratio") or 0.0),
                "word_count": int((report.get("lexicon") or {}).get("word_count") or 0),
                "defined_word_count": int(
                    (report.get("lexicon") or {}).get("defined_word_count") or 0
                ),
                "semantic_word_vector_count": int(
                    (report.get("semantic_index") or {}).get("word_vector_count") or 0
                ),
                "aligned_vector_count": int(
                    (report.get("aligned_vectors") or {}).get("aligned_vector_count") or 0
                ),
                "language_count": int(
                    (report.get("multilingual") or {}).get("language_count") or 0
                ),
                "next_actions": list(report.get("next_actions") or []),
            }
        )
        write_json(status_path, status_payload)
        append_history(history_path, status_payload)
        return {"status": status_payload, "report": report, "artifacts": artifacts}
    except Exception as exc:
        status_payload.update(
            {"status": "failed", "phase": "failed", "finished_at": now_iso(), "error": str(exc)}
        )
        write_json(status_path, status_payload)
        append_history(history_path, status_payload)
        raise


def request_word_forge_completion_daemon_stop(*, repo_root: Path) -> dict[str, Any]:
    payload = {
        "contract": "eidos.word_forge.completion_daemon.stop.v1",
        "status": "requested",
        "requested_at": now_iso(),
    }
    write_json(_daemon_stop_path(repo_root.resolve()), payload)
    return payload


def _target_progress(audit: dict[str, Any], *, target_word_count: int | None) -> dict[str, Any]:
    raw_lexicon = audit.get("lexicon")
    lexicon: dict[str, Any] = raw_lexicon if isinstance(raw_lexicon, dict) else {}
    word_count = int(lexicon.get("word_count") or 0)
    semantic_word_vector_count = int(
        (audit.get("semantic_index") or {}).get("word_vector_count") or 0
    )
    raw_dimensions = audit.get("dimensions")
    dimensions: dict[str, Any] = raw_dimensions if isinstance(raw_dimensions, dict) else {}
    raw_semantic_dimension = dimensions.get("semantic_word_vectors")
    semantic_dimension: dict[str, Any] = (
        raw_semantic_dimension if isinstance(raw_semantic_dimension, dict) else {}
    )
    vector_ratio = float(semantic_dimension.get("ratio") or 0.0)
    return {
        "current_word_count": word_count,
        "semantic_word_vector_count": semantic_word_vector_count,
        "semantic_vector_ratio": vector_ratio,
        "target_word_count": int(target_word_count) if target_word_count is not None else None,
        "word_progress_ratio": _ratio(word_count, int(target_word_count or 0))
        if target_word_count
        else None,
        "words_remaining": max(0, int(target_word_count or 0) - word_count)
        if target_word_count
        else None,
        "completion_ratio": float(audit.get("completion_ratio") or 0.0),
    }


def run_word_forge_completion_daemon(
    *,
    repo_root: Path,
    db_path: Path,
    vector_db_path: Path,
    rawdata_html_path: Path,
    download_dir: Path,
    interval_seconds: int = 1800,
    preferred_languages: list[str] | None = None,
    include_languages: list[str] | None = None,
    max_languages: int | None = 0,
    max_compressed_bytes: int | None = 0,
    repair_limit: int | None = None,
    growth_forge_local_limit: int | None = None,
    growth_lexicon_limit: int | None = None,
    growth_thesaurus_limit: int | None = None,
    growth_phonetic_limit: int | None = None,
    cycles: int | None = None,
    target_word_count: int | None = None,
    force_first_cycle: bool = False,
    enable_multilingual: bool = True,
    enable_repair: bool = True,
    enable_growth: bool = True,
    enable_emotion: bool = True,
    enable_vector_index: bool = True,
    vector_timeout: float = 900.0,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    db_path = db_path.resolve()
    vector_db_path = vector_db_path.resolve()
    rawdata_html_path = rawdata_html_path.resolve()
    download_dir = download_dir.resolve()
    interval_seconds = max(60, int(interval_seconds))
    status_path = _daemon_status_path(repo_root)
    history_path = _daemon_history_path(repo_root)
    stop_path = _daemon_stop_path(repo_root)
    if stop_path.exists():
        stop_path.unlink()
    pid = os.getpid()
    initial = build_word_forge_completion_audit(
        repo_root=repo_root, db_path=db_path, vector_db_path=vector_db_path
    )
    status_payload: dict[str, Any] = {
        "contract": "eidos.word_forge.completion_daemon.status.v1",
        "status": "running",
        "phase": "starting",
        "pid": pid,
        "repo_root": str(repo_root),
        "db_path": str(db_path),
        "vector_db_path": str(vector_db_path),
        "rawdata_html_path": str(rawdata_html_path),
        "download_dir": str(download_dir),
        "interval_seconds": interval_seconds,
        "preferred_languages": list(preferred_languages or []),
        "include_languages": list(include_languages or []),
        "max_languages": None
        if max_languages is None or int(max_languages) <= 0
        else int(max_languages),
        "max_compressed_bytes": None
        if max_compressed_bytes is None or int(max_compressed_bytes) <= 0
        else int(max_compressed_bytes),
        "repair_limit": repair_limit,
        "cycles_requested": cycles,
        "target_word_count": int(target_word_count) if target_word_count is not None else None,
        "started_at": now_iso(),
        "cycle_count": 0,
        "progress": {
            "initial": _target_progress(initial, target_word_count=target_word_count),
            "current": _target_progress(initial, target_word_count=target_word_count),
            "eta_seconds": None,
        },
        "latest_audit": initial,
    }
    write_json(status_path, status_payload)
    append_history(history_path, status_payload)

    try:
        cycle_count = 0
        while True:
            if stop_path.exists():
                status_payload.update(
                    {
                        "status": "stopped",
                        "phase": "stopped",
                        "finished_at": now_iso(),
                        "stop_request": read_json(stop_path),
                    }
                )
                write_json(status_path, status_payload)
                append_history(history_path, status_payload)
                return status_payload
            if cycles is not None and cycle_count >= max(1, int(cycles)):
                status_payload.update(
                    {"status": "completed", "phase": "completed", "finished_at": now_iso()}
                )
                write_json(status_path, status_payload)
                append_history(history_path, status_payload)
                return status_payload

            before_audit = build_word_forge_completion_audit(
                repo_root=repo_root, db_path=db_path, vector_db_path=vector_db_path
            )
            before_progress = _target_progress(before_audit, target_word_count=target_word_count)
            if target_word_count is not None and int(
                before_progress.get("current_word_count") or 0
            ) >= int(target_word_count):
                status_payload.update(
                    {
                        "status": "completed",
                        "phase": "target_reached",
                        "finished_at": now_iso(),
                        "progress": {
                            **(status_payload.get("progress") or {}),
                            "current": before_progress,
                            "eta_seconds": 0,
                        },
                        "latest_audit": before_audit,
                    }
                )
                write_json(status_path, status_payload)
                append_history(history_path, status_payload)
                return status_payload

            cycle_count += 1
            cycle_started = time.time()
            current_cycle: dict[str, Any] = {
                "cycle_number": cycle_count,
                "started_at": now_iso(),
                "phase": "audit",
                "before": before_progress,
                "operations": [],
                "coverage_focus": list(before_audit.get("coverage_focus") or [])[:6],
            }
            status_payload.update(
                {
                    "phase": "completion_cycle",
                    "cycle_count": cycle_count,
                    "current_cycle": current_cycle,
                }
            )
            write_json(status_path, status_payload)

            force_cycle = bool(force_first_cycle and cycle_count == 1)
            if enable_multilingual:
                current_cycle["phase"] = "multilingual_batch"
                write_json(status_path, status_payload)
                batch_run = run_multilingual_batch_ingest(
                    repo_root=repo_root,
                    rawdata_html_path=rawdata_html_path,
                    download_dir=download_dir,
                    db_path=db_path,
                    preferred_languages=preferred_languages,
                    include_languages=include_languages,
                    max_languages=max_languages,
                    max_compressed_bytes=max_compressed_bytes,
                    force_download=False,
                    force_ingest=force_cycle,
                    limit=None,
                )
                raw_batch_status = batch_run.get("status")
                batch_status: dict[str, Any] = (
                    raw_batch_status if isinstance(raw_batch_status, dict) else {}
                )
                current_cycle["operations"].append(
                    {
                        "family": "multilingual_batch",
                        "status": str(batch_status.get("status") or "unknown"),
                        "phase": str(batch_status.get("phase") or ""),
                        "download_count": int(batch_status.get("download_count") or 0),
                        "ingest_count": int(batch_status.get("ingest_count") or 0),
                    }
                )

            if enable_repair:
                current_cycle["phase"] = "multilingual_repair"
                write_json(status_path, status_payload)
                repair_run = run_multilingual_repair(
                    repo_root=repo_root, db_path=db_path, limit=repair_limit, apply=True
                )
                raw_repair_status = repair_run.get("status")
                repair_status: dict[str, Any] = (
                    raw_repair_status if isinstance(raw_repair_status, dict) else {}
                )
                current_cycle["operations"].append(
                    {
                        "family": "multilingual_repair",
                        "status": str(repair_status.get("status") or "unknown"),
                        "phase": str(repair_status.get("phase") or ""),
                        "candidate_count": int(repair_status.get("candidate_count") or 0),
                        "repaired_count": int(repair_status.get("repaired_count") or 0),
                    }
                )

            if enable_growth:
                current_cycle["phase"] = "growth"
                write_json(status_path, status_payload)
                growth_run = run_word_forge_growth(
                    repo_root=repo_root,
                    db_path=db_path,
                    vector_db_path=vector_db_path,
                    force=force_cycle,
                    multilingual_limit=None,
                    fasttext_limit=None,
                    polyglot_limit=None,
                    forge_local_limit=growth_forge_local_limit,
                    lexicon_limit=growth_lexicon_limit,
                    thesaurus_limit=growth_thesaurus_limit,
                    phonetic_limit=growth_phonetic_limit,
                )
                raw_growth_status = growth_run.get("status")
                growth_status: dict[str, Any] = (
                    raw_growth_status if isinstance(raw_growth_status, dict) else {}
                )
                current_cycle["operations"].append(
                    {
                        "family": "growth",
                        "status": str(growth_status.get("status") or "unknown"),
                        "phase": str(growth_status.get("phase") or ""),
                        "lexicon_term_delta": int(growth_status.get("lexicon_term_delta") or 0),
                        "semantic_vector_delta": int(
                            growth_status.get("semantic_vector_delta") or 0
                        ),
                        "relationship_delta": int(growth_status.get("relationship_delta") or 0),
                    }
                )

            if enable_emotion:
                current_cycle["phase"] = "emotion_annotation"
                write_json(status_path, status_payload)
                emotion_ok = run_emotion_annotation(
                    db_manager=DBManager(db_path=db_path),
                    timeout=max(60.0, min(float(vector_timeout), 600.0)),
                )
                current_cycle["operations"].append(
                    {
                        "family": "emotion_annotation",
                        "status": "completed" if emotion_ok else "partial",
                        "phase": "completed" if emotion_ok else "timeout",
                    }
                )

            if enable_vector_index:
                current_cycle["phase"] = "semantic_vector_index"
                write_json(status_path, status_payload)
                vector_ok = run_vector_index(
                    db_manager=DBManager(db_path=db_path), timeout=float(vector_timeout)
                )
                current_cycle["operations"].append(
                    {
                        "family": "semantic_vector_index",
                        "status": "completed" if vector_ok else "partial",
                        "phase": "completed" if vector_ok else "timeout",
                    }
                )

            current_cycle["phase"] = "audit"
            after_run = run_word_forge_completion_audit(
                repo_root=repo_root, db_path=db_path, vector_db_path=vector_db_path
            )
            raw_after_audit = after_run.get("report")
            after_audit: dict[str, Any] = (
                raw_after_audit
                if isinstance(raw_after_audit, dict)
                else build_word_forge_completion_audit(
                    repo_root=repo_root, db_path=db_path, vector_db_path=vector_db_path
                )
            )
            after_progress = _target_progress(after_audit, target_word_count=target_word_count)
            current_cycle.update(
                {
                    "finished_at": now_iso(),
                    "duration_seconds": round(max(0.0, time.time() - cycle_started), 3),
                    "after": after_progress,
                    "completion_ratio": float(after_audit.get("completion_ratio") or 0.0),
                    "coverage_focus": list(after_audit.get("coverage_focus") or [])[:6],
                }
            )
            eta_seconds = None
            if target_word_count is not None:
                words_delta = max(
                    0,
                    int(after_progress.get("current_word_count") or 0)
                    - int(before_progress.get("current_word_count") or 0),
                )
                words_remaining = int(after_progress.get("words_remaining") or 0)
                if words_delta > 0:
                    eta_seconds = int(
                        (float(words_remaining) / float(words_delta))
                        * max(1.0, time.time() - cycle_started)
                    )
            status_payload.update(
                {
                    "phase": "sleeping",
                    "last_completion_finished_at": now_iso(),
                    "latest_audit": after_audit,
                    "progress": {
                        **(status_payload.get("progress") or {}),
                        "current": after_progress,
                        "eta_seconds": eta_seconds,
                    },
                    "current_cycle": current_cycle,
                }
            )
            write_json(status_path, status_payload)
            append_history(history_path, status_payload)

            if cycles is not None and cycle_count >= max(1, int(cycles)):
                continue
            time.sleep(interval_seconds)
    except Exception as exc:
        status_payload.update(
            {"status": "failed", "phase": "failed", "finished_at": now_iso(), "error": str(exc)}
        )
        write_json(status_path, status_payload)
        append_history(history_path, status_payload)
        raise
