from __future__ import annotations

import json
import signal
import sqlite3
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from word_forge.database.database_manager import DBManager
from word_forge.multilingual import ingest_kaikki_jsonl, ingest_wiktextract_jsonl
from word_forge.multilingual.multilingual_manager import MultilingualManager
from word_forge.multilingual.source_catalog import (
    build_kaikki_catalog_report,
    choose_download_candidates,
    download_catalog_entry,
)
from word_forge.multilingual.wiktextract_ingest import _iter_jsonl


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def append_history(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def db_counts(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        return {
            "word_count": 0,
            "defined_word_count": 0,
            "usage_word_count": 0,
            "lexeme_count": 0,
            "translation_count": 0,
            "base_aligned_count": 0,
            "relationship_count": 0,
            "language_count": 0,
            "top_languages": [],
            "top_translation_languages": [],
        }
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        word_count = int(conn.execute("SELECT COUNT(*) FROM words").fetchone()[0])
        defined_word_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM words WHERE COALESCE(trim(definition), '') <> ''"
            ).fetchone()[0]
        )
        usage_word_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM words WHERE COALESCE(trim(usage_examples), '') <> ''"
            ).fetchone()[0]
        )
        lexeme_count = int(conn.execute("SELECT COUNT(*) FROM lexemes").fetchone()[0])
        translation_count = int(conn.execute("SELECT COUNT(*) FROM translations").fetchone()[0])
        base_aligned_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM lexemes WHERE COALESCE(base_term, '') != ''"
            ).fetchone()[0]
        )
        relationship_count = int(conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0])
        languages = conn.execute(
            "SELECT lang, COUNT(*) AS count FROM lexemes GROUP BY lang ORDER BY count DESC LIMIT 12"
        ).fetchall()
        translation_languages = conn.execute(
            "SELECT target_lang AS lang, COUNT(*) AS count FROM translations GROUP BY target_lang ORDER BY count DESC LIMIT 12"
        ).fetchall()
    return {
        "word_count": word_count,
        "defined_word_count": defined_word_count,
        "usage_word_count": usage_word_count,
        "lexeme_count": lexeme_count,
        "translation_count": translation_count,
        "base_aligned_count": base_aligned_count,
        "relationship_count": relationship_count,
        "language_count": len(languages),
        "top_languages": [{"lang": row["lang"], "count": int(row["count"])} for row in languages],
        "top_translation_languages": [
            {"lang": row["lang"], "count": int(row["count"])} for row in translation_languages
        ],
    }


def build_multilingual_ingest_report(
    source_path: Path,
    source_type: str,
    db_path: Path,
    *,
    limit: int | None = None,
    include_languages: list[str] | None = None,
    exclude_languages: list[str] | None = None,
    max_examples_per_entry: int = 4,
    max_translations_per_entry: int = 24,
    materialize_words: bool = True,
) -> dict[str, Any]:
    manager = MultilingualManager(DBManager(db_path=db_path))
    before = db_counts(db_path)
    if source_type == "kaikki":
        ingest_kaikki_jsonl(
            str(source_path),
            manager=manager,
            limit=limit,
            include_languages=include_languages,
            exclude_languages=exclude_languages,
            max_examples_per_entry=max_examples_per_entry,
            max_translations_per_entry=max_translations_per_entry,
            materialize_words=materialize_words,
        )
    elif source_type == "wiktextract":
        ingest_wiktextract_jsonl(
            str(source_path),
            manager=manager,
            limit=limit,
            include_languages=include_languages,
            exclude_languages=exclude_languages,
            max_examples_per_entry=max_examples_per_entry,
            max_translations_per_entry=max_translations_per_entry,
            materialize_words=materialize_words,
        )
    else:
        raise ValueError(f"Unsupported source_type: {source_type}")
    after = db_counts(db_path)
    deltas = {
        "word_delta": after["word_count"] - before["word_count"],
        "defined_word_delta": after["defined_word_count"] - before["defined_word_count"],
        "usage_word_delta": after["usage_word_count"] - before["usage_word_count"],
        "lexeme_delta": after["lexeme_count"] - before["lexeme_count"],
        "translation_delta": after["translation_count"] - before["translation_count"],
        "base_aligned_delta": after["base_aligned_count"] - before["base_aligned_count"],
        "relationship_delta": after["relationship_count"] - before["relationship_count"],
    }
    return {
        "contract": "eidos.word_forge.multilingual_ingest.v1",
        "generated_at": now_iso(),
        "source_type": source_type,
        "source_path": str(source_path),
        "source_signature": source_signature(source_path),
        "db_path": str(db_path),
        "limit": limit,
        "include_languages": include_languages or [],
        "exclude_languages": exclude_languages or [],
        "max_examples_per_entry": int(max_examples_per_entry),
        "max_translations_per_entry": int(max_translations_per_entry),
        "materialize_words": bool(materialize_words),
        "before": before,
        "after": after,
        "deltas": deltas,
    }


def render_multilingual_ingest_markdown(report: dict[str, Any]) -> str:
    after = report.get("after") or {}
    deltas = report.get("deltas") or {}
    top_languages = after.get("top_languages") or []
    top_translation_languages = after.get("top_translation_languages") or []
    lines = [
        "# Word Forge Multilingual Ingest",
        "",
        f"- Generated: `{report.get('generated_at')}`",
        f"- Source type: `{report.get('source_type')}`",
        f"- Source path: `{report.get('source_path')}`",
        f"- Lexeme delta: `{deltas.get('lexeme_delta', 0)}`",
        f"- Translation delta: `{deltas.get('translation_delta', 0)}`",
        f"- Base aligned delta: `{deltas.get('base_aligned_delta', 0)}`",
        f"- Word delta: `{deltas.get('word_delta', 0)}`",
        f"- Relationship delta: `{deltas.get('relationship_delta', 0)}`",
        "",
        "## Totals",
        "",
        f"- Words: `{after.get('word_count', 0)}`",
        f"- Defined words: `{after.get('defined_word_count', 0)}`",
        f"- Usage-rich words: `{after.get('usage_word_count', 0)}`",
        f"- Lexemes: `{after.get('lexeme_count', 0)}`",
        f"- Translations: `{after.get('translation_count', 0)}`",
        f"- Base aligned: `{after.get('base_aligned_count', 0)}`",
        f"- Relationships: `{after.get('relationship_count', 0)}`",
        f"- Languages: `{after.get('language_count', 0)}`",
        "",
        "## Top Lexeme Languages",
        "",
    ]
    lines.extend(
        [f"- `{row.get('lang')}`: `{row.get('count')}`" for row in top_languages] or ["- None"]
    )
    lines.extend(["", "## Top Translation Languages", ""])
    lines.extend(
        [f"- `{row.get('lang')}`: `{row.get('count')}`" for row in top_translation_languages]
        or ["- None"]
    )
    return "\n".join(lines) + "\n"


def write_multilingual_ingest_report(repo_root: Path, report: dict[str, Any]) -> dict[str, str]:
    report_dir = repo_root / "reports" / "word_forge_multilingual_ingest"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = report_dir / f"word_forge_multilingual_ingest_{stamp}.json"
    md_path = report_dir / f"word_forge_multilingual_ingest_{stamp}.md"
    latest_json = report_dir / "latest.json"
    latest_md = report_dir / "latest.md"
    write_json(json_path, report)
    md_path.write_text(render_multilingual_ingest_markdown(report), encoding="utf-8")
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    return {
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "latest_json": str(latest_json),
        "latest_markdown": str(latest_md),
    }


def run_multilingual_ingest(
    repo_root: Path,
    source_path: Path,
    source_type: str,
    db_path: Path,
    *,
    limit: int | None = None,
    force: bool = False,
    include_languages: list[str] | None = None,
    exclude_languages: list[str] | None = None,
    max_examples_per_entry: int = 4,
    max_translations_per_entry: int = 24,
    materialize_words: bool = True,
) -> dict[str, Any]:
    runtime_dir = repo_root / "data" / "runtime"
    status_path = runtime_dir / "word_forge_multilingual_ingest_status.json"
    history_path = runtime_dir / "word_forge_multilingual_ingest_history.jsonl"
    signature = source_signature(source_path)
    previous = read_json(status_path)
    status_payload: dict[str, Any] = {
        "contract": "eidos.word_forge.multilingual_ingest.status.v1",
        "status": "running",
        "phase": "ingesting",
        "source_type": source_type,
        "source_path": str(source_path),
        "source_signature": signature,
        "db_path": str(db_path),
        "started_at": now_iso(),
        "limit": limit,
        "force": force,
        "include_languages": include_languages or [],
        "exclude_languages": exclude_languages or [],
        "max_examples_per_entry": int(max_examples_per_entry),
        "max_translations_per_entry": int(max_translations_per_entry),
        "materialize_words": bool(materialize_words),
    }
    write_json(status_path, status_payload)
    try:
        if (
            not force
            and previous.get("source_signature") == signature
            and previous.get("status") == "completed"
        ):
            status_payload.update(
                {"status": "skipped", "phase": "unchanged", "finished_at": now_iso()}
            )
            write_json(status_path, status_payload)
            append_history(history_path, status_payload)
            return {"status": status_payload, "report": None, "artifacts": {}}

        report = build_multilingual_ingest_report(
            source_path,
            source_type,
            db_path,
            limit=limit,
            include_languages=include_languages,
            exclude_languages=exclude_languages,
            max_examples_per_entry=max_examples_per_entry,
            max_translations_per_entry=max_translations_per_entry,
            materialize_words=materialize_words,
        )
        artifacts = write_multilingual_ingest_report(repo_root, report)
        status_payload.update(
            {
                "status": "completed",
                "phase": "completed",
                "finished_at": now_iso(),
                "report_path": artifacts["latest_json"],
                "lexeme_delta": report["deltas"]["lexeme_delta"],
                "translation_delta": report["deltas"]["translation_delta"],
                "base_aligned_delta": report["deltas"]["base_aligned_delta"],
                "word_delta": report["deltas"]["word_delta"],
                "relationship_delta": report["deltas"]["relationship_delta"],
            }
        )
        write_json(status_path, status_payload)
        append_history(history_path, status_payload)
        return {"status": status_payload, "report": report, "artifacts": artifacts}
    except Exception as exc:
        status_payload.update(
            {
                "status": "failed",
                "phase": "failed",
                "finished_at": now_iso(),
                "error": str(exc),
            }
        )
        write_json(status_path, status_payload)
        append_history(history_path, status_payload)
        raise


def write_multilingual_source_report(repo_root: Path, report: dict[str, Any]) -> dict[str, str]:
    report_dir = repo_root / "reports" / "word_forge_multilingual_sources"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = report_dir / f"word_forge_multilingual_sources_{stamp}.json"
    latest_json = report_dir / "latest.json"
    write_json(json_path, report)
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    return {"json_path": str(json_path), "latest_json": str(latest_json)}


def write_multilingual_source_download_report(
    repo_root: Path, report: dict[str, Any]
) -> dict[str, str]:
    report_dir = repo_root / "reports" / "word_forge_multilingual_source_download"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = report_dir / f"word_forge_multilingual_source_download_{stamp}.json"
    latest_json = report_dir / "latest.json"
    write_json(json_path, report)
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    return {"json_path": str(json_path), "latest_json": str(latest_json)}


def write_multilingual_batch_report(repo_root: Path, report: dict[str, Any]) -> dict[str, str]:
    report_dir = repo_root / "reports" / "word_forge_multilingual_batch"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = report_dir / f"word_forge_multilingual_batch_{stamp}.json"
    latest_json = report_dir / "latest.json"
    write_json(json_path, report)
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    return {"json_path": str(json_path), "latest_json": str(latest_json)}


def run_multilingual_source_catalog(
    repo_root: Path,
    *,
    rawdata_html_path: Path,
    download_dir: Path,
    preferred_languages: list[str] | None = None,
) -> dict[str, Any]:
    runtime_dir = repo_root / "data" / "runtime"
    status_path = runtime_dir / "word_forge_multilingual_sources_status.json"
    history_path = runtime_dir / "word_forge_multilingual_sources_history.jsonl"
    status_payload: dict[str, Any] = {
        "contract": "eidos.word_forge.multilingual.sources.status.v1",
        "status": "running",
        "phase": "catalog",
        "rawdata_html_path": str(rawdata_html_path),
        "download_dir": str(download_dir),
        "preferred_languages": preferred_languages or [],
        "started_at": now_iso(),
    }
    write_json(status_path, status_payload)
    try:
        report = build_kaikki_catalog_report(
            rawdata_html_path,
            download_dir=download_dir,
            preferred_languages=preferred_languages,
        )
        artifacts = write_multilingual_source_report(repo_root, report)
        status_payload.update(
            {
                "status": "completed",
                "phase": "completed",
                "finished_at": now_iso(),
                "report_path": artifacts["latest_json"],
                "language_count": int(report.get("language_count") or 0),
                "downloaded_count": int(report.get("downloaded_count") or 0),
                "preferred_count": int(report.get("preferred_count") or 0),
            }
        )
        write_json(status_path, status_payload)
        append_history(history_path, status_payload)
        return {"status": status_payload, "report": report, "artifacts": artifacts}
    except Exception as exc:
        status_payload.update(
            {
                "status": "failed",
                "phase": "failed",
                "finished_at": now_iso(),
                "error": str(exc),
            }
        )
        write_json(status_path, status_payload)
        append_history(history_path, status_payload)
        raise


def run_multilingual_source_download(
    repo_root: Path,
    *,
    rawdata_html_path: Path,
    download_dir: Path,
    preferred_languages: list[str] | None = None,
    include_languages: list[str] | None = None,
    max_languages: int | None = 0,
    max_compressed_bytes: int | None = 0,
    force: bool = False,
) -> dict[str, Any]:
    normalized_max_languages = (
        None if max_languages is None or int(max_languages) <= 0 else int(max_languages)
    )
    normalized_max_compressed_bytes = (
        None
        if max_compressed_bytes is None or int(max_compressed_bytes) <= 0
        else int(max_compressed_bytes)
    )
    runtime_dir = repo_root / "data" / "runtime"
    status_path = runtime_dir / "word_forge_multilingual_source_download_status.json"
    history_path = runtime_dir / "word_forge_multilingual_source_download_history.jsonl"
    status_payload = {
        "contract": "eidos.word_forge.multilingual.source_download.status.v1",
        "status": "running",
        "phase": "downloading",
        "rawdata_html_path": str(rawdata_html_path),
        "download_dir": str(download_dir),
        "preferred_languages": preferred_languages or [],
        "include_languages": include_languages or [],
        "max_languages": normalized_max_languages,
        "max_compressed_bytes": normalized_max_compressed_bytes,
        "started_at": now_iso(),
    }
    write_json(status_path, status_payload)
    try:
        catalog = build_kaikki_catalog_report(
            rawdata_html_path,
            download_dir=download_dir,
            preferred_languages=preferred_languages,
        )
        candidates = choose_download_candidates(
            catalog,
            include_languages=include_languages,
            max_languages=normalized_max_languages,
            max_compressed_bytes=normalized_max_compressed_bytes,
        )
        results = [
            download_catalog_entry(entry, download_dir=download_dir, force=force)
            for entry in candidates
        ]
        downloaded_languages = [str(item.get("lang") or "") for item in results if item.get("lang")]
        report = {
            "contract": "eidos.word_forge.multilingual.source_download.v1",
            "generated_at": now_iso(),
            "rawdata_html_path": str(rawdata_html_path),
            "download_dir": str(download_dir),
            "candidate_count": len(candidates),
            "download_count": len(results),
            "downloaded_languages": downloaded_languages,
            "results": results,
        }
        artifacts = write_multilingual_source_download_report(repo_root, report)
        status_payload.update(
            {
                "status": "completed",
                "phase": "completed",
                "finished_at": now_iso(),
                "report_path": artifacts["latest_json"],
                "candidate_count": len(candidates),
                "download_count": len(results),
                "downloaded_languages": downloaded_languages,
            }
        )
        write_json(status_path, status_payload)
        append_history(history_path, status_payload)
        return {"status": status_payload, "report": report, "artifacts": artifacts}
    except Exception as exc:
        status_payload.update(
            {
                "status": "failed",
                "phase": "failed",
                "finished_at": now_iso(),
                "error": str(exc),
            }
        )
        write_json(status_path, status_payload)
        append_history(history_path, status_payload)
        raise


def run_multilingual_batch_ingest(
    repo_root: Path,
    *,
    rawdata_html_path: Path,
    download_dir: Path,
    db_path: Path,
    preferred_languages: list[str] | None = None,
    include_languages: list[str] | None = None,
    max_languages: int | None = 0,
    max_compressed_bytes: int | None = 0,
    force_download: bool = False,
    force_ingest: bool = False,
    limit: int | None = None,
    exclude_languages: list[str] | None = None,
    max_examples_per_entry: int = 4,
    max_translations_per_entry: int = 24,
    materialize_words: bool = True,
) -> dict[str, Any]:
    normalized_max_languages = (
        None if max_languages is None or int(max_languages) <= 0 else int(max_languages)
    )
    normalized_max_compressed_bytes = (
        None
        if max_compressed_bytes is None or int(max_compressed_bytes) <= 0
        else int(max_compressed_bytes)
    )
    runtime_dir = repo_root / "data" / "runtime"
    status_path = runtime_dir / "word_forge_multilingual_batch_status.json"
    history_path = runtime_dir / "word_forge_multilingual_batch_history.jsonl"
    status_payload: dict[str, Any] = {
        "contract": "eidos.word_forge.multilingual.batch.status.v1",
        "status": "running",
        "phase": "download",
        "rawdata_html_path": str(rawdata_html_path),
        "download_dir": str(download_dir),
        "db_path": str(db_path),
        "preferred_languages": preferred_languages or [],
        "include_languages": include_languages or [],
        "max_languages": normalized_max_languages,
        "max_compressed_bytes": normalized_max_compressed_bytes,
        "started_at": now_iso(),
    }
    write_json(status_path, status_payload)
    try:
        catalog = build_kaikki_catalog_report(
            rawdata_html_path,
            download_dir=download_dir,
            preferred_languages=preferred_languages,
        )
        selected_entries = choose_download_candidates(
            catalog,
            include_languages=include_languages,
            max_languages=normalized_max_languages,
            max_compressed_bytes=normalized_max_compressed_bytes,
            include_downloaded=True,
        )
        download_rows: list[dict[str, Any]] = []
        for entry in selected_entries:
            local_path = (
                Path(str(entry.get("local_path") or "")) if entry.get("local_path") else None
            )
            if local_path is not None and local_path.exists() and not force_download:
                download_rows.append(
                    {
                        "lang": entry.get("lang"),
                        "status": "existing",
                        "path": str(local_path),
                        "bytes": int(local_path.stat().st_size),
                        "url": str(entry.get("jsonl_gz_url") or ""),
                    }
                )
                continue
            download_rows.append(
                download_catalog_entry(entry, download_dir=download_dir, force=force_download)
            )
        download_report: dict[str, Any] = {
            "download_count": len(download_rows),
            "downloaded_languages": [
                str(item.get("lang") or "") for item in download_rows if item.get("lang")
            ],
            "results": download_rows,
        }
        download_result: dict[str, Any] = {"report": download_report}
        status_payload.update(
            {
                "phase": "ingest",
                "download_count": len(download_rows),
                "downloaded_languages": download_report["downloaded_languages"],
            }
        )
        write_json(status_path, status_payload)
        results: list[dict[str, Any]] = []
        aggregate: defaultdict[str, int] = defaultdict(int)
        for row in download_rows:
            if not isinstance(row, dict):
                continue
            path_text = str(row.get("path") or "").strip()
            if not path_text:
                continue
            source_path = Path(path_text)
            ingest_result = run_multilingual_ingest(
                repo_root=repo_root,
                source_path=source_path,
                source_type="kaikki",
                db_path=db_path,
                limit=limit,
                force=force_ingest,
                include_languages=include_languages,
                exclude_languages=exclude_languages,
                max_examples_per_entry=max_examples_per_entry,
                max_translations_per_entry=max_translations_per_entry,
                materialize_words=materialize_words,
            )
            raw_ingest_report = ingest_result.get("report")
            ingest_report: dict[str, Any] = (
                raw_ingest_report if isinstance(raw_ingest_report, dict) else {}
            )
            raw_deltas = ingest_report.get("deltas")
            deltas: dict[str, Any] = raw_deltas if isinstance(raw_deltas, dict) else {}
            for key, value in deltas.items():
                aggregate[str(key)] += int(value or 0)
            results.append(
                {
                    "lang": row.get("lang"),
                    "path": path_text,
                    "status": ingest_result.get("status", {}),
                    "deltas": deltas,
                }
            )
        report: dict[str, Any] = {
            "contract": "eidos.word_forge.multilingual.batch.v1",
            "generated_at": now_iso(),
            "rawdata_html_path": str(rawdata_html_path),
            "download_dir": str(download_dir),
            "db_path": str(db_path),
            "download": download_result.get("report") or {},
            "ingest_count": len(results),
            "ingested_languages": [
                str(item.get("lang") or "") for item in results if item.get("lang")
            ],
            "aggregate_deltas": dict(sorted(aggregate.items())),
            "results": results,
        }
        artifacts = write_multilingual_batch_report(repo_root, report)
        status_payload.update(
            {
                "status": "completed",
                "phase": "completed",
                "finished_at": now_iso(),
                "report_path": artifacts["latest_json"],
                "ingest_count": len(results),
                "ingested_languages": report["ingested_languages"],
                "aggregate_deltas": report["aggregate_deltas"],
            }
        )
        write_json(status_path, status_payload)
        append_history(history_path, status_payload)
        return {"status": status_payload, "report": report, "artifacts": artifacts}
    except Exception as exc:
        status_payload.update(
            {
                "status": "failed",
                "phase": "failed",
                "finished_at": now_iso(),
                "error": str(exc),
            }
        )
        write_json(status_path, status_payload)
        append_history(history_path, status_payload)
        raise


def _discover_local_multilingual_sources(repo_root: Path) -> list[Path]:
    candidates: list[Path] = []
    search_roots = [repo_root / "word_forge" / "data", repo_root / "data" / "word_forge"]
    patterns = ("*.jsonl", "*.jsonl.gz")
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            candidates.extend(sorted(root.rglob(pattern)))
    rows: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        if (
            "wiktextract" not in path.name.lower()
            and "extract" not in path.name.lower()
            and "kaikki" not in str(path).lower()
        ):
            continue
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        rows.append(path)
    return rows


def _looks_like_language_code_label(value: str) -> bool:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return False
    if any(not (ch.isascii() and (ch.isalpha() or ch.isdigit() or ch == "-")) for ch in normalized):
        return False
    parts = [part for part in normalized.split("-") if part]
    if not parts:
        return False
    return all(1 < len(part) <= 4 for part in parts)


def build_translation_language_aliases(
    source_paths: Iterable[Path],
    *,
    per_source_limit: int | None = 50_000,
) -> dict[str, dict[str, Any]]:
    alias_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for source_path in source_paths:
        count = 0
        for entry in _iter_jsonl(source_path):
            translations = entry.get("translations") or []
            if not isinstance(translations, list):
                continue
            for translation in translations:
                if not isinstance(translation, dict):
                    continue
                name = str(translation.get("lang") or "").strip().lower()
                code = str(translation.get("lang_code") or "").strip().lower()
                if not name or not code or name == code:
                    continue
                if _looks_like_language_code_label(name):
                    continue
                alias_counts[name][code] += 1
            count += 1
            if per_source_limit is not None and count >= max(1, int(per_source_limit)):
                break
    alias_map: dict[str, dict[str, Any]] = {}
    for name, counter in alias_counts.items():
        code, confidence = counter.most_common(1)[0]
        alias_map[name] = {
            "code": code,
            "confidence": int(confidence),
            "choices": [
                {"code": item_code, "count": int(item_count)}
                for item_code, item_count in counter.most_common(4)
            ],
        }
    return alias_map


def normalize_translation_language_codes(
    db_path: Path,
    alias_map: dict[str, dict[str, Any]],
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    if not db_path.exists():
        return {
            "updated_count": 0,
            "deleted_duplicate_count": 0,
            "candidate_count": 0,
            "normalized_labels": [],
        }
    updated_count = 0
    deleted_duplicate_count = 0
    normalized_labels: list[dict[str, Any]] = []
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, lexeme_id, target_lang, target_term, relation, source, last_refreshed FROM translations ORDER BY id"
        ).fetchall()
        candidate_count = 0
        for row in rows:
            current_lang = str(row["target_lang"] or "").strip().lower()
            mapping = alias_map.get(current_lang)
            if not mapping:
                continue
            normalized_code = str(mapping.get("code") or "").strip().lower()
            if not normalized_code or normalized_code == current_lang:
                continue
            candidate_count += 1
            duplicate = conn.execute(
                "SELECT id FROM translations WHERE lexeme_id = ? AND target_lang = ? AND target_term = ? AND relation = ?",
                (row["lexeme_id"], normalized_code, row["target_term"], row["relation"]),
            ).fetchone()
            if duplicate:
                conn.execute("DELETE FROM translations WHERE id = ?", (row["id"],))
                deleted_duplicate_count += 1
            else:
                conn.execute(
                    "UPDATE translations SET target_lang = ? WHERE id = ?",
                    (normalized_code, row["id"]),
                )
                updated_count += 1
            if len(normalized_labels) < 24:
                normalized_labels.append(
                    {
                        "from": current_lang,
                        "to": normalized_code,
                        "target_term": str(row["target_term"] or ""),
                        "confidence": int(mapping.get("confidence") or 0),
                    }
                )
            if limit is not None and (updated_count + deleted_duplicate_count) >= max(
                1, int(limit)
            ):
                break
        conn.commit()
    return {
        "updated_count": updated_count,
        "deleted_duplicate_count": deleted_duplicate_count,
        "candidate_count": candidate_count,
        "normalized_labels": normalized_labels,
    }


def write_multilingual_normalize_report(repo_root: Path, report: dict[str, Any]) -> dict[str, str]:
    report_dir = repo_root / "reports" / "word_forge_multilingual_normalize"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = report_dir / f"word_forge_multilingual_normalize_{stamp}.json"
    latest_json = report_dir / "latest.json"
    write_json(json_path, report)
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    return {"json_path": str(json_path), "latest_json": str(latest_json)}


def run_multilingual_normalize(
    repo_root: Path,
    db_path: Path,
    *,
    per_source_limit: int | None = 50_000,
    apply: bool = True,
    limit: int | None = None,
) -> dict[str, Any]:
    runtime_dir = repo_root / "data" / "runtime"
    status_path = runtime_dir / "word_forge_multilingual_normalize_status.json"
    history_path = runtime_dir / "word_forge_multilingual_normalize_history.jsonl"
    source_paths = _discover_local_multilingual_sources(repo_root)
    status_payload: dict[str, Any] = {
        "contract": "eidos.word_forge.multilingual.normalize.status.v1",
        "status": "running",
        "phase": "analyzing",
        "db_path": str(db_path),
        "source_count": len(source_paths),
        "source_paths": [str(path) for path in source_paths[:12]],
        "per_source_limit": per_source_limit,
        "apply": bool(apply),
        "limit": limit,
        "started_at": now_iso(),
    }
    write_json(status_path, status_payload)
    try:
        before = db_counts(db_path)
        alias_map = build_translation_language_aliases(
            source_paths, per_source_limit=per_source_limit
        )
        result: dict[str, Any] = (
            normalize_translation_language_codes(db_path, alias_map, limit=limit)
            if apply
            else {
                "updated_count": 0,
                "deleted_duplicate_count": 0,
                "candidate_count": sum(
                    1
                    for row in sqlite3.connect(str(db_path)).execute(
                        "SELECT target_lang FROM translations"
                    )
                    if alias_map.get(str(row[0] or "").strip().lower())
                )
                if db_path.exists()
                else 0,
                "normalized_labels": [],
            }
        )
        after = db_counts(db_path)
        report = {
            "contract": "eidos.word_forge.multilingual.normalize.v1",
            "generated_at": now_iso(),
            "db_path": str(db_path),
            "source_count": len(source_paths),
            "source_paths": [str(path) for path in source_paths[:24]],
            "per_source_limit": per_source_limit,
            "apply": bool(apply),
            "limit": limit,
            "alias_count": len(alias_map),
            "alias_preview": [
                {"label": label, **payload} for label, payload in list(alias_map.items())[:24]
            ],
            "before": before,
            "after": after,
            "result": result,
        }
        artifacts = write_multilingual_normalize_report(repo_root, report)
        status_payload.update(
            {
                "status": "completed",
                "phase": "completed",
                "finished_at": now_iso(),
                "report_path": artifacts["latest_json"],
                "alias_count": len(alias_map),
                "updated_count": int(result.get("updated_count") or 0),
                "deleted_duplicate_count": int(result.get("deleted_duplicate_count") or 0),
                "candidate_count": int(result.get("candidate_count") or 0),
            }
        )
        write_json(status_path, status_payload)
        append_history(history_path, status_payload)
        return {"status": status_payload, "report": report, "artifacts": artifacts}
    except Exception as exc:
        status_payload.update(
            {
                "status": "failed",
                "phase": "failed",
                "finished_at": now_iso(),
                "error": str(exc),
            }
        )
        write_json(status_path, status_payload)
        append_history(history_path, status_payload)
        raise


def _is_placeholder_public_definition(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return True
    return "translation linked to" in normalized or normalized.endswith("lexical entry")


def get_multilingual_repair_backlog_snapshot(db_path: Path) -> dict[str, int]:
    if not db_path.exists():
        return {
            "translation_word_count": 0,
            "missing_definition_count": 0,
            "missing_pos_count": 0,
            "placeholder_definition_count": 0,
            "repair_backlog": 0,
        }
    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute(
            """
            WITH translation_words AS (
              SELECT DISTINCT w.id, w.definition, w.part_of_speech,
                     CASE WHEN COALESCE(trim(w.definition),'')='' THEN 1 ELSE 0 END AS missing_definition,
                     CASE WHEN COALESCE(trim(w.part_of_speech),'')='' THEN 1 ELSE 0 END AS missing_pos,
                     CASE WHEN lower(COALESCE(w.definition,'')) LIKE '%translation linked to%' THEN 1 ELSE 0 END AS placeholder_definition
              FROM words w
              WHERE EXISTS (
                SELECT 1 FROM relationships r WHERE r.word_id=w.id AND r.relationship_type='translation'
              )
            )
            SELECT COUNT(*) AS translation_word_count,
                   SUM(missing_definition) AS missing_definition_count,
                   SUM(missing_pos) AS missing_pos_count,
                   SUM(placeholder_definition) AS placeholder_definition_count,
                   SUM(CASE WHEN missing_definition=1 OR missing_pos=1 OR placeholder_definition=1 THEN 1 ELSE 0 END) AS repair_backlog
            FROM translation_words
            """
        ).fetchone()
    if row is None:
        return {
            "translation_word_count": 0,
            "missing_definition_count": 0,
            "missing_pos_count": 0,
            "placeholder_definition_count": 0,
            "repair_backlog": 0,
        }
    return {
        "translation_word_count": int(row[0] or 0),
        "missing_definition_count": int(row[1] or 0),
        "missing_pos_count": int(row[2] or 0),
        "placeholder_definition_count": int(row[3] or 0),
        "repair_backlog": int(row[4] or 0),
    }


def _majority_nonempty(values: Iterable[str]) -> str:
    counter: Counter[str] = Counter()
    for value in values:
        item = str(value or "").strip()
        if item:
            counter[item] += 1
    return counter.most_common(1)[0][0] if counter else ""


def _translation_source_rows(conn: sqlite3.Connection, term: str) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT w.term, w.definition, w.part_of_speech, w.usage_examples
        FROM relationships r
        JOIN words w ON w.id = r.word_id
        WHERE lower(r.related_term) = lower(?) AND r.relationship_type = 'translation'
        ORDER BY w.last_refreshed DESC, w.term ASC
        LIMIT 24
        """,
        (term,),
    ).fetchall()


def _matching_lexeme_rows(conn: sqlite3.Connection, term: str) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT lemma, lang, part_of_speech, gloss, source, last_refreshed
        FROM lexemes
        WHERE lower(lemma) = lower(?)
        ORDER BY COALESCE(length(gloss), 0) DESC, last_refreshed DESC, lang ASC
        LIMIT 12
        """,
        (term,),
    ).fetchall()


def _repair_translation_word_entry(
    conn: sqlite3.Connection, row: sqlite3.Row
) -> dict[str, Any] | None:
    term = str(row["term"] or "").strip()
    if not term:
        return None
    lexeme_rows = _matching_lexeme_rows(conn, term)
    source_rows = _translation_source_rows(conn, term)
    existing_definition = str(row["definition"] or "").strip()
    existing_pos = str(row["part_of_speech"] or "").strip()

    repaired_pos = existing_pos or _majority_nonempty(
        [str(item["part_of_speech"] or "") for item in lexeme_rows]
        + [str(item["part_of_speech"] or "") for item in source_rows]
    )

    repaired_definition = existing_definition
    if _is_placeholder_public_definition(existing_definition):
        chosen_gloss = next(
            (
                str(item["gloss"] or "").strip()
                for item in lexeme_rows
                if str(item["gloss"] or "").strip()
            ),
            "",
        )
        chosen_lang = next(
            (
                str(item["lang"] or "").strip()
                for item in lexeme_rows
                if str(item["lang"] or "").strip()
            ),
            "",
        )
        if chosen_gloss:
            prefix = " ".join(part for part in [chosen_lang, repaired_pos] if part).strip()
            repaired_definition = f"{prefix} meaning: {chosen_gloss}".strip()
        else:
            source_terms = [
                str(item["term"] or "").strip()
                for item in source_rows
                if str(item["term"] or "").strip()
            ]
            if source_terms:
                repaired_definition = f"Crosslingual translation linked to {', '.join(repr(term) for term in source_terms[:4])}."

    if repaired_definition == existing_definition and repaired_pos == existing_pos:
        return None

    return {
        "term": term,
        "definition": repaired_definition,
        "part_of_speech": repaired_pos,
        "source_term_count": len(source_rows),
        "lexeme_match_count": len(lexeme_rows),
    }


def _write_multilingual_repair_progress(
    status_path: Path,
    status_payload: dict[str, Any],
    *,
    candidate_count: int,
    processed_count: int,
    repaired_count: int,
    current_term: str,
    translation_word_count: int,
    repair_backlog_before: int,
    started_monotonic: float,
) -> None:
    elapsed_seconds = max(0.0, time.monotonic() - started_monotonic)
    batch_progress_ratio = 0.0
    if candidate_count > 0:
        batch_progress_ratio = max(0.0, min(1.0, processed_count / float(candidate_count)))
    completion_ratio = 0.0
    if translation_word_count > 0:
        completion_ratio = max(
            0.0, min(1.0, 1.0 - (repair_backlog_before / float(translation_word_count)))
        )
    batch_eta_seconds = None
    if processed_count > 0 and candidate_count > processed_count and elapsed_seconds > 0.0:
        rate_per_second = processed_count / elapsed_seconds
        if rate_per_second > 0.0:
            batch_eta_seconds = round((candidate_count - processed_count) / rate_per_second, 3)
    status_payload.update(
        {
            "candidate_count": int(candidate_count),
            "processed_count": int(processed_count),
            "repaired_count": int(repaired_count),
            "current_term": current_term,
            "translation_word_count": int(translation_word_count),
            "repair_backlog_before": int(repair_backlog_before),
            "completion_ratio": round(completion_ratio, 6),
            "batch_progress_ratio": round(batch_progress_ratio, 6),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "batch_eta_seconds": batch_eta_seconds,
        }
    )
    write_json(status_path, status_payload)


def write_multilingual_repair_report(repo_root: Path, report: dict[str, Any]) -> dict[str, str]:
    report_dir = repo_root / "reports" / "word_forge_multilingual_repair"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = report_dir / f"word_forge_multilingual_repair_{stamp}.json"
    latest_json = report_dir / "latest.json"
    write_json(json_path, report)
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    return {"json_path": str(json_path), "latest_json": str(latest_json)}


def run_multilingual_repair(
    repo_root: Path,
    db_path: Path,
    *,
    limit: int | None = None,
    apply: bool = True,
) -> dict[str, Any]:
    commit_interval = 250
    runtime_dir = repo_root / "data" / "runtime"
    status_path = runtime_dir / "word_forge_multilingual_repair_status.json"
    history_path = runtime_dir / "word_forge_multilingual_repair_history.jsonl"
    status_payload: dict[str, Any] = {
        "contract": "eidos.word_forge.multilingual.repair.status.v1",
        "status": "running",
        "phase": "repairing",
        "db_path": str(db_path),
        "limit": limit,
        "apply": bool(apply),
        "started_at": now_iso(),
    }
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    previous_sigint = signal.getsignal(signal.SIGINT)

    def _raise_interrupted(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"signal:{signum}")

    signal.signal(signal.SIGTERM, _raise_interrupted)
    signal.signal(signal.SIGINT, _raise_interrupted)
    try:
        started_monotonic = time.monotonic()
        before = db_counts(db_path)
        backlog_before = get_multilingual_repair_backlog_snapshot(db_path)
        status_payload.update(
            {
                "word_count": int(before.get("word_count") or 0),
                "defined_word_count": int(before.get("defined_word_count") or 0),
                "relationship_count": int(before.get("relationship_count") or 0),
                "translation_word_count": int(backlog_before.get("translation_word_count") or 0),
                "repair_backlog_before": int(backlog_before.get("repair_backlog") or 0),
                "candidate_count": 0,
                "processed_count": 0,
                "repaired_count": 0,
                "current_term": "",
                "completion_ratio": 0.0,
                "batch_progress_ratio": 0.0,
                "elapsed_seconds": 0.0,
                "batch_eta_seconds": None,
            }
        )
        write_json(status_path, status_payload)
        repaired_rows: list[dict[str, Any]] = []
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            candidate_sql = """
                SELECT id, term, definition, part_of_speech, usage_examples,
                       CASE WHEN lower(COALESCE(definition, '')) LIKE '%translation linked to%' THEN 1 ELSE 0 END AS placeholder_definition,
                       CASE WHEN COALESCE(trim(part_of_speech), '') = '' THEN 1 ELSE 0 END AS missing_pos,
                       (
                         SELECT COUNT(*)
                         FROM relationships r
                         WHERE r.word_id = words.id AND r.relationship_type = 'translation'
                       ) AS translation_link_count
                FROM words
                WHERE EXISTS (
                    SELECT 1 FROM relationships r
                    WHERE r.word_id = words.id AND r.relationship_type = 'translation'
                )
                  AND (
                    COALESCE(trim(definition), '') = ''
                    OR COALESCE(trim(part_of_speech), '') = ''
                    OR lower(COALESCE(definition, '')) LIKE '%translation linked to%'
                  )
                ORDER BY placeholder_definition DESC, missing_pos DESC, translation_link_count DESC, term ASC
                """
            if limit is not None:
                candidate_sql += f" LIMIT {max(1, int(limit))}"
            candidates = conn.execute(candidate_sql).fetchall()
            candidate_count = len(candidates)
            _write_multilingual_repair_progress(
                status_path,
                status_payload,
                candidate_count=candidate_count,
                processed_count=0,
                repaired_count=0,
                current_term="",
                translation_word_count=int(backlog_before.get("translation_word_count") or 0),
                repair_backlog_before=int(backlog_before.get("repair_backlog") or 0),
                started_monotonic=started_monotonic,
            )
            for index, row in enumerate(candidates, start=1):
                repaired = _repair_translation_word_entry(conn, row)
                if not repaired:
                    if index == 1 or index == candidate_count or index % 250 == 0:
                        _write_multilingual_repair_progress(
                            status_path,
                            status_payload,
                            candidate_count=candidate_count,
                            processed_count=index,
                            repaired_count=len(repaired_rows),
                            current_term=str(row["term"] or ""),
                            translation_word_count=int(
                                backlog_before.get("translation_word_count") or 0
                            ),
                            repair_backlog_before=int(backlog_before.get("repair_backlog") or 0),
                            started_monotonic=started_monotonic,
                        )
                    continue
                repaired_rows.append(repaired)
                if apply:
                    conn.execute(
                        "UPDATE words SET definition = ?, part_of_speech = ?, last_refreshed = strftime('%s','now') WHERE term = ?",
                        (repaired["definition"], repaired["part_of_speech"], repaired["term"]),
                    )
                    if index % commit_interval == 0:
                        conn.commit()
                if index == 1 or index == candidate_count or index % 250 == 0:
                    _write_multilingual_repair_progress(
                        status_path,
                        status_payload,
                        candidate_count=candidate_count,
                        processed_count=index,
                        repaired_count=len(repaired_rows),
                        current_term=str(repaired["term"] or row["term"] or ""),
                        translation_word_count=int(
                            backlog_before.get("translation_word_count") or 0
                        ),
                        repair_backlog_before=int(backlog_before.get("repair_backlog") or 0),
                        started_monotonic=started_monotonic,
                    )
            if apply:
                conn.commit()
        after = db_counts(db_path)
        backlog_after = get_multilingual_repair_backlog_snapshot(db_path)
        report = {
            "contract": "eidos.word_forge.multilingual.repair.v1",
            "generated_at": now_iso(),
            "db_path": str(db_path),
            "limit": limit,
            "apply": bool(apply),
            "candidate_count": candidate_count,
            "repaired_count": len(repaired_rows),
            "repaired_preview": repaired_rows[:24],
            "before": before,
            "after": after,
            "backlog_before": backlog_before,
            "backlog_after": backlog_after,
        }
        artifacts = write_multilingual_repair_report(repo_root, report)
        status_payload.update(
            {
                "status": "completed",
                "phase": "completed",
                "finished_at": now_iso(),
                "report_path": artifacts["latest_json"],
                "candidate_count": candidate_count,
                "repaired_count": len(repaired_rows),
                "repair_backlog": int(backlog_after.get("repair_backlog") or 0),
            }
        )
        write_json(status_path, status_payload)
        append_history(history_path, status_payload)
        return {"status": status_payload, "report": report, "artifacts": artifacts}
    except (KeyboardInterrupt, SystemExit) as exc:
        status_payload.update(
            {
                "status": "interrupted",
                "phase": "interrupted",
                "finished_at": now_iso(),
                "error": str(exc) or exc.__class__.__name__,
            }
        )
        write_json(status_path, status_payload)
        append_history(history_path, status_payload)
        raise
    except Exception as exc:
        status_payload.update(
            {
                "status": "failed",
                "phase": "failed",
                "finished_at": now_iso(),
                "error": str(exc),
            }
        )
        write_json(status_path, status_payload)
        append_history(history_path, status_payload)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)
