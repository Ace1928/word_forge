from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from word_forge.bridge.audit import (
    _BRIDGE_STOPWORDS,
    _load_code_tokens,
    _load_file_tokens,
    _load_knowledge_tokens,
    _load_morpheme_metrics,
    _load_word_metrics,
    run_bridge_audit,
)
from word_forge.database.database_manager import DBManager, TermNotFoundError
from word_forge.forge import run_vector_index
from word_forge.linguistics.morphology import MorphologyManager
from word_forge.multilingual.fasttext_runtime import run_fasttext_ingest
from word_forge.multilingual.polyglot_runtime import run_polyglot_decomposition
from word_forge.multilingual.runtime import (
    append_history,
    now_iso,
    read_json,
    run_multilingual_ingest,
    source_signature,
    write_json,
)
from word_forge.thesaurus_runtime import run_word_forge_thesaurus

REPORT_DIR_NAME = "word_forge_growth"
STATUS_FILE_NAME = "word_forge_growth_status.json"
HISTORY_FILE_NAME = "word_forge_growth_history.jsonl"
DAEMON_STATUS_FILE_NAME = "word_forge_growth_daemon_status.json"
DAEMON_HISTORY_FILE_NAME = "word_forge_growth_daemon_history.jsonl"
DAEMON_STOP_FILE_NAME = "word_forge_growth_daemon_stop.json"
_DISCOVERY_TERM_RE = re.compile(r"^[a-z][a-z0-9-]{2,31}$")
_FORGE_LOCAL_DISCOVERY_DEFINITION_PREFIX = "A cross-forge lexical anchor discovered across the "
_FORGE_LOCAL_BLOCKLIST = {
    "abc",
    "abspath",
    "ach",
    "ache",
    "achine",
    "ackage",
    "ackend",
    "ackups",
    "added",
    "adding",
    "adjacent",
    "advanced",
    "against",
    "alias",
    "aliases",
    "all",
    "allowed",
    "allows",
    "annotate",
    "annotation",
    "any",
    "api",
    "apis",
    "app",
    "append",
    "applicable",
    "applications",
    "applies",
    "apply",
    "arg",
    "args",
    "arguments",
    "ascii",
    "cache",
    "catalog",
    "client",
    "clone",
    "code",
    "commit",
    "config",
    "context",
    "debug",
    "directory",
    "docs",
    "eval",
    "file",
    "files",
    "forge",
    "format",
    "graph",
    "html",
    "info",
    "json",
    "links",
    "linux",
    "meta",
    "openai",
    "output",
    "outputs",
    "parser",
    "phase2",
    "ports",
    "profile",
    "pytest",
    "query",
    "readme",
    "resources",
    "runit",
    "runtime",
    "schema",
    "self",
    "spec",
    "status",
    "syntax",
    "temp",
    "testing",
    "tika",
    "tools",
    "tree",
    "upgrade",
    "venv",
    "word",
    "issues",
    "ingest",
}


_DEFINITION_DISCOVERY_FUNCTION_WORDS = {
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "from",
    "into",
    "onto",
    "over",
    "under",
    "than",
    "then",
    "they",
    "them",
    "their",
    "there",
    "these",
    "those",
    "have",
    "has",
    "had",
    "been",
    "being",
    "were",
    "was",
    "are",
    "is",
    "be",
    "can",
    "could",
    "would",
    "should",
    "may",
    "might",
    "must",
    "will",
    "shall",
    "about",
    "after",
    "before",
    "because",
    "while",
    "where",
    "when",
    "which",
    "whose",
    "whom",
    "what",
    "such",
    "also",
    "other",
    "another",
    "some",
    "any",
    "each",
    "every",
    "both",
    "either",
    "neither",
    "more",
    "most",
    "less",
    "least",
    "used",
    "using",
    "use",
    "often",
    "very",
    "much",
    "many",
    "within",
    "without",
    "through",
    "across",
    "between",
    "among",
    "during",
    "including",
    "includes",
    "include",
    "consists",
    "consisting",
    "consist",
    "type",
    "types",
    "kind",
    "kinds",
    "form",
    "forms",
    "made",
    "make",
    "makes",
    "like",
    "same",
    "different",
    "something",
    "someone",
    "anything",
    "nothing",
    "everything",
}

_DEFINITION_DISCOVERY_EXCLUDE = {
    "ita",
    "spa",
    "fra",
    "eng",
    "deu",
    "lat",
    "por",
    "rus",
    "jpn",
    "zho",
    "kor",
    "nld",
    "swe",
    "dan",
    "nor",
    "fin",
    "pol",
    "ces",
    "slk",
    "ron",
    "hun",
}


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


def _discover_multilingual_sources(repo_root: Path) -> list[dict[str, Any]]:
    candidates: list[Path] = []
    search_roots = [repo_root / "word_forge" / "data", repo_root / "data" / "word_forge"]
    for root in search_roots:
        if not root.exists():
            continue
        candidates.extend(sorted(root.rglob("*wiktextract*.jsonl")))
        candidates.extend(sorted(root.rglob("*kaikki*.jsonl")))
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in candidates:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        name = path.name.lower()
        source_type = "kaikki" if "kaikki" in name else "wiktextract"
        deduped.append(
            {
                "kind": "multilingual",
                "source_type": source_type,
                "path": resolved,
                "is_sample": "sample" in name,
            }
        )
    deduped.sort(key=lambda row: (row["is_sample"], row["path"]))
    return deduped


def _lang_from_vec(path: Path) -> str:
    stem = path.stem.lower()
    for sep in ("_", "-", "."):
        parts = [part for part in stem.split(sep) if part]
        if parts:
            tail = parts[-1]
            if 2 <= len(tail) <= 5 and tail.isalpha():
                return tail
    return "en"


def _discover_fasttext_sources(repo_root: Path) -> list[dict[str, Any]]:
    candidates: list[Path] = []
    search_roots = [repo_root / "word_forge" / "data", repo_root / "data" / "word_forge"]
    for root in search_roots:
        if not root.exists():
            continue
        candidates.extend(sorted(root.rglob("*.vec")))
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in candidates:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(
            {
                "kind": "fasttext",
                "lang": _lang_from_vec(path),
                "path": resolved,
                "is_sample": "sample" in path.name.lower(),
            }
        )
    deduped.sort(key=lambda row: (row["is_sample"], row["path"]))
    return deduped


def _discover_forge_local_sources(repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_type, path in (
        ("knowledge", repo_root / "data" / "kb.json"),
        ("file", repo_root / "data" / "file_forge" / "library.sqlite"),
        ("code", repo_root / "data" / "code_forge" / "library.sqlite"),
        ("docs", repo_root / "docs"),
        ("prompts", repo_root / "prompts"),
        ("proof", repo_root / "reports" / "proof" / "entity_proof_scorecard_latest.json"),
        ("bridge", repo_root / "reports" / "word_forge_bridge_audit" / "latest.json"),
    ):
        if not path.exists():
            continue
        rows.append(
            {
                "kind": "forge_local",
                "source_type": source_type,
                "path": str(path.resolve()),
                "lang": "",
                "is_sample": False,
            }
        )
    return rows


def _load_text_corpus_tokens(
    paths: list[Path], *, max_files: int = 32, max_bytes: int = 40000
) -> set[str]:
    tokens: set[str] = set()
    seen_paths: set[str] = set()
    selected: list[Path] = []
    for path in paths:
        resolved = str(path.resolve())
        if resolved in seen_paths or not path.exists():
            continue
        seen_paths.add(resolved)
        selected.append(path)
        if len(selected) >= max_files:
            break
    for path in selected:
        try:
            if path.is_dir():
                for child in sorted(path.rglob("*")):
                    if not child.is_file() or child.suffix.lower() not in {
                        ".md",
                        ".txt",
                        ".json",
                        ".jsonl",
                        ".py",
                        ".toml",
                    }:
                        continue
                    selected.append(child)
                    if len(selected) >= max_files:
                        break
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")[:max_bytes]
        except Exception:
            continue
        for raw in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,31}", text):
            for token in re.split(r"[_\-]+", raw.lower()):
                if token:
                    tokens.add(token)
    return {token for token in tokens if _valid_discovery_term(token)}


def _discover_repo_domain_tokens(repo_root: Path) -> set[str]:
    tokens: set[str] = set()
    candidate_paths = [
        repo_root / "atlas_forge",
        repo_root / "word_forge",
        repo_root / "file_forge",
        repo_root / "code_forge",
        repo_root / "archive_forge",
        repo_root / "memory_forge",
        repo_root / "knowledge_forge",
        repo_root / "doc_forge",
        repo_root / "neural_forge",
        repo_root / "scripts",
        repo_root / "docs",
    ]
    for path in candidate_paths:
        if not path.exists():
            continue
        tokens.update(token for token in re.split(r"[_\-]+", path.name.lower()) if token)
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if not child.name or child.name.startswith("."):
                    continue
                tokens.update(
                    token for token in re.split(r"[_\-.]+", child.stem.lower()) if len(token) >= 4
                )
    return tokens


def _valid_discovery_term(term: str) -> bool:
    normalized = str(term or "").strip().lower()
    if not normalized or normalized in _BRIDGE_STOPWORDS:
        return False
    if normalized in _FORGE_LOCAL_BLOCKLIST:
        return False
    if not _DISCOVERY_TERM_RE.fullmatch(normalized):
        return False
    if normalized.isdigit():
        return False
    if len(normalized) < 4:
        return False
    if normalized.count("-") > 1:
        return False
    return any(char.isalpha() for char in normalized)


def _rank_forge_local_candidates(
    *,
    existing_terms: set[str],
    knowledge_tag_tokens: set[str],
    file_path_tokens: set[str],
    code_tokens: set[str],
    morpheme_anchor_terms: set[str],
    repo_domain_tokens: set[str],
    docs_tokens: set[str],
    prompt_tokens: set[str],
    proof_tokens: set[str],
    limit: int = 48,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    universe = (
        set(knowledge_tag_tokens)
        | set(file_path_tokens)
        | set(code_tokens)
        | set(morpheme_anchor_terms)
        | set(repo_domain_tokens)
        | set(docs_tokens)
        | set(prompt_tokens)
        | set(proof_tokens)
    )
    for term in sorted(universe):
        normalized = str(term or "").strip().lower()
        if normalized in existing_terms or not _valid_discovery_term(normalized):
            continue
        sources = {
            "knowledge": normalized in knowledge_tag_tokens,
            "file": normalized in file_path_tokens,
            "code": normalized in code_tokens,
            "morpheme": normalized in morpheme_anchor_terms,
            "repo": normalized in repo_domain_tokens,
            "docs": normalized in docs_tokens,
            "prompt": normalized in prompt_tokens,
            "proof": normalized in proof_tokens,
        }
        support_count = sum(int(value) for value in sources.values())
        if support_count < 2:
            continue
        if not (
            sources["code"]
            or sources["file"]
            or sources["repo"]
            or sources["docs"]
            or sources["prompt"]
            or sources["proof"]
        ):
            continue
        if not (
            sources["repo"]
            or sources["morpheme"]
            or sources["docs"]
            or sources["proof"]
            or (sources["knowledge"] and sources["code"] and sources["file"])
        ):
            continue
        score = support_count * 10
        if sources["code"] and sources["file"]:
            score += 5
        if sources["knowledge"] and sources["morpheme"]:
            score += 3
        if sources["repo"]:
            score += 4
        if sources["docs"]:
            score += 4
        if sources["proof"]:
            score += 3
        if sources["prompt"]:
            score += 2
        if sources["morpheme"]:
            score += 2
        score -= min(len(normalized), 12)
        ranked.append(
            {
                "term": normalized,
                "support_count": support_count,
                "score": score,
                "sources": [name for name, enabled in sources.items() if enabled],
            }
        )
    ranked.sort(key=lambda row: (-int(row["score"]), -int(row["support_count"]), str(row["term"])))
    return ranked[: max(1, int(limit))]


def _forge_local_definition(candidate: dict[str, Any]) -> str:
    sources = list(candidate.get("sources") or [])
    support = ", ".join(sources) if sources else "forge-local"
    term = str(candidate.get("term") or "").strip()
    return f"{_FORGE_LOCAL_DISCOVERY_DEFINITION_PREFIX}{support} substrates of the Eidosian system for the term '{term}'."


def _forge_local_examples(candidate: dict[str, Any]) -> list[str]:
    term = str(candidate.get("term") or "").strip()
    sources = ", ".join(list(candidate.get("sources") or [])) or "forge-local"
    return [
        f"The Eidosian system uses {term} as a {sources} anchor during governed lexicon discovery and graph expansion."
    ]


def _semantic_vector_count(repo_root: Path) -> int:
    sqlite_path = repo_root / "word_forge" / "data" / "vector_indices" / "vector_store.sqlite3"
    if not sqlite_path.exists():
        return 0
    try:
        with sqlite3.connect(str(sqlite_path)) as conn:
            row = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()
        return int(row[0] or 0) if row else 0
    except Exception:
        return 0


def cleanup_forge_local_discovery_terms(*, repo_root: Path, db_path: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    db_path = db_path.resolve()
    vector_db_path = repo_root / "word_forge" / "data" / "vector_indices" / "vector_store.sqlite3"
    hnsw_index_path = repo_root / "word_forge" / "data" / "vector_indices" / "vector_store.hnsw.bin"

    deleted_word_ids: list[int] = []
    deleted_terms: list[str] = []
    if db_path.exists():
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute(
                "SELECT id, term FROM words WHERE definition LIKE ? ORDER BY id",
                (f"{_FORGE_LOCAL_DISCOVERY_DEFINITION_PREFIX}%",),
            ).fetchall()
            deleted_word_ids = [int(row[0]) for row in rows]
            deleted_terms = [str(row[1]).strip().lower() for row in rows if str(row[1]).strip()]
            if deleted_word_ids:
                placeholders = ",".join("?" for _ in deleted_word_ids)
                for table, column in (
                    ("relationships", "word_id"),
                    ("phonetics", "word_id"),
                    ("word_morphemes", "word_id"),
                    ("emotional_relationships", "word_id"),
                    ("word_emotion", "word_id"),
                ):
                    try:
                        conn.execute(
                            f"DELETE FROM {table} WHERE {column} IN ({placeholders})",
                            deleted_word_ids,
                        )
                    except sqlite3.OperationalError:
                        continue
                conn.execute(f"DELETE FROM words WHERE id IN ({placeholders})", deleted_word_ids)
                conn.commit()

    deleted_vector_count = 0
    if vector_db_path.exists() and deleted_terms:
        with sqlite3.connect(str(vector_db_path)) as conn:
            rows = conn.execute("SELECT id, metadata FROM vectors").fetchall()
            delete_ids: list[str] = []
            deleted_id_set = set(deleted_word_ids)
            deleted_term_set = set(deleted_terms)
            for vec_id, metadata_blob in rows:
                metadata: dict[str, Any] = {}
                if metadata_blob:
                    try:
                        metadata = json.loads(str(metadata_blob))
                    except Exception:
                        metadata = {}
                original_id = metadata.get("original_id")
                term = str(metadata.get("term") or "").strip().lower()
                if (isinstance(original_id, int) and original_id in deleted_id_set) or (
                    term and term in deleted_term_set
                ):
                    delete_ids.append(str(vec_id))
            if delete_ids:
                conn.executemany(
                    "DELETE FROM vectors WHERE id = ?", [(item,) for item in delete_ids]
                )
                conn.commit()
                deleted_vector_count = len(delete_ids)
    if deleted_vector_count and hnsw_index_path.exists():
        hnsw_index_path.unlink()

    return {
        "contract": "eidos.word_forge.forge_local_cleanup.v1",
        "status": "completed",
        "deleted_word_count": len(deleted_word_ids),
        "deleted_vector_count": deleted_vector_count,
        "deleted_terms": deleted_terms[:64],
    }


def request_word_forge_growth_daemon_stop(*, repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    stop_path = _daemon_stop_path(repo_root)
    payload = {
        "contract": "eidos.word_forge.growth_daemon.stop_request.v1",
        "status": "requested",
        "requested_at": now_iso(),
        "stop_path": str(stop_path),
    }
    write_json(stop_path, payload)
    return payload


def run_word_forge_growth_daemon(
    *,
    repo_root: Path,
    db_path: Path,
    vector_db_path: Path,
    interval_seconds: int = 900,
    forge_local_limit: int | None = None,
    multilingual_limit: int | None = None,
    fasttext_limit: int | None = None,
    polyglot_limit: int | None = None,
    lexicon_limit: int | None = None,
    thesaurus_limit: int | None = None,
    phonetic_limit: int | None = None,
    thesaurus_terms: list[str] | None = None,
    cycles: int | None = None,
    force_first_cycle: bool = False,
    target_word_count: int | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    db_path = db_path.resolve()
    vector_db_path = vector_db_path.resolve()
    interval_seconds = max(30, int(interval_seconds))
    daemon_status_path = _daemon_status_path(repo_root)
    daemon_history_path = _daemon_history_path(repo_root)
    stop_path = _daemon_stop_path(repo_root)
    if stop_path.exists():
        stop_path.unlink()

    pid = int(__import__("os").getpid())
    cycle_count = 0
    started_at = now_iso()
    initial_snapshot = _lexicon_progress_snapshot(db_path)
    status_payload: dict[str, Any] = {
        "contract": "eidos.word_forge.growth_daemon.status.v1",
        "status": "running",
        "phase": "starting",
        "pid": pid,
        "repo_root": str(repo_root),
        "db_path": str(db_path),
        "vector_db_path": str(vector_db_path),
        "interval_seconds": interval_seconds,
        "forge_local_limit": forge_local_limit,
        "multilingual_limit": multilingual_limit,
        "fasttext_limit": fasttext_limit,
        "polyglot_limit": polyglot_limit,
        "lexicon_limit": lexicon_limit,
        "thesaurus_limit": thesaurus_limit,
        "phonetic_limit": phonetic_limit,
        "thesaurus_terms": list(thesaurus_terms or []),
        "cycles_requested": cycles,
        "target_word_count": int(target_word_count or 0) if target_word_count is not None else None,
        "cycle_count": 0,
        "started_at": started_at,
        "progress": {
            "initial": initial_snapshot,
            "current": initial_snapshot,
            "words_added": 0,
            "target_progress_ratio": (
                min(
                    1.0,
                    (initial_snapshot.get("word_count", 0) / max(1, int(target_word_count or 0))),
                )
                if target_word_count
                else None
            ),
            "words_remaining": (
                max(0, int(target_word_count or 0) - int(initial_snapshot.get("word_count", 0)))
                if target_word_count
                else None
            ),
            "eta_seconds": None,
        },
    }
    write_json(daemon_status_path, status_payload)
    append_history(daemon_history_path, status_payload)

    try:
        while True:
            if stop_path.exists():
                stop_request = read_json(stop_path)
                status_payload.update(
                    {
                        "status": "stopped",
                        "phase": "stopped",
                        "finished_at": now_iso(),
                        "stop_request": stop_request,
                    }
                )
                write_json(daemon_status_path, status_payload)
                append_history(daemon_history_path, status_payload)
                return status_payload

            if cycles is not None and cycle_count >= max(1, int(cycles)):
                status_payload.update(
                    {
                        "status": "completed",
                        "phase": "completed",
                        "finished_at": now_iso(),
                    }
                )
                write_json(daemon_status_path, status_payload)
                append_history(daemon_history_path, status_payload)
                return status_payload

            current_snapshot = _lexicon_progress_snapshot(db_path)
            if target_word_count is not None and current_snapshot.get("word_count", 0) >= max(
                1, int(target_word_count)
            ):
                status_payload.update(
                    {
                        "status": "completed",
                        "phase": "target_reached",
                        "finished_at": now_iso(),
                        "progress": {
                            **(status_payload.get("progress") or {}),
                            "current": current_snapshot,
                            "words_added": max(
                                0,
                                current_snapshot.get("word_count", 0)
                                - initial_snapshot.get("word_count", 0),
                            ),
                            "target_progress_ratio": 1.0,
                            "words_remaining": 0,
                            "eta_seconds": 0,
                        },
                    }
                )
                write_json(daemon_status_path, status_payload)
                append_history(daemon_history_path, status_payload)
                return status_payload

            cycle_count += 1
            status_payload.update(
                {
                    "phase": "growth_cycle",
                    "cycle_count": cycle_count,
                    "cycle_started_at": now_iso(),
                }
            )
            write_json(daemon_status_path, status_payload)

            try:
                growth_result = run_word_forge_growth(
                    repo_root=repo_root,
                    db_path=db_path,
                    vector_db_path=vector_db_path,
                    force=bool(force_first_cycle and cycle_count == 1),
                    multilingual_limit=multilingual_limit,
                    fasttext_limit=fasttext_limit,
                    polyglot_limit=polyglot_limit,
                    forge_local_limit=forge_local_limit,
                    lexicon_limit=lexicon_limit,
                    thesaurus_limit=thesaurus_limit,
                    phonetic_limit=phonetic_limit,
                    thesaurus_terms=thesaurus_terms,
                )
                growth_status = (
                    growth_result.get("status") if isinstance(growth_result, dict) else {}
                )
                status_payload.update(
                    {
                        "last_growth_status": growth_status,
                        "last_growth_finished_at": str(
                            (growth_status or {}).get("finished_at") or now_iso()
                        ),
                        "last_error": "",
                    }
                )
                current_snapshot = _lexicon_progress_snapshot(db_path)
                words_added = max(
                    0, current_snapshot.get("word_count", 0) - initial_snapshot.get("word_count", 0)
                )
                average_words_per_cycle = (words_added / cycle_count) if cycle_count > 0 else 0.0
                words_remaining = None
                target_progress_ratio = None
                eta_seconds = None
                if target_word_count is not None:
                    target_word_count = max(1, int(target_word_count))
                    words_remaining = max(
                        0, target_word_count - current_snapshot.get("word_count", 0)
                    )
                    target_progress_ratio = min(
                        1.0, current_snapshot.get("word_count", 0) / target_word_count
                    )
                    if words_remaining <= 0:
                        eta_seconds = 0
                    elif average_words_per_cycle > 0:
                        eta_seconds = int(
                            (words_remaining / average_words_per_cycle) * interval_seconds
                        )
                last_report = growth_result.get("report") if isinstance(growth_result, dict) else {}
                status_payload["progress"] = {
                    "initial": initial_snapshot,
                    "current": current_snapshot,
                    "words_added": words_added,
                    "average_words_per_cycle": round(average_words_per_cycle, 3),
                    "target_progress_ratio": target_progress_ratio,
                    "words_remaining": words_remaining,
                    "eta_seconds": eta_seconds,
                    "last_cycle": {
                        "source_count": (
                            int((last_report or {}).get("discovery", {}).get("source_count", 0))
                            if isinstance((last_report or {}).get("discovery"), dict)
                            else 0
                        ),
                        "forge_local_term_delta": int(
                            (last_report or {}).get("forge_local_term_delta") or 0
                        ),
                        "lexicon_term_delta": int(
                            (last_report or {}).get("lexicon_term_delta") or 0
                        ),
                        "relationship_delta": int(
                            (last_report or {}).get("relationship_delta") or 0
                        ),
                        "semantic_vector_delta": int(
                            (last_report or {}).get("semantic_vector_delta") or 0
                        ),
                        "thesaurus_seed_term_count": int(
                            (last_report or {}).get("thesaurus_seed_term_count") or 0
                        ),
                        "multilingual_seed_term_count": int(
                            (last_report or {}).get("multilingual_seed_term_count") or 0
                        ),
                        "incomplete_seed_term_count": int(
                            (last_report or {}).get("incomplete_seed_term_count") or 0
                        ),
                        "definition_seed_term_count": int(
                            (last_report or {}).get("definition_seed_term_count") or 0
                        ),
                        "usage_seed_term_count": int(
                            (last_report or {}).get("usage_seed_term_count") or 0
                        ),
                        "emotion_seed_term_count": int(
                            (last_report or {}).get("emotion_seed_term_count") or 0
                        ),
                        "derivation_seed_term_count": int(
                            (last_report or {}).get("derivation_seed_term_count") or 0
                        ),
                        "priority_queue_count": len(
                            (last_report or {}).get("priority_queue") or []
                        ),
                    },
                }
            except Exception as exc:
                status_payload.update(
                    {
                        "last_growth_status": {"status": "failed", "error": str(exc)},
                        "last_growth_finished_at": now_iso(),
                        "last_error": str(exc),
                    }
                )

            append_history(
                daemon_history_path,
                {
                    "contract": "eidos.word_forge.growth_daemon.cycle.v1",
                    "status": status_payload.get("status", "running"),
                    "phase": status_payload.get("phase", "growth_cycle"),
                    "pid": pid,
                    "cycle_count": cycle_count,
                    "cycle_started_at": status_payload.get("cycle_started_at"),
                    "last_growth_finished_at": status_payload.get("last_growth_finished_at"),
                    "last_growth_status": status_payload.get("last_growth_status"),
                    "last_error": status_payload.get("last_error", ""),
                },
            )

            status_payload.update({"phase": "sleeping", "sleep_started_at": now_iso()})
            write_json(daemon_status_path, status_payload)
            remaining = interval_seconds
            while remaining > 0:
                if stop_path.exists():
                    break
                sleep_for = 5 if remaining > 5 else remaining
                __import__("time").sleep(sleep_for)
                remaining -= sleep_for
                status_payload["sleep_remaining_seconds"] = remaining
                write_json(daemon_status_path, status_payload)
    except Exception as exc:
        status_payload.update(
            {
                "status": "failed",
                "phase": "failed",
                "finished_at": now_iso(),
                "last_error": str(exc),
            }
        )
        write_json(daemon_status_path, status_payload)
        append_history(daemon_history_path, status_payload)
        raise


def run_forge_local_discovery(
    *, repo_root: Path, db_path: Path, limit: int | None = None, apply: bool = True
) -> dict[str, Any]:
    word_metrics = _load_word_metrics(db_path)
    knowledge_metrics = _load_knowledge_tokens(repo_root / "data" / "kb.json")
    file_metrics = _load_file_tokens(repo_root)
    code_metrics = _load_code_tokens(repo_root)
    morpheme_metrics = _load_morpheme_metrics(db_path)
    repo_domain_tokens = _discover_repo_domain_tokens(repo_root)
    docs_tokens = _load_text_corpus_tokens([repo_root / "docs"], max_files=24)
    prompt_tokens = _load_text_corpus_tokens(
        [repo_root / "prompts", repo_root / "prompt_forge"], max_files=20
    )
    proof_tokens = _load_text_corpus_tokens(
        [
            repo_root / "reports" / "proof" / "entity_proof_scorecard_latest.json",
            repo_root / "reports" / "proof_bundle" / "latest_manifest.json",
            repo_root / "reports" / "word_forge_bridge_audit" / "latest.json",
        ],
        max_files=12,
    )

    candidates = _rank_forge_local_candidates(
        existing_terms=set(word_metrics.get("word_terms", set())),
        knowledge_tag_tokens=set(knowledge_metrics.get("tag_tokens", set())),
        file_path_tokens=set(file_metrics.get("path_tokens", set())),
        code_tokens=set(code_metrics.get("code_tokens", set())),
        morpheme_anchor_terms=set(
            (morpheme_metrics.get("informative_anchor_morphemes") or {}).keys()
        ),
        repo_domain_tokens=repo_domain_tokens,
        docs_tokens=docs_tokens,
        prompt_tokens=prompt_tokens,
        proof_tokens=proof_tokens,
        limit=limit or 48,
    )

    inserted_terms: list[str] = []
    if apply:
        db = DBManager(db_path=db_path)
        try:
            db.create_tables()
            for row in candidates:
                term = str(row.get("term") or "").strip()
                if not term:
                    continue
                try:
                    if db.get_word_id(term) > 0:
                        continue
                except TermNotFoundError:
                    pass
                db.insert_or_update_word(
                    term=term,
                    definition=_forge_local_definition(row),
                    part_of_speech="noun",
                    usage_examples=_forge_local_examples(row),
                )
                inserted_terms.append(term)
        finally:
            db.close()

    return {
        "contract": "eidos.word_forge.forge_local_discovery.v1",
        "status": "completed" if apply else "preview",
        "apply": apply,
        "candidate_count": len(candidates),
        "inserted_count": len(inserted_terms),
        "inserted_terms": inserted_terms[:24],
        "candidates": candidates[:24],
        "repo_domain_token_count": len(repo_domain_tokens),
        "docs_token_count": len(docs_tokens),
        "prompt_token_count": len(prompt_tokens),
        "proof_token_count": len(proof_tokens),
    }


def discover_growth_sources(repo_root: Path) -> dict[str, Any]:
    multilingual_sources = _discover_multilingual_sources(repo_root)
    fasttext_sources = _discover_fasttext_sources(repo_root)
    forge_local_sources = _discover_forge_local_sources(repo_root)
    corpus_local_source_count = sum(
        1
        for row in forge_local_sources
        if str(row.get("source_type") or "") in {"docs", "prompts", "proof", "bridge"}
    )
    return {
        "multilingual_sources": multilingual_sources,
        "fasttext_sources": fasttext_sources,
        "forge_local_sources": forge_local_sources,
        "multilingual_source_count": len(multilingual_sources),
        "fasttext_source_count": len(fasttext_sources),
        "forge_local_source_count": len(forge_local_sources),
        "corpus_local_source_count": corpus_local_source_count,
        "source_count": len(multilingual_sources)
        + len(fasttext_sources)
        + len(forge_local_sources),
    }


def _valid_lexicon_term(term: str) -> bool:
    normalized = str(term or "").strip().lower()
    if not normalized or len(normalized) > 64:
        return False
    if normalized.isdigit() or not any(char.isalpha() for char in normalized):
        return False
    if len(normalized) == 1:
        return normalized.isalpha()
    if any(char in normalized for char in "\n\r\t/:{}[]()<>|"):
        return False
    if normalized.count(" ") > 0:
        return False
    return True


def _coerce_usage_examples(existing_examples: list[str], new_examples: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for item in list(existing_examples) + list(new_examples):
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        merged.append(text)
    return merged[:6]


def _synthesise_growth_thesaurus_terms(
    *,
    explicit_terms: list[str] | None,
    forge_local_run: dict[str, Any] | None,
    lexicon_backfill_run: dict[str, Any] | None,
    multilingual_seed_terms: list[str] | None = None,
    incomplete_seed_terms: list[str] | None = None,
    definition_seed_terms: list[str] | None = None,
    usage_seed_terms: list[str] | None = None,
    vector_seed_terms: list[str] | None = None,
    emotion_seed_terms: list[str] | None = None,
    derivation_seed_terms: list[str] | None = None,
    limit: int = 24,
) -> list[str]:
    bounded_limit = max(4, min(int(limit or 24), 64))
    ordered: list[str] = []
    seen: set[str] = set()

    def _push(value: str) -> None:
        normalized = str(value or "").strip().lower()
        if not _valid_lexicon_term(normalized) or normalized in seen:
            return
        seen.add(normalized)
        ordered.append(normalized)

    for term in list(explicit_terms or []):
        _push(term)
    forge_local_payload = forge_local_run if isinstance(forge_local_run, dict) else {}
    for term in forge_local_payload.get("inserted_terms") or []:
        _push(str(term))
    for row in forge_local_payload.get("candidates") or []:
        if isinstance(row, dict):
            _push(str(row.get("term") or ""))
    for term in (lexicon_backfill_run or {}).get("terms") or []:
        _push(str(term))
    for term in list(multilingual_seed_terms or []):
        _push(term)
    for term in list(incomplete_seed_terms or []):
        _push(term)
    for term in list(definition_seed_terms or []):
        _push(term)
    for term in list(usage_seed_terms or []):
        _push(term)
    for term in list(vector_seed_terms or []):
        _push(term)
    for term in list(emotion_seed_terms or []):
        _push(term)
    for term in list(derivation_seed_terms or []):
        _push(term)
    return ordered[:bounded_limit]


def _harvest_definition_seed_rows(db_path: Path, *, limit: int = 48) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    bounded_limit = max(8, min(int(limit or 48), 256))
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        word_rows = conn.execute("""
            SELECT lower(term) AS term, COALESCE(definition, '') AS definition
            FROM words
            WHERE COALESCE(trim(definition), '') <> ''
            """).fetchall()
    existing_terms = {
        str(row["term"] or "").strip().lower()
        for row in word_rows
        if str(row["term"] or "").strip()
    }
    mentions: dict[str, dict[str, Any]] = {}
    for row in word_rows:
        source_term = str(row["term"] or "").strip().lower()
        definition = str(row["definition"] or "")
        if not source_term or not definition:
            continue
        source_seen: set[str] = set()
        for raw in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,31}", definition):
            for token in re.split(r"[_\-]+", raw.lower()):
                normalized = token.strip().lower()
                if not _valid_lexicon_term(normalized):
                    continue
                if normalized in _DEFINITION_DISCOVERY_EXCLUDE:
                    continue
                if normalized in existing_terms or normalized == source_term:
                    continue
                payload = mentions.setdefault(
                    normalized,
                    {
                        "term": normalized,
                        "mention_count": 0,
                        "source_term_count": 0,
                        "is_function_word": normalized in _DEFINITION_DISCOVERY_FUNCTION_WORDS,
                        "source_terms": [],
                    },
                )
                payload["mention_count"] += 1
                if normalized not in source_seen:
                    source_seen.add(normalized)
                    payload["source_term_count"] += 1
                    if (
                        len(payload["source_terms"]) < 6
                        and source_term not in payload["source_terms"]
                    ):
                        payload["source_terms"].append(source_term)
    rows = list(mentions.values())
    for row in rows:
        weight = 0.15 if bool(row.get("is_function_word")) else 1.0
        row["priority_score"] = round(
            (int(row.get("source_term_count") or 0) * 10 + int(row.get("mention_count") or 0))
            * weight,
            3,
        )
    rows.sort(
        key=lambda row: (
            -float(row.get("priority_score") or 0),
            -int(row.get("source_term_count") or 0),
            -int(row.get("mention_count") or 0),
            str(row.get("term") or ""),
        )
    )
    return rows[:bounded_limit]


def _harvest_usage_seed_rows(db_path: Path, *, limit: int = 48) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    bounded_limit = max(8, min(int(limit or 48), 256))
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        word_rows = conn.execute("""
            SELECT lower(term) AS term, COALESCE(usage_examples, '') AS usage_examples
            FROM words
            WHERE COALESCE(trim(usage_examples), '') <> ''
            """).fetchall()
    existing_terms = {
        str(row["term"] or "").strip().lower()
        for row in word_rows
        if str(row["term"] or "").strip()
    }
    mentions: dict[str, dict[str, Any]] = {}
    for row in word_rows:
        source_term = str(row["term"] or "").strip().lower()
        usage_examples = str(row["usage_examples"] or "")
        if not source_term or not usage_examples:
            continue
        source_seen: set[str] = set()
        for raw in re.findall(r"[A-Za-z][A-Za-z0-9_-]{0,31}", usage_examples):
            for token in re.split(r"[_\-]+", raw.lower()):
                normalized = token.strip().lower()
                if not _valid_lexicon_term(normalized):
                    continue
                if normalized in _DEFINITION_DISCOVERY_EXCLUDE:
                    continue
                if normalized in existing_terms or normalized == source_term:
                    continue
                payload = mentions.setdefault(
                    normalized,
                    {
                        "term": normalized,
                        "mention_count": 0,
                        "source_term_count": 0,
                        "is_function_word": normalized in _DEFINITION_DISCOVERY_FUNCTION_WORDS,
                        "source_terms": [],
                    },
                )
                payload["mention_count"] += 1
                if normalized not in source_seen:
                    source_seen.add(normalized)
                    payload["source_term_count"] += 1
                    if (
                        len(payload["source_terms"]) < 6
                        and source_term not in payload["source_terms"]
                    ):
                        payload["source_terms"].append(source_term)
    rows = list(mentions.values())
    for row in rows:
        weight = 0.2 if bool(row.get("is_function_word")) else 1.0
        row["priority_score"] = round(
            (int(row.get("source_term_count") or 0) * 12 + int(row.get("mention_count") or 0))
            * weight,
            3,
        )
    rows.sort(
        key=lambda row: (
            -float(row.get("priority_score") or 0),
            -int(row.get("source_term_count") or 0),
            -int(row.get("mention_count") or 0),
            str(row.get("term") or ""),
        )
    )
    return rows[:bounded_limit]


def _harvest_vector_seed_rows(
    repo_root: Path, db_path: Path, *, limit: int = 24, per_term_neighbors: int = 3
) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    bounded_limit = max(4, min(int(limit or 24), 96))
    bounded_neighbors = max(1, min(int(per_term_neighbors or 3), 8))
    try:
        from word_forge.configs.config_essentials import StorageType
        from word_forge.vectorizer.vector_store import VectorStore
    except Exception:
        return []

    incomplete_rows = _harvest_incomplete_word_rows(db_path, limit=max(bounded_limit, 24))
    definition_rows = _harvest_definition_seed_rows(db_path, limit=max(bounded_limit, 24))
    usage_rows = _harvest_usage_seed_rows(db_path, limit=max(bounded_limit, 24))
    emotion_rows = _harvest_emotion_seed_rows(db_path, limit=max(12, bounded_limit // 2))
    derivation_rows = _harvest_derivation_seed_rows(db_path, limit=max(12, bounded_limit // 2))

    candidate_terms: list[str] = []
    seen_candidate_terms: set[str] = set()
    for row_group in (incomplete_rows, definition_rows, usage_rows, emotion_rows, derivation_rows):
        for row in row_group:
            if not isinstance(row, dict):
                continue
            term = str(row.get("term") or "").strip().lower()
            if not _valid_lexicon_term(term) or term in seen_candidate_terms:
                continue
            seen_candidate_terms.add(term)
            candidate_terms.append(term)
            if len(candidate_terms) >= bounded_limit:
                break
        if len(candidate_terms) >= bounded_limit:
            break
    if not candidate_terms:
        return []

    def _placeholder_definition(text: str) -> bool:
        normalized = str(text or "").strip().lower()
        if not normalized:
            return True
        return (
            " lexical translation aligned to '" in normalized
            or normalized.startswith(_FORGE_LOCAL_DISCOVERY_DEFINITION_PREFIX.strip().lower())
            or normalized.startswith("emotion anchor node representing '")
        )

    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        existing_rows = conn.execute("""
            SELECT
                lower(term) AS term,
                COALESCE(definition, '') AS definition,
                COALESCE(part_of_speech, '') AS part_of_speech
            FROM words
            """).fetchall()
        existing_meta = {
            str(row["term"] or "").strip().lower(): {
                "definition": str(row["definition"] or ""),
                "part_of_speech": str(row["part_of_speech"] or ""),
            }
            for row in existing_rows
            if str(row["term"] or "").strip()
        }
        existing_terms = set(existing_meta)
        translation_rows = conn.execute("""
                SELECT lower(w.term) AS term, COUNT(*) AS translation_count
                FROM relationships r
                JOIN words w ON w.id = r.word_id
                WHERE lower(r.relationship_type) = 'translation'
                GROUP BY w.id
                """).fetchall()
        translation_counts = {
            str(row["term"] or "").strip().lower(): int(row["translation_count"] or 0)
            for row in translation_rows
            if str(row["term"] or "").strip()
        }
        language_counts: dict[str, int] = {}
        language_rows = conn.execute("""
            WITH lexeme_terms AS (
                SELECT lower(lemma) AS term, lower(lang) AS lang
                FROM lexemes
                WHERE COALESCE(lemma, '') <> '' AND COALESCE(lang, '') <> ''
                UNION ALL
                SELECT lower(base_term) AS term, lower(lang) AS lang
                FROM lexemes
                WHERE COALESCE(base_term, '') <> '' AND COALESCE(lang, '') <> ''
                UNION ALL
                SELECT lower(target_term) AS term, lower(target_lang) AS lang
                FROM translations
                WHERE COALESCE(target_term, '') <> '' AND COALESCE(target_lang, '') <> ''
            )
            SELECT term, COUNT(DISTINCT lang) AS language_count
            FROM lexeme_terms
            WHERE COALESCE(term, '') <> ''
            GROUP BY term
            """).fetchall()
        for row in language_rows:
            term = str(row["term"] or "").strip().lower()
            if not term:
                continue
            language_counts[term] = int(row["language_count"] or 0)
    weak_terms = {
        str(row.get("term") or "").strip().lower()
        for row_group in (
            incomplete_rows,
            definition_rows,
            usage_rows,
            emotion_rows,
            derivation_rows,
        )
        for row in row_group
        if isinstance(row, dict) and _valid_lexicon_term(str(row.get("term") or "").strip().lower())
    }
    for term, meta in existing_meta.items():
        if not term:
            continue
        if not str(meta.get("part_of_speech") or "").strip() or _placeholder_definition(
            str(meta.get("definition") or "")
        ):
            weak_terms.add(term)
    weak_terms = {term for term in weak_terms if term in existing_terms}

    mentions: dict[str, dict[str, Any]] = {}
    vector_store = VectorStore(storage_type=StorageType.DISK)
    for source_term in candidate_terms:
        source_seen: set[str] = set()
        try:
            raw_results = vector_store.search(
                query_text=source_term,
                k=bounded_neighbors + 4,
                filter_metadata={"content_type": "word"},
            )
        except Exception:
            continue
        for result_row in raw_results:
            metadata = result_row.get("metadata") or {}
            candidate_term = str(metadata.get("term") or "").strip().lower()
            if not _valid_lexicon_term(candidate_term):
                continue
            if candidate_term == source_term:
                continue
            if candidate_term in source_seen:
                continue
            if candidate_term in existing_terms and candidate_term not in weak_terms:
                continue
            source_seen.add(candidate_term)
            payload = mentions.setdefault(
                candidate_term,
                {
                    "term": candidate_term,
                    "source_term_count": 0,
                    "semantic_neighbor_count": 0,
                    "max_score": 0.0,
                    "source_terms": [],
                    "existing_weak_term": candidate_term in weak_terms,
                    "missing_term": candidate_term not in existing_terms,
                    "missing_part_of_speech": not bool(
                        str(
                            (existing_meta.get(candidate_term) or {}).get("part_of_speech") or ""
                        ).strip()
                    ),
                    "placeholder_definition": _placeholder_definition(
                        str((existing_meta.get(candidate_term) or {}).get("definition") or "")
                    ),
                    "translation_links": int(translation_counts.get(candidate_term) or 0),
                    "language_count": int(language_counts.get(candidate_term) or 0),
                },
            )
            distance = float(result_row.get("distance") or 1.0)
            score = round(max(0.0, 1.0 - distance), 6)
            payload["source_term_count"] += 1
            payload["semantic_neighbor_count"] += 1
            payload["max_score"] = max(float(payload.get("max_score") or 0.0), score)
            if len(payload["source_terms"]) < 6 and source_term not in payload["source_terms"]:
                payload["source_terms"].append(source_term)

    rows = list(mentions.values())
    for row in rows:
        translation_links = int(row.get("translation_links") or 0)
        language_count = int(row.get("language_count") or 0)
        translation_gap = max(0, language_count - translation_links)
        row["translation_gap"] = translation_gap
        row["translation_thin"] = bool(
            (language_count > 0 and translation_gap > 0)
            or (translation_links > 0 and bool(row.get("placeholder_definition")))
        )
        row["priority_score"] = round(
            (int(row.get("source_term_count") or 0) * 12)
            + (int(row.get("semantic_neighbor_count") or 0) * 5)
            + int(round(float(row.get("max_score") or 0.0) * 20))
            + (20 if bool(row.get("missing_term")) else 0)
            + (12 if bool(row.get("existing_weak_term")) else 0)
            + (14 if bool(row.get("missing_part_of_speech")) else 0)
            + (18 if bool(row.get("placeholder_definition")) else 0)
            + (10 if bool(row.get("translation_thin")) else 0)
            + (translation_gap * 7),
            3,
        )
    rows.sort(
        key=lambda row: (
            -float(row.get("priority_score") or 0),
            -int(row.get("source_term_count") or 0),
            -float(row.get("max_score") or 0.0),
            str(row.get("term") or ""),
        )
    )
    return rows[:bounded_limit]


def _harvest_multilingual_seed_rows(db_path: Path, *, limit: int = 24) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    bounded_limit = max(4, min(int(limit or 24), 64))
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            WITH translation_relationships AS (
                SELECT w.term,
                       COALESCE(w.part_of_speech, '') AS part_of_speech,
                       COUNT(*) AS translation_links,
                       MAX(COALESCE(w.last_refreshed, 0)) AS last_refreshed
                FROM relationships r
                JOIN words w ON w.id = r.word_id
                WHERE lower(r.relationship_type) = 'translation'
                GROUP BY w.id
            ),
            multilingual_lexemes AS (
                SELECT lower(COALESCE(NULLIF(base_term, ''), lemma)) AS term,
                       COUNT(DISTINCT lower(lang)) AS language_count,
                       MAX(COALESCE(last_refreshed, 0)) AS last_refreshed
                FROM lexemes
                GROUP BY lower(COALESCE(NULLIF(base_term, ''), lemma))
            )
            SELECT tr.term,
                   tr.part_of_speech,
                   tr.translation_links,
                   COALESCE(ml.language_count, 0) AS language_count,
                   MAX(tr.last_refreshed, COALESCE(ml.last_refreshed, 0)) AS freshness
            FROM translation_relationships tr
            LEFT JOIN multilingual_lexemes ml ON ml.term = lower(tr.term)
            WHERE length(tr.term) >= 2
            ORDER BY tr.translation_links DESC, language_count DESC, freshness DESC, tr.term ASC
            LIMIT ?
            """,
            (bounded_limit * 2,),
        ).fetchall()
    harvested: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        term = str(row["term"] or "").strip().lower()
        if not _valid_lexicon_term(term) or term in seen:
            continue
        seen.add(term)
        harvested.append(
            {
                "term": term,
                "part_of_speech": str(row["part_of_speech"] or ""),
                "translation_links": int(row["translation_links"] or 0),
                "language_count": int(row["language_count"] or 0),
                "freshness": int(row["freshness"] or 0),
            }
        )
        if len(harvested) >= bounded_limit:
            break
    return harvested


def _harvest_multilingual_seed_terms(db_path: Path, *, limit: int = 24) -> list[str]:
    return [
        str(row.get("term") or "") for row in _harvest_multilingual_seed_rows(db_path, limit=limit)
    ]


def _build_growth_priority_queue(
    *,
    forge_local_run: dict[str, Any] | None,
    lexicon_backfill_run: dict[str, Any] | None,
    multilingual_seed_rows: list[dict[str, Any]] | None,
    incomplete_seed_rows: list[dict[str, Any]] | None,
    definition_seed_rows: list[dict[str, Any]] | None = None,
    usage_seed_rows: list[dict[str, Any]] | None = None,
    vector_seed_rows: list[dict[str, Any]] | None = None,
    emotion_seed_terms: list[str] | None = None,
    derivation_seed_terms: list[str] | None = None,
    coverage_focus: list[dict[str, Any]] | None = None,
    limit: int = 16,
) -> list[dict[str, Any]]:
    bounded_limit = max(4, min(int(limit or 16), 32))
    queue: dict[str, dict[str, Any]] = {}

    def _ensure(term: str) -> dict[str, Any] | None:
        normalized = str(term or "").strip().lower()
        if not _valid_lexicon_term(normalized):
            return None
        row = queue.get(normalized)
        if row is None:
            row = {
                "term": normalized,
                "priority_score": 0,
                "source_families": [],
                "reasons": [],
                "translation_links": 0,
                "language_count": 0,
                "support_count": 0,
            }
            queue[normalized] = row
        return row

    def _push_family(row: dict[str, Any], family: str) -> None:
        if family not in row["source_families"]:
            row["source_families"].append(family)

    def _push_reason(row: dict[str, Any], reason: str) -> None:
        if reason not in row["reasons"]:
            row["reasons"].append(reason)

    focus_bonus_by_family: dict[str, int] = {}
    semantic_focus_bonus = 0
    for item in coverage_focus or []:
        if not isinstance(item, dict):
            continue
        family = str(item.get("recommended_family") or "").strip()
        gap = float(item.get("gap") or 0.0)
        if family:
            focus_bonus_by_family[family] = max(
                focus_bonus_by_family.get(family, 0), int(round(gap * 40))
            )
        if str(item.get("preferred_mode") or "").strip() == "semantic":
            semantic_focus_bonus = max(semantic_focus_bonus, int(round(gap * 40)))

    forge_local_payload = forge_local_run if isinstance(forge_local_run, dict) else {}
    for candidate in forge_local_payload.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        row = _ensure(str(candidate.get("term") or ""))
        if row is None:
            continue
        row["priority_score"] += int(candidate.get("score") or 0)
        row["support_count"] = max(row["support_count"], int(candidate.get("support_count") or 0))
        _push_family(row, "forge_local")
        for source in candidate.get("sources") or []:
            text = str(source or "").strip()
            if text:
                _push_reason(row, text)

    for term in (lexicon_backfill_run or {}).get("terms") or []:
        row = _ensure(str(term))
        if row is None:
            continue
        row["priority_score"] += 36
        _push_family(row, "lexicon_backfill")
        _push_reason(row, "translation_backfill")

    for item in multilingual_seed_rows or []:
        if not isinstance(item, dict):
            continue
        row = _ensure(str(item.get("term") or ""))
        if row is None:
            continue
        translation_links = int(item.get("translation_links") or 0)
        language_count = int(item.get("language_count") or 0)
        row["translation_links"] = max(row["translation_links"], translation_links)
        row["language_count"] = max(row["language_count"], language_count)
        row["priority_score"] += 24 + (translation_links * 4) + (language_count * 5)
        row["priority_score"] += focus_bonus_by_family.get("multilingual", 0)
        _push_family(row, "multilingual")
        if translation_links:
            _push_reason(row, f"{translation_links} translation links")
        if language_count:
            _push_reason(row, f"{language_count} languages")
        if focus_bonus_by_family.get("multilingual"):
            _push_reason(row, "growth focus: translations")

    for item in incomplete_seed_rows or []:
        if not isinstance(item, dict):
            continue
        row = _ensure(str(item.get("term") or ""))
        if row is None:
            continue
        missing_fields = [
            str(field).strip() for field in (item.get("missing_fields") or []) if str(field).strip()
        ]
        relationship_count = int(item.get("relationship_count") or 0)
        translation_count = int(item.get("translation_count") or 0)
        row["translation_links"] = max(row["translation_links"], translation_count)
        row["priority_score"] += 18 + (len(missing_fields) * 7) + max(0, 3 - relationship_count) * 6
        row["priority_score"] += focus_bonus_by_family.get("revisit", 0)
        _push_family(row, "revisit")
        for field in missing_fields[:4]:
            _push_reason(row, f"missing {field}")
        if translation_count:
            _push_reason(row, f"{translation_count} translation links")
        if focus_bonus_by_family.get("revisit"):
            _push_reason(row, "growth focus: structural gaps")

    for item in definition_seed_rows or []:
        if not isinstance(item, dict):
            continue
        row = _ensure(str(item.get("term") or ""))
        if row is None:
            continue
        mention_count = int(item.get("mention_count") or 0)
        source_term_count = int(item.get("source_term_count") or 0)
        is_function_word = bool(item.get("is_function_word"))
        if is_function_word:
            row["priority_score"] += 1 + min(source_term_count, 3) + min(mention_count, 3)
        else:
            row["priority_score"] += 14 + (source_term_count * 6) + min(mention_count, 8)
        row["priority_score"] += focus_bonus_by_family.get("definitions", 0)
        _push_family(row, "definitions")
        if is_function_word:
            _push_reason(row, "function word")
        if source_term_count:
            _push_reason(row, f"defined by {source_term_count} terms")
        if mention_count:
            _push_reason(row, f"{mention_count} definition mentions")
        if focus_bonus_by_family.get("definitions"):
            _push_reason(row, "growth focus: definitions")

    for item in usage_seed_rows or []:
        if not isinstance(item, dict):
            continue
        row = _ensure(str(item.get("term") or ""))
        if row is None:
            continue
        mention_count = int(item.get("mention_count") or 0)
        source_term_count = int(item.get("source_term_count") or 0)
        is_function_word = bool(item.get("is_function_word"))
        if is_function_word:
            row["priority_score"] += 1 + min(source_term_count, 3) + min(mention_count, 3)
        else:
            row["priority_score"] += 16 + (source_term_count * 7) + min(mention_count, 10)
        row["priority_score"] += focus_bonus_by_family.get("usage_examples", 0)
        _push_family(row, "usage_examples")
        if is_function_word:
            _push_reason(row, "function word")
        if source_term_count:
            _push_reason(row, f"used by {source_term_count} terms")
        if mention_count:
            _push_reason(row, f"{mention_count} usage mentions")
        if focus_bonus_by_family.get("usage_examples"):
            _push_reason(row, "growth focus: usage")

    for item in vector_seed_rows or []:
        if not isinstance(item, dict):
            continue
        row = _ensure(str(item.get("term") or ""))
        if row is None:
            continue
        source_term_count = int(item.get("source_term_count") or 0)
        semantic_neighbor_count = int(item.get("semantic_neighbor_count") or 0)
        max_score = float(item.get("max_score") or 0.0)
        translation_gap = int(item.get("translation_gap") or 0)
        language_count = int(item.get("language_count") or 0)
        row["priority_score"] += (
            20
            + (source_term_count * 6)
            + (semantic_neighbor_count * 4)
            + int(round(max_score * 16))
        )
        row["priority_score"] += (
            semantic_focus_bonus
            + (translation_gap * 7)
            + (6 if bool(item.get("translation_thin")) else 0)
        )
        _push_family(row, "vector_guidance")
        if source_term_count:
            _push_reason(row, f"semantic neighbors from {source_term_count} weak terms")
        if semantic_neighbor_count:
            _push_reason(row, f"{semantic_neighbor_count} vector hits")
        if item.get("missing_term"):
            _push_reason(row, "missing semantic term")
        elif item.get("existing_weak_term"):
            _push_reason(row, "existing weak semantic term")
        if item.get("translation_thin"):
            _push_reason(row, f"translation-thin across {language_count} languages")
        if semantic_focus_bonus:
            _push_reason(row, "growth focus: semantic")

    for term in emotion_seed_terms or []:
        row = _ensure(str(term))
        if row is None:
            continue
        row["priority_score"] += 18 + focus_bonus_by_family.get("emotion", 0)
        _push_family(row, "emotion")
        _push_reason(row, "emotion enrichment")
        if focus_bonus_by_family.get("emotion"):
            _push_reason(row, "growth focus: emotion")

    for term in derivation_seed_terms or []:
        row = _ensure(str(term))
        if row is None:
            continue
        row["priority_score"] += 22 + focus_bonus_by_family.get("derivation", 0)
        _push_family(row, "derivation")
        _push_reason(row, "derivation enrichment")
        if focus_bonus_by_family.get("derivation"):
            _push_reason(row, "growth focus: derivation")

    ranked = list(queue.values())
    for row in ranked:
        row["source_families"] = sorted(row["source_families"])
        row["reasons"] = row["reasons"][:4]
    ranked.sort(
        key=lambda row: (
            -int(row.get("priority_score") or 0),
            -int(row.get("translation_links") or 0),
            -int(row.get("language_count") or 0),
            -int(row.get("support_count") or 0),
            str(row.get("term") or ""),
        )
    )
    return ranked[:bounded_limit]


def _harvest_incomplete_word_rows(db_path: Path, *, limit: int = 24) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    bounded_limit = max(4, min(int(limit or 24), 64))
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        has_word_emotion = "word_emotion" in tables
        word_emotion_join = (
            "LEFT JOIN word_emotion we ON we.word_id = w.id" if has_word_emotion else ""
        )
        word_emotion_select = (
            "CASE WHEN COALESCE(we.word_id, 0) = 0 THEN 1 ELSE 0 END" if has_word_emotion else "1"
        )
        word_emotion_filter = "OR COALESCE(we.word_id, 0) = 0" if has_word_emotion else "OR 1 = 1"
        word_emotion_weight = (
            "CASE WHEN COALESCE(we.word_id, 0) = 0 THEN 10 ELSE 0 END +"
            if has_word_emotion
            else "10 +"
        )
        rows = conn.execute(
            f"""
            WITH relationship_counts AS (
                SELECT lower(term) AS term, COUNT(*) AS relationship_count
                FROM (
                    SELECT w.term AS term
                    FROM relationships r
                    JOIN words w ON w.id = r.word_id
                )
                GROUP BY lower(term)
            ),
            translation_counts AS (
                SELECT lower(w.term) AS term, COUNT(*) AS translation_count
                FROM relationships r
                JOIN words w ON w.id = r.word_id
                WHERE lower(r.relationship_type) = 'translation'
                GROUP BY lower(w.term)
            ),
            derivation_counts AS (
                SELECT lower(w.term) AS term, COUNT(*) AS derivation_count
                FROM relationships r
                JOIN words w ON w.id = r.word_id
                WHERE lower(r.relationship_type) IN ('derived_from', 'etymological_source')
                GROUP BY lower(w.term)
            ),
            phonetic_counts AS (
                SELECT lower(w.term) AS term, COUNT(*) AS phonetic_count
                FROM phonetics p
                JOIN words w ON w.id = p.word_id
                GROUP BY lower(w.term)
            )
            SELECT lower(w.term) AS term,
                   COALESCE(w.part_of_speech, '') AS part_of_speech,
                   CASE WHEN COALESCE(trim(w.definition), '') = '' THEN 1 ELSE 0 END AS missing_definition,
                   CASE WHEN COALESCE(trim(w.part_of_speech), '') = '' THEN 1 ELSE 0 END AS missing_pos,
                   CASE WHEN COALESCE(pc.phonetic_count, 0) = 0 THEN 1 ELSE 0 END AS missing_phonetics,
                   CASE WHEN COALESCE(dc.derivation_count, 0) = 0 THEN 1 ELSE 0 END AS missing_derivation,
                   {word_emotion_select} AS missing_emotion,
                   COALESCE(rc.relationship_count, 0) AS relationship_count,
                   COALESCE(tc.translation_count, 0) AS translation_count,
                   COALESCE(w.last_refreshed, 0) AS freshness
            FROM words w
            LEFT JOIN relationship_counts rc ON rc.term = lower(w.term)
            LEFT JOIN translation_counts tc ON tc.term = lower(w.term)
            LEFT JOIN derivation_counts dc ON dc.term = lower(w.term)
            LEFT JOIN phonetic_counts pc ON pc.term = lower(w.term)
            {word_emotion_join}
            WHERE length(COALESCE(w.term, '')) >= 2
              AND (
                COALESCE(trim(w.definition), '') = ''
                OR COALESCE(trim(w.part_of_speech), '') = ''
                OR COALESCE(pc.phonetic_count, 0) = 0
                OR COALESCE(dc.derivation_count, 0) = 0
                {word_emotion_filter}
                OR COALESCE(rc.relationship_count, 0) < 3
              )
            ORDER BY (
                CASE WHEN COALESCE(trim(w.definition), '') = '' THEN 40 ELSE 0 END +
                CASE WHEN COALESCE(trim(w.part_of_speech), '') = '' THEN 20 ELSE 0 END +
                CASE WHEN COALESCE(pc.phonetic_count, 0) = 0 THEN 12 ELSE 0 END +
                CASE WHEN COALESCE(dc.derivation_count, 0) = 0 THEN 11 ELSE 0 END +
                {word_emotion_weight}
                CASE WHEN COALESCE(rc.relationship_count, 0) < 3
                     THEN (3 - COALESCE(rc.relationship_count, 0)) * 8
                     ELSE 0 END +
                CASE WHEN COALESCE(tc.translation_count, 0) > 0 THEN 10 ELSE 0 END
            ) DESC,
            freshness DESC,
            lower(w.term) ASC
            LIMIT ?
            """,
            (bounded_limit,),
        ).fetchall()
    harvested: list[dict[str, Any]] = []
    for row in rows:
        term = str(row["term"] or "").strip().lower()
        if not _valid_lexicon_term(term):
            continue
        missing: list[str] = []
        if int(row["missing_definition"] or 0):
            missing.append("definition")
        if int(row["missing_pos"] or 0):
            missing.append("part_of_speech")
        if int(row["missing_phonetics"] or 0):
            missing.append("phonetics")
        if int(row["missing_derivation"] or 0):
            missing.append("derivation")
        if int(row["missing_emotion"] or 0):
            missing.append("emotion")
        if int(row["relationship_count"] or 0) < 3:
            missing.append("relationships")
        harvested.append(
            {
                "term": term,
                "part_of_speech": str(row["part_of_speech"] or ""),
                "missing_fields": missing,
                "relationship_count": int(row["relationship_count"] or 0),
                "translation_count": int(row["translation_count"] or 0),
            }
        )
    return harvested


def _lexicon_progress_snapshot(db_path: Path) -> dict[str, int]:
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
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        emotion_count = (
            int(conn.execute("SELECT COUNT(*) FROM word_emotion").fetchone()[0])
            if "word_emotion" in tables
            else 0
        )
        return {
            "word_count": int(conn.execute("SELECT COUNT(*) FROM words").fetchone()[0]),
            "defined_word_count": int(
                conn.execute(
                    "SELECT COUNT(*) FROM words WHERE COALESCE(trim(definition), '') <> ''"
                ).fetchone()[0]
            ),
            "pos_word_count": int(
                conn.execute(
                    "SELECT COUNT(*) FROM words WHERE COALESCE(trim(part_of_speech), '') <> ''"
                ).fetchone()[0]
            ),
            "usage_example_word_count": int(
                conn.execute(
                    "SELECT COUNT(*) FROM words WHERE COALESCE(trim(usage_examples), '') <> ''"
                ).fetchone()[0]
            ),
            "phonetic_word_count": int(
                conn.execute("SELECT COUNT(DISTINCT word_id) FROM phonetics").fetchone()[0]
            ),
            "emotion_word_count": emotion_count,
            "derivation_word_count": int(
                conn.execute("""
                    SELECT COUNT(DISTINCT word_id)
                    FROM relationships
                    WHERE lower(relationship_type) IN ('derived_from', 'etymological_source')
                    """).fetchone()[0]
            ),
            "relationship_count": int(
                conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
            ),
            "translation_word_count": int(
                conn.execute(
                    "SELECT COUNT(DISTINCT word_id) FROM relationships WHERE lower(relationship_type) = 'translation'"
                ).fetchone()[0]
            ),
        }


def _resolve_growth_focus(snapshot: dict[str, int]) -> list[dict[str, Any]]:
    word_count = max(0, int(snapshot.get("word_count") or 0))
    if word_count <= 0:
        return []
    dimensions = [
        ("definitions", int(snapshot.get("defined_word_count") or 0), "definitions", "community"),
        ("part_of_speech", int(snapshot.get("pos_word_count") or 0), "revisit", "community"),
        (
            "usage_examples",
            int(snapshot.get("usage_example_word_count") or 0),
            "usage_examples",
            "context",
        ),
        ("phonetics", int(snapshot.get("phonetic_word_count") or 0), "revisit", "semantic"),
        ("emotion", int(snapshot.get("emotion_word_count") or 0), "emotion", "semantic"),
        ("derivation", int(snapshot.get("derivation_word_count") or 0), "derivation", "derivation"),
        (
            "translations",
            int(snapshot.get("translation_word_count") or 0),
            "multilingual",
            "crosslingual",
        ),
    ]
    ranked: list[dict[str, Any]] = []
    for name, count, family, mode in dimensions:
        ratio = max(0.0, min(1.0, float(count) / float(word_count)))
        ranked.append(
            {
                "dimension": name,
                "count": count,
                "ratio": round(ratio, 4),
                "gap": round(1.0 - ratio, 4),
                "recommended_family": family,
                "preferred_mode": mode,
            }
        )
    ranked.sort(key=lambda row: (float(row.get("ratio") or 0.0), str(row.get("dimension") or "")))
    return ranked


def _resolve_growth_phonetic_limit(
    snapshot: dict[str, int],
    *,
    requested_limit: int | None,
    thesaurus_limit: int | None,
) -> int:
    if requested_limit is not None:
        return max(0, int(requested_limit))
    word_count = max(0, int(snapshot.get("word_count") or 0))
    phonetic_word_count = max(0, int(snapshot.get("phonetic_word_count") or 0))
    if word_count <= 0:
        return max(4, min(16, int(thesaurus_limit or 24)))
    coverage = float(phonetic_word_count) / float(word_count)
    ceiling = max(4, min(48, int(thesaurus_limit or 24)))
    if coverage < 0.02:
        return min(ceiling, 16)
    if coverage < 0.08:
        return min(ceiling, 12)
    if coverage < 0.18:
        return min(ceiling, 8)
    return min(ceiling, 4)


def _resolve_growth_emotion_limit(
    snapshot: dict[str, int],
    *,
    requested_limit: int | None,
    thesaurus_limit: int | None,
) -> int:
    if requested_limit is not None:
        return max(0, int(requested_limit))
    word_count = max(0, int(snapshot.get("word_count") or 0))
    emotion_word_count = max(0, int(snapshot.get("emotion_word_count") or 0))
    if word_count <= 0:
        return max(4, min(16, int(thesaurus_limit or 24)))
    coverage = float(emotion_word_count) / float(word_count)
    ceiling = max(4, min(48, int(thesaurus_limit or 24)))
    if coverage < 0.02:
        return min(ceiling, 16)
    if coverage < 0.08:
        return min(ceiling, 12)
    if coverage < 0.18:
        return min(ceiling, 8)
    return min(ceiling, 4)


def _resolve_growth_derivation_limit(
    snapshot: dict[str, int],
    *,
    requested_limit: int | None,
    thesaurus_limit: int | None,
) -> int:
    if requested_limit is not None:
        return max(0, int(requested_limit))
    word_count = max(0, int(snapshot.get("word_count") or 0))
    derivation_word_count = max(0, int(snapshot.get("derivation_word_count") or 0))
    if word_count <= 0:
        return max(4, min(24, int(thesaurus_limit or 24)))
    coverage = float(derivation_word_count) / float(word_count)
    ceiling = max(4, min(48, int(thesaurus_limit or 24)))
    if coverage < 0.04:
        return min(ceiling, 24)
    if coverage < 0.12:
        return min(ceiling, 18)
    if coverage < 0.24:
        return min(ceiling, 12)
    return min(ceiling, 6)


def _harvest_emotion_seed_rows(db_path: Path, *, limit: int = 24) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    bounded_limit = max(4, min(int(limit or 24), 64))
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        emotion_join = (
            "LEFT JOIN word_emotion we ON we.word_id = w.id" if "word_emotion" in tables else ""
        )
        emotion_filter = "we.word_id IS NULL" if "word_emotion" in tables else "1 = 1"
        rows = conn.execute(
            f"""
            WITH relationship_counts AS (
                SELECT word_id, COUNT(*) AS relationship_count
                FROM relationships
                GROUP BY word_id
            ),
            translation_counts AS (
                SELECT word_id, COUNT(*) AS translation_count
                FROM relationships
                WHERE lower(relationship_type) = 'translation'
                GROUP BY word_id
            )
            SELECT lower(w.term) AS term,
                   COALESCE(w.part_of_speech, '') AS part_of_speech,
                   COALESCE(rc.relationship_count, 0) AS relationship_count,
                   COALESCE(tc.translation_count, 0) AS translation_count,
                   COALESCE(trim(w.definition), '') AS definition,
                   COALESCE(w.last_refreshed, 0) AS freshness
            FROM words w
            LEFT JOIN relationship_counts rc ON rc.word_id = w.id
            LEFT JOIN translation_counts tc ON tc.word_id = w.id
            {emotion_join}
            WHERE {emotion_filter}
              AND COALESCE(trim(w.definition), '') <> ''
              AND length(COALESCE(w.term, '')) >= 1
            ORDER BY (
                CASE WHEN COALESCE(tc.translation_count, 0) > 0 THEN 18 ELSE 0 END +
                CASE WHEN COALESCE(rc.relationship_count, 0) >= 3
                     THEN 14
                     ELSE COALESCE(rc.relationship_count, 0) * 4 END +
                CASE WHEN COALESCE(trim(w.part_of_speech), '') <> '' THEN 6 ELSE 0 END +
                CASE WHEN length(COALESCE(trim(w.definition), '')) >= 24 THEN 6 ELSE 0 END
            ) DESC,
            freshness DESC,
            lower(w.term) ASC
            LIMIT ?
            """,
            (bounded_limit,),
        ).fetchall()
    harvested: list[dict[str, Any]] = []
    for row in rows:
        term = str(row["term"] or "").strip().lower()
        if not _valid_lexicon_term(term):
            continue
        harvested.append(
            {
                "term": term,
                "part_of_speech": str(row["part_of_speech"] or ""),
                "relationship_count": int(row["relationship_count"] or 0),
                "translation_count": int(row["translation_count"] or 0),
            }
        )
    return harvested


def _harvest_derivation_seed_rows(db_path: Path, *, limit: int = 24) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    bounded_limit = max(4, min(int(limit or 24), 64))
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            WITH derivation_counts AS (
                SELECT word_id, COUNT(*) AS derivation_count
                FROM relationships
                WHERE lower(relationship_type) IN ('derived_from', 'etymological_source')
                GROUP BY word_id
            ),
            relationship_counts AS (
                SELECT word_id, COUNT(*) AS relationship_count
                FROM relationships
                GROUP BY word_id
            ),
            root_counts AS (
                SELECT word_id, COUNT(*) AS root_count
                FROM word_morphemes
                GROUP BY word_id
            )
            SELECT lower(w.term) AS term,
                   COALESCE(w.part_of_speech, '') AS part_of_speech,
                   COALESCE(trim(w.definition), '') AS definition,
                   COALESCE(dc.derivation_count, 0) AS derivation_count,
                   COALESCE(rc.relationship_count, 0) AS relationship_count,
                   COALESCE(rt.root_count, 0) AS root_count,
                   COALESCE(w.last_refreshed, 0) AS freshness
            FROM words w
            LEFT JOIN derivation_counts dc ON dc.word_id = w.id
            LEFT JOIN relationship_counts rc ON rc.word_id = w.id
            LEFT JOIN root_counts rt ON rt.word_id = w.id
            WHERE COALESCE(trim(w.definition), '') <> ''
              AND COALESCE(dc.derivation_count, 0) = 0
              AND length(COALESCE(w.term, '')) >= 2
            ORDER BY (
                CASE WHEN COALESCE(rt.root_count, 0) > 0 THEN 18 ELSE 0 END +
                CASE WHEN COALESCE(rc.relationship_count, 0) >= 3
                     THEN 14
                     ELSE COALESCE(rc.relationship_count, 0) * 4 END +
                CASE WHEN COALESCE(trim(w.part_of_speech), '') <> '' THEN 6 ELSE 0 END +
                CASE WHEN length(COALESCE(trim(w.definition), '')) >= 24 THEN 6 ELSE 0 END
            ) DESC,
            freshness DESC,
            lower(w.term) ASC
            LIMIT ?
            """,
            (bounded_limit,),
        ).fetchall()
    harvested: list[dict[str, Any]] = []
    for row in rows:
        term = str(row["term"] or "").strip().lower()
        if not _valid_lexicon_term(term):
            continue
        harvested.append(
            {
                "term": term,
                "part_of_speech": str(row["part_of_speech"] or ""),
                "relationship_count": int(row["relationship_count"] or 0),
                "root_count": int(row["root_count"] or 0),
            }
        )
    return harvested


def _run_emotion_enrichment_batch(
    db_path: Path, *, seed_terms: list[str], limit: int
) -> dict[str, Any]:
    bounded_limit = max(0, min(int(limit or 0), 64))
    selected_terms = [
        str(term or "").strip().lower() for term in seed_terms if str(term or "").strip()
    ][:bounded_limit]
    if bounded_limit <= 0 or not selected_terms:
        return {
            "contract": "eidos.word_forge.emotion_enrichment.v1",
            "status": "skipped",
            "seed_terms": [],
            "processed_count": 0,
            "emotion_delta": 0,
        }
    from word_forge.database.database_manager import DBManager
    from word_forge.emotion.emotion_manager import EmotionManager

    db = DBManager(db_path=db_path)
    db.create_tables()
    emotion_manager = EmotionManager(db)
    processed = 0
    delta = 0
    enriched_terms: list[str] = []
    try:
        for term in selected_terms:
            try:
                word_id = db.get_word_id(term)
            except Exception:
                continue
            processed += 1
            before = db.execute_query(
                "SELECT word_id FROM word_emotion WHERE word_id = ?", (word_id,)
            )
            if emotion_manager.enrich_word_emotions(term, word_id):
                after = db.execute_query(
                    "SELECT word_id FROM word_emotion WHERE word_id = ?", (word_id,)
                )
                if after and not before:
                    delta += 1
                    enriched_terms.append(term)
        return {
            "contract": "eidos.word_forge.emotion_enrichment.v1",
            "status": "completed",
            "seed_terms": selected_terms,
            "processed_count": processed,
            "emotion_delta": delta,
            "enriched_terms": enriched_terms[:24],
        }
    finally:
        db.close()


def _word_row_map(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    rows = conn.execute("""
        SELECT
            id,
            term,
            COALESCE(definition, '') AS definition,
            COALESCE(part_of_speech, '') AS part_of_speech,
            COALESCE(usage_examples, '') AS usage_examples
        FROM words
        """).fetchall()
    payload: dict[str, dict[str, Any]] = {}
    for row in rows:
        term = str(row["term"] or "").strip().lower()
        if not term:
            continue
        payload[term] = {
            "id": int(row["id"] or 0),
            "definition": str(row["definition"] or ""),
            "part_of_speech": str(row["part_of_speech"] or ""),
            "usage_examples": [
                item for item in str(row["usage_examples"] or "").splitlines() if item
            ],
        }
    return payload


def run_lexicon_backfill(
    *, repo_root: Path, db_path: Path, limit: int | None = None, apply: bool = True
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    db_path = db_path.resolve()
    bounded_limit = max(8, min(int(limit or 96), 320))
    if not db_path.exists():
        return {
            "contract": "eidos.word_forge.lexicon_backfill.v1",
            "status": "missing",
            "limit": bounded_limit,
            "inserted_word_count": 0,
            "updated_word_count": 0,
            "translation_relationship_delta": 0,
            "morpheme_link_delta": 0,
            "language_count": 0,
            "terms": [],
        }

    db = DBManager(db_path=db_path)
    morphology = MorphologyManager(db)
    inserted_word_count = 0
    updated_word_count = 0
    translation_relationship_delta = 0
    morpheme_link_delta = 0
    enriched_terms: list[str] = []
    languages: set[str] = set()

    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        word_map = _word_row_map(conn)
        existing_morpheme_word_ids = {
            int(row[0])
            for row in conn.execute("SELECT DISTINCT word_id FROM word_morphemes").fetchall()
        }
        lexeme_rows = conn.execute(
            """
            SELECT id, lemma, lang, part_of_speech, gloss, base_term, source
            FROM lexemes
            ORDER BY last_refreshed DESC
            LIMIT ?
            """,
            (bounded_limit,),
        ).fetchall()
        lexeme_ids = [int(row["id"]) for row in lexeme_rows]
        translation_rows = []
        if lexeme_ids:
            placeholders = ",".join("?" for _ in lexeme_ids)
            translation_rows = conn.execute(
                f"""
                SELECT lexeme_id, target_lang, target_term, relation, source
                FROM translations
                WHERE lexeme_id IN ({placeholders})
                ORDER BY last_refreshed DESC
                LIMIT ?
                """,
                tuple(lexeme_ids) + (bounded_limit * 8,),
            ).fetchall()
        translations_by_lexeme: dict[int, list[sqlite3.Row]] = {}
        for row in translation_rows:
            translations_by_lexeme.setdefault(int(row["lexeme_id"]), []).append(row)

    def _upsert_word(
        term: str,
        *,
        definition: str = "",
        part_of_speech: str = "",
        usage_examples: list[str] | None = None,
    ) -> str:
        nonlocal inserted_word_count, updated_word_count
        normalized = str(term or "").strip().lower()
        if not _valid_lexicon_term(normalized):
            return "skipped"
        current = word_map.get(normalized) or {}
        merged_definition = str(current.get("definition") or "") or str(definition or "")
        merged_part = str(current.get("part_of_speech") or "") or str(part_of_speech or "")
        merged_examples = _coerce_usage_examples(
            list(current.get("usage_examples") or []), list(usage_examples or [])
        )
        if not apply:
            if current:
                return "existing"
            return "preview"
        if not current:
            db.insert_or_update_word(normalized, merged_definition, merged_part, merged_examples)
            word_id = db.get_word_id(normalized)
            word_map[normalized] = {
                "id": word_id,
                "definition": merged_definition,
                "part_of_speech": merged_part,
                "usage_examples": merged_examples,
            }
            inserted_word_count += 1
            enriched_terms.append(normalized)
            return "inserted"
        if (
            merged_definition != str(current.get("definition") or "")
            or merged_part != str(current.get("part_of_speech") or "")
            or merged_examples != list(current.get("usage_examples") or [])
        ):
            db.insert_or_update_word(normalized, merged_definition, merged_part, merged_examples)
            word_map[normalized] = {
                **current,
                "definition": merged_definition,
                "part_of_speech": merged_part,
                "usage_examples": merged_examples,
            }
            updated_word_count += 1
            return "updated"
        return "existing"

    for row in lexeme_rows:
        lemma = str(row["lemma"] or "").strip().lower()
        lang = str(row["lang"] or "").strip().lower()
        part_of_speech = str(row["part_of_speech"] or "").strip()
        gloss = str(row["gloss"] or "").strip()
        base_term = str(row["base_term"] or "").strip().lower()
        anchor_term = base_term if _valid_lexicon_term(base_term) else lemma
        if not _valid_lexicon_term(anchor_term):
            continue
        if lang:
            languages.add(lang)
        anchor_definition = gloss or (
            f"A {lang or 'multilingual'} lexical anchor aligned to '{anchor_term}'."
            if lemma and lemma != anchor_term
            else ""
        )
        anchor_examples = [f"{lang}:{lemma}" if lang and lemma else lemma or anchor_term]
        _upsert_word(
            anchor_term,
            definition=anchor_definition,
            part_of_speech=part_of_speech,
            usage_examples=anchor_examples,
        )

        if lemma and lemma != anchor_term and _valid_lexicon_term(lemma):
            lemma_definition = (
                gloss or f"A {lang or 'multilingual'} lexical form aligned to '{anchor_term}'."
            )
            _upsert_word(
                lemma,
                definition=lemma_definition,
                part_of_speech=part_of_speech,
                usage_examples=anchor_examples,
            )
            if apply:
                if db.insert_relationship(anchor_term, lemma, "translation"):
                    translation_relationship_delta += 1
                if db.word_exists(lemma) and db.insert_relationship(
                    lemma, anchor_term, "translation"
                ):
                    translation_relationship_delta += 1

        for translation in translations_by_lexeme.get(int(row["id"]), []):
            target_term = str(translation["target_term"] or "").strip().lower()
            target_lang = str(translation["target_lang"] or "").strip().lower()
            if not _valid_lexicon_term(target_term):
                continue
            if target_lang:
                languages.add(target_lang)
            target_definition = (
                f"A {target_lang or 'multilingual'} translation linked to '{anchor_term}'."
            )
            _upsert_word(
                target_term,
                definition=target_definition,
                part_of_speech="",
                usage_examples=[f"{target_lang}:{target_term}" if target_lang else target_term],
            )
            if apply and target_term != anchor_term:
                if db.insert_relationship(anchor_term, target_term, "translation"):
                    translation_relationship_delta += 1
                if db.word_exists(target_term) and db.insert_relationship(
                    target_term, anchor_term, "translation"
                ):
                    translation_relationship_delta += 1

        if apply:
            try:
                word_id = int((word_map.get(anchor_term) or {}).get("id") or 0)
                if word_id and word_id not in existing_morpheme_word_ids:
                    morphemes = [
                        piece
                        for piece in morphology.decompose(anchor_term)
                        if _valid_lexicon_term(piece)
                    ]
                    if len(morphemes) > 1:
                        morpheme_ids = []
                        for index, piece in enumerate(morphemes):
                            m_type = "root"
                            if index == 0 and len(morphemes) > 1:
                                m_type = "prefix"
                            elif index == len(morphemes) - 1 and len(morphemes) > 1:
                                m_type = "suffix"
                            morpheme_ids.append(morphology.upsert_morpheme(piece, m_type))
                        morphology.set_word_morphemes(word_id, morpheme_ids)
                        existing_morpheme_word_ids.add(word_id)
                        morpheme_link_delta += 1
            except Exception:
                pass

    db.close()
    return {
        "contract": "eidos.word_forge.lexicon_backfill.v1",
        "status": "completed" if apply else "preview",
        "limit": bounded_limit,
        "inserted_word_count": inserted_word_count,
        "updated_word_count": updated_word_count,
        "translation_relationship_delta": translation_relationship_delta,
        "morpheme_link_delta": morpheme_link_delta,
        "language_count": len(languages),
        "languages": sorted(languages),
        "terms": enriched_terms[:32],
    }


def _write_growth_report(repo_root: Path, report: dict[str, Any]) -> dict[str, str]:
    report_dir = _report_dir(repo_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = report_dir / f"word_forge_growth_{stamp}.json"
    md_path = report_dir / f"word_forge_growth_{stamp}.md"
    latest_json = report_dir / "latest.json"
    latest_md = report_dir / "latest.md"

    markdown_lines = [
        "# Word Forge Growth Run",
        "",
        f"- Generated: `{report.get('generated_at')}`",
        f"- Source count: `{report.get('discovery', {}).get('source_count', 0)}`",
        f"- Multilingual sources: `{report.get('discovery', {}).get('multilingual_source_count', 0)}`",
        f"- FastText sources: `{report.get('discovery', {}).get('fasttext_source_count', 0)}`",
        f"- Forge-local sources: `{report.get('discovery', {}).get('forge_local_source_count', 0)}`",
        f"- Corpus-local sources: `{report.get('discovery', {}).get('corpus_local_source_count', 0)}`",
        f"- Forge-local inserted: `{((report.get('forge_local_run') or {}).get('inserted_count', 0))}`",
        f"- Lexicon backfill inserted: `{((report.get('lexicon_backfill_run') or {}).get('inserted_word_count', 0))}`",
        f"- Lexicon languages: `{((report.get('lexicon_backfill_run') or {}).get('language_count', 0))}`",
        f"- Thesaurus seed terms: `{report.get('thesaurus_seed_term_count', 0)}`",
        f"- OMW translation delta: `{report.get('omw_translation_delta', 0)}`",
        f"- Multilingual runs: `{len(report.get('multilingual_runs') or [])}`",
        f"- FastText runs: `{len(report.get('fasttext_runs') or [])}`",
        f"- Polyglot status: `{((report.get('polyglot_run') or {}).get('status') or {}).get('status', 'unknown')}`",
        f"- Semantic vector indexing: `{(report.get('semantic_vector_run') or {}).get('status', 'unknown')}`",
        f"- Bridge status: `{((report.get('bridge_run') or {}).get('status') or {}).get('status', 'unknown')}`",
        "",
        "## Deltas",
        "",
        f"- Lexeme delta: `{report.get('lexeme_delta', 0)}`",
        f"- Translation delta: `{report.get('translation_delta', 0)}`",
        f"- Vector delta: `{report.get('vector_delta', 0)}`",
        f"- Forge-local term delta: `{report.get('forge_local_term_delta', 0)}`",
        f"- Lexicon term delta: `{report.get('lexicon_term_delta', 0)}`",
        f"- Translation relationship delta: `{report.get('translation_relationship_delta', 0)}`",
        f"- Semantic vector delta: `{report.get('semantic_vector_delta', 0)}`",
        f"- Applied delta: `{report.get('applied_delta', 0)}`",
        f"- Emotion delta: `{report.get('emotion_delta', 0)}`",
        f"- Morpheme delta: `{report.get('decomposed_lexeme_delta', 0)}`",
        f"- Morpheme link delta: `{report.get('morpheme_link_delta', 0)}`",
        f"- Fully bridged: `{report.get('fully_bridged', 0)}`",
        "",
    ]

    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    return {
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "latest_json": str(latest_json),
        "latest_markdown": str(latest_md),
    }


def run_word_forge_growth(
    *,
    repo_root: Path,
    db_path: Path,
    vector_db_path: Path,
    force: bool = False,
    multilingual_limit: int | None = None,
    fasttext_limit: int | None = None,
    polyglot_limit: int | None = None,
    forge_local_limit: int | None = None,
    lexicon_limit: int | None = None,
    thesaurus_limit: int | None = None,
    phonetic_limit: int | None = None,
    thesaurus_terms: list[str] | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    db_path = db_path.resolve()
    vector_db_path = vector_db_path.resolve()
    status_path = _status_path(repo_root)
    history_path = _history_path(repo_root)
    discovery = discover_growth_sources(repo_root)
    previous = read_json(status_path)
    source_signature_payload = {
        "multilingual": [
            source_signature(Path(item["path"]))
            for item in discovery.get("multilingual_sources", [])
            if Path(item["path"]).exists()
        ],
        "fasttext": [
            source_signature(Path(item["path"]))
            for item in discovery.get("fasttext_sources", [])
            if Path(item["path"]).exists()
        ],
        "forge_local": [
            source_signature(Path(item["path"]))
            for item in discovery.get("forge_local_sources", [])
            if Path(item["path"]).exists()
        ],
    }
    status_payload = {
        "contract": "eidos.word_forge.growth.status.v1",
        "status": "running",
        "phase": "discovering",
        "repo_root": str(repo_root),
        "db_path": str(db_path),
        "vector_db_path": str(vector_db_path),
        "discovery": discovery,
        "source_signature": source_signature_payload,
        "force": force,
        "lexicon_limit": lexicon_limit,
        "started_at": now_iso(),
    }
    write_json(status_path, status_payload)
    try:
        if (
            not force
            and previous.get("status") == "completed"
            and previous.get("source_signature") == source_signature_payload
        ):
            status_payload.update(
                {"status": "skipped", "phase": "unchanged", "finished_at": now_iso()}
            )
            write_json(status_path, status_payload)
            append_history(history_path, status_payload)
            return {"status": status_payload, "report": None, "artifacts": {}}

        multilingual_runs: list[dict[str, Any]] = []
        fasttext_runs: list[dict[str, Any]] = []
        before_vector_count = _semantic_vector_count(repo_root)

        status_payload["phase"] = "forge_local_discovery"
        write_json(status_path, status_payload)
        forge_local_run = run_forge_local_discovery(
            repo_root=repo_root,
            db_path=db_path,
            limit=forge_local_limit,
        )

        status_payload["phase"] = "multilingual_ingest"
        write_json(status_path, status_payload)
        for source in discovery.get("multilingual_sources", [])[:2]:
            multilingual_runs.append(
                run_multilingual_ingest(
                    repo_root=repo_root,
                    source_path=Path(source["path"]),
                    source_type=str(source["source_type"]),
                    db_path=db_path,
                    limit=multilingual_limit,
                    force=force,
                )
            )

        status_payload["phase"] = "fasttext_ingest"
        write_json(status_path, status_payload)
        for source in discovery.get("fasttext_sources", [])[:4]:
            lang = str(source["lang"])
            fasttext_runs.append(
                run_fasttext_ingest(
                    repo_root=repo_root,
                    source_path=Path(source["path"]),
                    lang=lang,
                    db_path=db_path,
                    vector_db_path=vector_db_path,
                    limit=fasttext_limit,
                    bootstrap_lang=None if lang == "en" else "en",
                    top_k=1,
                    min_score=0.55,
                    apply=(lang != "en"),
                    force=force,
                )
            )

        status_payload["phase"] = "polyglot"
        write_json(status_path, status_payload)
        polyglot_run = run_polyglot_decomposition(
            repo_root=repo_root,
            db_path=db_path,
            lang=None,
            limit=polyglot_limit,
            force=force,
        )

        status_payload["phase"] = "lexicon_backfill"
        write_json(status_path, status_payload)
        lexicon_backfill_run = run_lexicon_backfill(
            repo_root=repo_root,
            db_path=db_path,
            limit=lexicon_limit,
        )

        multilingual_seed_rows = _harvest_multilingual_seed_rows(
            db_path, limit=max(12, int(thesaurus_limit or 24))
        )
        multilingual_seed_terms = [str(row.get("term") or "") for row in multilingual_seed_rows]
        incomplete_seed_rows = _harvest_incomplete_word_rows(
            db_path, limit=max(12, int(thesaurus_limit or 24))
        )
        incomplete_seed_terms = [str(row.get("term") or "") for row in incomplete_seed_rows]
        definition_seed_rows = _harvest_definition_seed_rows(
            db_path, limit=max(24, int(thesaurus_limit or 24) * 2)
        )
        definition_seed_terms = [str(row.get("term") or "") for row in definition_seed_rows]
        usage_seed_rows = _harvest_usage_seed_rows(
            db_path, limit=max(24, int(thesaurus_limit or 24) * 2)
        )
        usage_seed_terms = [str(row.get("term") or "") for row in usage_seed_rows]
        vector_seed_rows = _harvest_vector_seed_rows(
            repo_root, db_path, limit=max(12, int(thesaurus_limit or 24))
        )
        vector_seed_terms = [str(row.get("term") or "") for row in vector_seed_rows]
        emotion_seed_rows = _harvest_emotion_seed_rows(
            db_path, limit=max(12, int(thesaurus_limit or 24))
        )
        emotion_seed_terms = [str(row.get("term") or "") for row in emotion_seed_rows]
        derivation_seed_rows = _harvest_derivation_seed_rows(
            db_path, limit=max(12, int(thesaurus_limit or 24))
        )
        derivation_seed_terms = [str(row.get("term") or "") for row in derivation_seed_rows]
        current_snapshot = _lexicon_progress_snapshot(db_path)
        resolved_thesaurus_terms = _synthesise_growth_thesaurus_terms(
            explicit_terms=thesaurus_terms,
            forge_local_run=forge_local_run,
            lexicon_backfill_run=lexicon_backfill_run,
            multilingual_seed_terms=multilingual_seed_terms,
            incomplete_seed_terms=incomplete_seed_terms,
            definition_seed_terms=definition_seed_terms,
            usage_seed_terms=usage_seed_terms,
            vector_seed_terms=vector_seed_terms,
            emotion_seed_terms=emotion_seed_terms,
            derivation_seed_terms=derivation_seed_terms,
            limit=max(12, int(thesaurus_limit or 24)),
        )
        coverage_focus = _resolve_growth_focus(current_snapshot)
        effective_phonetic_limit = _resolve_growth_phonetic_limit(
            current_snapshot,
            requested_limit=phonetic_limit,
            thesaurus_limit=thesaurus_limit,
        )
        effective_emotion_limit = _resolve_growth_emotion_limit(
            current_snapshot,
            requested_limit=None,
            thesaurus_limit=thesaurus_limit,
        )
        effective_derivation_limit = _resolve_growth_derivation_limit(
            current_snapshot,
            requested_limit=None,
            thesaurus_limit=thesaurus_limit,
        )

        priority_queue = _build_growth_priority_queue(
            forge_local_run=forge_local_run,
            lexicon_backfill_run=lexicon_backfill_run,
            multilingual_seed_rows=multilingual_seed_rows,
            incomplete_seed_rows=incomplete_seed_rows,
            definition_seed_rows=definition_seed_rows,
            usage_seed_rows=usage_seed_rows,
            vector_seed_rows=vector_seed_rows,
            emotion_seed_terms=emotion_seed_terms[:effective_emotion_limit],
            derivation_seed_terms=derivation_seed_terms[:effective_derivation_limit],
            coverage_focus=coverage_focus[:4],
            limit=16,
        )

        status_payload["phase"] = "thesaurus"
        status_payload["thesaurus_seed_terms"] = resolved_thesaurus_terms
        status_payload["effective_phonetic_limit"] = effective_phonetic_limit
        status_payload["usage_seed_term_count"] = len(usage_seed_terms)
        status_payload["usage_seed_terms"] = usage_seed_terms[:24]
        status_payload["vector_seed_term_count"] = len(vector_seed_terms)
        status_payload["vector_seed_terms"] = vector_seed_terms[:24]
        status_payload["vector_seed_placeholder_count"] = sum(
            1 for row in vector_seed_rows if bool(row.get("placeholder_definition"))
        )
        status_payload["vector_seed_missing_pos_count"] = sum(
            1 for row in vector_seed_rows if bool(row.get("missing_part_of_speech"))
        )
        status_payload["vector_seed_translation_thin_count"] = sum(
            1 for row in vector_seed_rows if bool(row.get("translation_thin"))
        )
        status_payload["vector_seed_translation_thin_terms"] = [
            str(row.get("term") or "")
            for row in vector_seed_rows
            if bool(row.get("translation_thin"))
        ][:8]
        status_payload["emotion_seed_term_count"] = len(emotion_seed_terms)
        status_payload["emotion_seed_terms"] = emotion_seed_terms[:24]
        status_payload["effective_emotion_limit"] = effective_emotion_limit
        status_payload["derivation_seed_term_count"] = len(derivation_seed_terms)
        status_payload["derivation_seed_terms"] = derivation_seed_terms[:24]
        status_payload["effective_derivation_limit"] = effective_derivation_limit
        status_payload["coverage_focus"] = coverage_focus[:4]
        write_json(status_path, status_payload)
        thesaurus_run = run_word_forge_thesaurus(
            repo_root=repo_root,
            db_path=db_path,
            limit=thesaurus_limit,
            phonetic_limit=effective_phonetic_limit,
            force=force,
            use_llm_phonetics=bool(effective_phonetic_limit > 0),
            seed_terms=resolved_thesaurus_terms or None,
        )

        status_payload["phase"] = "emotion_enrichment"
        write_json(status_path, status_payload)
        emotion_run = _run_emotion_enrichment_batch(
            db_path,
            seed_terms=emotion_seed_terms or resolved_thesaurus_terms,
            limit=effective_emotion_limit,
        )

        status_payload["phase"] = "semantic_vector_index"
        write_json(status_path, status_payload)
        vector_db = DBManager(db_path=db_path)
        try:
            vector_index_ok = run_vector_index(
                db_manager=vector_db,
                embedder="sentence-transformers/all-MiniLM-L6-v2",
                poll_interval=0.25,
                timeout=240.0,
            )
        finally:
            vector_db.close()
        after_vector_count = _semantic_vector_count(repo_root)
        semantic_vector_run = {
            "contract": "eidos.word_forge.semantic_vector_run.v1",
            "status": "completed" if vector_index_ok else "partial",
            "before_vector_count": before_vector_count,
            "after_vector_count": after_vector_count,
            "vector_delta": max(0, after_vector_count - before_vector_count),
        }

        status_payload["phase"] = "bridge_audit"
        write_json(status_path, status_payload)
        bridge_run = run_bridge_audit(repo_root=repo_root, db_path=db_path)

        report = {
            "contract": "eidos.word_forge.growth.report.v1",
            "generated_at": now_iso(),
            "repo_root": str(repo_root),
            "db_path": str(db_path),
            "vector_db_path": str(vector_db_path),
            "discovery": discovery,
            "forge_local_run": forge_local_run,
            "multilingual_runs": multilingual_runs,
            "fasttext_runs": fasttext_runs,
            "polyglot_run": polyglot_run,
            "lexicon_backfill_run": lexicon_backfill_run,
            "thesaurus_run": thesaurus_run,
            "semantic_vector_run": semantic_vector_run,
            "bridge_run": bridge_run,
            "forge_local_term_delta": int(forge_local_run.get("inserted_count") or 0),
            "lexicon_term_delta": int(lexicon_backfill_run.get("inserted_word_count") or 0)
            + int(lexicon_backfill_run.get("updated_word_count") or 0),
            "thesaurus_seed_term_count": len(resolved_thesaurus_terms),
            "thesaurus_seed_terms": resolved_thesaurus_terms[:24],
            "multilingual_seed_term_count": len(multilingual_seed_terms),
            "multilingual_seed_terms": multilingual_seed_terms[:24],
            "incomplete_seed_term_count": len(incomplete_seed_terms),
            "incomplete_seed_terms": incomplete_seed_terms[:24],
            "definition_seed_term_count": len(definition_seed_terms),
            "definition_seed_terms": definition_seed_terms[:24],
            "usage_seed_term_count": len(usage_seed_terms),
            "usage_seed_terms": usage_seed_terms[:24],
            "vector_seed_term_count": len(vector_seed_terms),
            "vector_seed_terms": vector_seed_terms[:24],
            "vector_seed_placeholder_count": sum(
                1 for row in vector_seed_rows if bool(row.get("placeholder_definition"))
            ),
            "vector_seed_missing_pos_count": sum(
                1 for row in vector_seed_rows if bool(row.get("missing_part_of_speech"))
            ),
            "vector_seed_translation_thin_count": sum(
                1 for row in vector_seed_rows if bool(row.get("translation_thin"))
            ),
            "vector_seed_translation_thin_terms": [
                str(row.get("term") or "")
                for row in vector_seed_rows
                if bool(row.get("translation_thin"))
            ][:8],
            "emotion_seed_term_count": len(emotion_seed_terms),
            "emotion_seed_terms": emotion_seed_terms[:24],
            "derivation_seed_term_count": len(derivation_seed_terms),
            "derivation_seed_terms": derivation_seed_terms[:24],
            "priority_queue": priority_queue,
            "lexeme_delta": sum(
                int((item.get("status") or {}).get("lexeme_delta") or 0)
                for item in multilingual_runs
            ),
            "translation_delta": sum(
                int((item.get("status") or {}).get("translation_delta") or 0)
                for item in multilingual_runs
            )
            + sum(
                int((item.get("status") or {}).get("translation_delta") or 0)
                for item in fasttext_runs
            ),
            "translation_relationship_delta": int(
                lexicon_backfill_run.get("translation_relationship_delta") or 0
            ),
            "vector_delta": sum(
                int((item.get("status") or {}).get("vector_delta") or 0) for item in fasttext_runs
            ),
            "semantic_vector_delta": int(semantic_vector_run.get("vector_delta") or 0),
            "relationship_delta": int(
                ((thesaurus_run.get("report") or {}).get("deltas") or {}).get("relationship_delta")
                or 0
            ),
            "phonetic_delta": int(
                ((thesaurus_run.get("report") or {}).get("deltas") or {}).get("phonetic_delta") or 0
            ),
            "effective_phonetic_limit": int(effective_phonetic_limit),
            "emotion_delta": int(emotion_run.get("emotion_delta") or 0),
            "effective_emotion_limit": int(effective_emotion_limit),
            "omw_translation_delta": int(
                ((thesaurus_run.get("report") or {}).get("deltas") or {}).get(
                    "omw_translation_delta"
                )
                or 0
            ),
            "morpheme_link_delta": int(lexicon_backfill_run.get("morpheme_link_delta") or 0),
            "lexicon_language_count": int(lexicon_backfill_run.get("language_count") or 0),
            "applied_delta": sum(
                int((item.get("status") or {}).get("applied_delta") or 0) for item in fasttext_runs
            ),
            "decomposed_lexeme_delta": int(
                (polyglot_run.get("status") or {}).get("decomposed_lexeme_delta") or 0
            ),
            "fully_bridged": int((bridge_run.get("status") or {}).get("fully_bridged") or 0),
        }
        artifacts = _write_growth_report(repo_root, report)
        status_payload.update(
            {
                "status": "completed",
                "phase": "completed",
                "finished_at": now_iso(),
                "report_path": artifacts["latest_json"],
                "lexeme_delta": report["lexeme_delta"],
                "translation_delta": report["translation_delta"],
                "translation_relationship_delta": report["translation_relationship_delta"],
                "vector_delta": report["vector_delta"],
                "forge_local_term_delta": report["forge_local_term_delta"],
                "lexicon_term_delta": report["lexicon_term_delta"],
                "thesaurus_seed_term_count": report["thesaurus_seed_term_count"],
                "thesaurus_seed_terms": report["thesaurus_seed_terms"],
                "multilingual_seed_term_count": report["multilingual_seed_term_count"],
                "multilingual_seed_terms": report["multilingual_seed_terms"],
                "incomplete_seed_term_count": report["incomplete_seed_term_count"],
                "incomplete_seed_terms": report["incomplete_seed_terms"],
                "definition_seed_term_count": report["definition_seed_term_count"],
                "definition_seed_terms": report["definition_seed_terms"],
                "usage_seed_term_count": report["usage_seed_term_count"],
                "usage_seed_terms": report["usage_seed_terms"],
                "vector_seed_term_count": report["vector_seed_term_count"],
                "vector_seed_terms": report["vector_seed_terms"],
                "vector_seed_placeholder_count": report["vector_seed_placeholder_count"],
                "vector_seed_missing_pos_count": report["vector_seed_missing_pos_count"],
                "vector_seed_translation_thin_count": report["vector_seed_translation_thin_count"],
                "vector_seed_translation_thin_terms": report["vector_seed_translation_thin_terms"],
                "emotion_seed_term_count": report["emotion_seed_term_count"],
                "emotion_seed_terms": report["emotion_seed_terms"],
                "derivation_seed_term_count": report["derivation_seed_term_count"],
                "derivation_seed_terms": report["derivation_seed_terms"],
                "priority_queue": report["priority_queue"],
                "semantic_vector_delta": report["semantic_vector_delta"],
                "relationship_delta": report["relationship_delta"],
                "phonetic_delta": report["phonetic_delta"],
                "emotion_delta": report["emotion_delta"],
                "omw_translation_delta": report["omw_translation_delta"],
                "morpheme_link_delta": report["morpheme_link_delta"],
                "lexicon_language_count": report["lexicon_language_count"],
                "applied_delta": report["applied_delta"],
                "decomposed_lexeme_delta": report["decomposed_lexeme_delta"],
                "fully_bridged": report["fully_bridged"],
                "source_count": discovery.get("source_count", 0),
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
