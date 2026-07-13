from __future__ import annotations

import json
import sqlite3
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from word_forge.database.database_manager import DBManager
from word_forge.linguistics.g2p_manager import G2PManager, LLMG2PProvider
from word_forge.linguistics.phonetics_manager import PhoneticsManager
from word_forge.multilingual.runtime import append_history, now_iso, write_json
from word_forge.parser.language_model import ModelState
from word_forge.parser.lexical_functions import (
    get_openthesaurus_data,
    get_synsets,
    get_wordnet_data,
)
from word_forge.parser.parser_config import ParserConfig

REPORT_DIR_NAME = "word_forge_thesaurus"
STATUS_FILE_NAME = "word_forge_thesaurus_status.json"
HISTORY_FILE_NAME = "word_forge_thesaurus_history.jsonl"
_WORDNET_TERM_RE = __import__("re").compile(r"^[a-z][a-z'-]{1,31}$")
_POS_LABELS = {
    "n": "noun",
    "v": "verb",
    "a": "adjective",
    "s": "adjective",
    "r": "adverb",
}


def _report_dir(repo_root: Path) -> Path:
    return repo_root / "reports" / REPORT_DIR_NAME


def _status_path(repo_root: Path) -> Path:
    return repo_root / "data" / "runtime" / STATUS_FILE_NAME


def _history_path(repo_root: Path) -> Path:
    return repo_root / "data" / "runtime" / HISTORY_FILE_NAME


def _dedupe_strings(values: Iterable[str], *, limit: int | None = None) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append(normalized)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def _normalize_term(term: str) -> str:
    return " ".join(str(term or "").replace("_", " ").strip().lower().split())


def _is_wordnet_candidate(term: str) -> bool:
    normalized = _normalize_term(term)
    return bool(normalized and _WORDNET_TERM_RE.match(normalized))


def _should_enrich_definition(term: str, definition: str) -> bool:
    normalized_term = _normalize_term(term)
    normalized_definition = _normalize_term(definition)
    return not normalized_definition or normalized_definition == normalized_term


def _coerce_seed_terms(terms: Iterable[str] | None, *, limit: int | None = None) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for value in terms or []:
        for token in str(value or "").replace("\n", ",").split(","):
            normalized = _normalize_term(token)
            if not normalized or not _is_wordnet_candidate(normalized):
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append(normalized)
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def _term_counts(db_path: Path) -> dict[str, int]:
    if not db_path.exists():
        return {"words": 0, "relationships": 0, "phonetics": 0}
    with sqlite3.connect(str(db_path)) as conn:
        word_count = int(conn.execute("SELECT COUNT(*) FROM words").fetchone()[0])
        relationship_count = int(conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0])
        phonetic_count = int(conn.execute("SELECT COUNT(*) FROM phonetics").fetchone()[0])
    return {
        "words": word_count,
        "relationships": relationship_count,
        "phonetics": phonetic_count,
    }


def _select_candidate_terms(db_path: Path, *, limit: int) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    bounded_limit = max(1, min(int(limit), 120))
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                w.term,
                w.definition,
                w.part_of_speech,
                w.last_refreshed,
                COALESCE(rel.rel_count, 0) AS relationship_count,
                COALESCE(ph.phonetic_count, 0) AS phonetic_count
            FROM words w
            LEFT JOIN (
                SELECT word_id, COUNT(*) AS rel_count
                FROM relationships
                GROUP BY word_id
            ) rel ON rel.word_id = w.id
            LEFT JOIN (
                SELECT word_id, COUNT(*) AS phonetic_count
                FROM phonetics
                GROUP BY word_id
            ) ph ON ph.word_id = w.id
            ORDER BY
                CASE WHEN COALESCE(rel.rel_count, 0) = 0 THEN 0 ELSE 1 END ASC,
                CASE WHEN COALESCE(w.definition, '') = '' THEN 0 ELSE 1 END ASC,
                CASE WHEN COALESCE(w.part_of_speech, '') = '' THEN 0 ELSE 1 END ASC,
                CASE WHEN COALESCE(ph.phonetic_count, 0) = 0 THEN 0 ELSE 1 END ASC,
                w.last_refreshed DESC,
                w.term ASC
            LIMIT ?
            """,
            (bounded_limit * 4,),
        ).fetchall()
    candidates: list[dict[str, Any]] = []
    for row in rows:
        term = _normalize_term(str(row["term"] or ""))
        if not _is_wordnet_candidate(term):
            continue
        candidates.append(
            {
                "term": term,
                "definition": str(row["definition"] or ""),
                "part_of_speech": str(row["part_of_speech"] or ""),
                "relationship_count": int(row["relationship_count"] or 0),
                "phonetic_count": int(row["phonetic_count"] or 0),
                "last_refreshed": float(row["last_refreshed"] or 0),
            }
        )
        if len(candidates) >= bounded_limit:
            break
    return candidates


def _seed_terms_from_reports(repo_root: Path, db_path: Path, *, limit: int) -> list[str]:
    candidates: list[str] = [
        "archive",
        "atlas",
        "eidos",
        "eidosian",
        "embedding",
        "graphrag",
        "integration",
        "memory",
        "ollama",
        "pipeline",
        "proof",
        "provenance",
        "registry",
        "termux",
        "workflow",
    ]
    for path in (
        repo_root / "reports" / "word_forge_bridge_audit" / "latest.json",
        repo_root / "reports" / "word_forge_growth" / "latest.json",
    ):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if path.name == "latest.json" and "top_bridged_terms" in payload:
            for row in payload.get("top_bridged_terms", []) or []:
                if isinstance(row, dict):
                    candidates.append(str(row.get("term") or ""))
        forge_local = (
            payload.get("forge_local_run")
            if isinstance(payload.get("forge_local_run"), dict)
            else {}
        )
        for term in forge_local.get("inserted_terms", []) or []:
            candidates.append(str(term or ""))
    existing_terms: set[str] = set()
    if db_path.exists():
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute("SELECT lower(term) FROM words").fetchall()
        existing_terms = {str(row[0] or "") for row in rows}
    seeded: list[str] = []
    seen: set[str] = set()
    for term in candidates:
        normalized = _normalize_term(term)
        if (
            not _is_wordnet_candidate(normalized)
            or normalized not in existing_terms
            or normalized in seen
        ):
            continue
        seen.add(normalized)
        seeded.append(normalized)
        if len(seeded) >= max(1, limit):
            break
    return seeded


def _semantic_related_terms(
    term: str, *, limit: int = 4, vector_store: Any | None = None
) -> list[dict[str, Any]]:
    normalized = _normalize_term(term)
    if not normalized:
        return []
    try:
        if vector_store is None:
            from word_forge.configs.config_essentials import StorageType
            from word_forge.vectorizer.vector_store import VectorStore

            vector_store = VectorStore(storage_type=StorageType.DISK)
        results = vector_store.search(
            query_text=normalized,
            k=max(4, int(limit) + 4),
            filter_metadata={"content_type": "word"},
        )
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in results:
            metadata = row.get("metadata") or {}
            related_term = _normalize_term(str(metadata.get("term") or ""))
            if (
                not related_term
                or related_term == normalized
                or related_term in seen
                or not _is_wordnet_candidate(related_term)
            ):
                continue
            score = round(max(0.0, 1.0 - float(row.get("distance") or 1.0)), 6)
            if score < 0.42:
                continue
            seen.add(related_term)
            rows.append(
                {
                    "term": related_term,
                    "score": score,
                    "definition": str(metadata.get("definition") or ""),
                    "part_of_speech": str(
                        metadata.get("part_of_speech") or metadata.get("pos") or ""
                    ),
                }
            )
            if len(rows) >= max(1, int(limit)):
                break
        return rows
    except Exception:
        return []


_OMW_LANGUAGE_CODES = ("spa", "fra", "deu", "ita", "por", "jpn", "lat")


def _omw_translation_terms(term: str, *, limit_per_language: int = 4) -> dict[str, list[str]]:
    normalized = _normalize_term(term)
    if not _is_wordnet_candidate(normalized):
        return {}
    rows: dict[str, list[str]] = {}
    for synset in get_synsets(normalized):
        for lang in _OMW_LANGUAGE_CODES:
            try:
                lemma_names = synset.lemma_names(lang)
            except Exception:
                lemma_names = []
            cleaned = [
                value
                for value in (_normalize_term(item) for item in lemma_names)
                if _is_wordnet_candidate(value) and value != normalized
            ]
            if cleaned:
                existing = rows.setdefault(lang, [])
                for item in cleaned:
                    if item not in existing:
                        existing.append(item)
                        if len(existing) >= max(1, int(limit_per_language)):
                            break
    return {
        lang: values[: max(1, int(limit_per_language))] for lang, values in rows.items() if values
    }


def _wordnet_derivational_terms(term: str, *, limit: int = 8) -> dict[str, list[str]]:
    normalized = _normalize_term(term)
    if not _is_wordnet_candidate(normalized):
        return {"derived_from": [], "etymological_source": []}
    derived_from: list[str] = []
    etymological_source: list[str] = []
    seen_derived: set[str] = set()
    seen_source: set[str] = set()
    for synset in get_synsets(normalized):
        try:
            lemmas = synset.lemmas() or []
        except Exception:
            lemmas = []
        for lemma in lemmas:
            lemma_name = _normalize_term(str(lemma.name() or "").replace("_", " "))
            try:
                related_forms = lemma.derivationally_related_forms() or []
            except Exception:
                related_forms = []
            for related in related_forms:
                related_name = _normalize_term(str(related.name() or "").replace("_", " "))
                if not _is_wordnet_candidate(related_name) or related_name == normalized:
                    continue
                if lemma_name == normalized:
                    if related_name not in seen_source:
                        seen_source.add(related_name)
                        etymological_source.append(related_name)
                elif related_name == normalized:
                    if (
                        lemma_name
                        and lemma_name not in seen_derived
                        and _is_wordnet_candidate(lemma_name)
                    ):
                        seen_derived.add(lemma_name)
                        derived_from.append(lemma_name)
                else:
                    if (
                        lemma_name
                        and lemma_name != normalized
                        and lemma_name not in seen_derived
                        and _is_wordnet_candidate(lemma_name)
                    ):
                        seen_derived.add(lemma_name)
                        derived_from.append(lemma_name)
                    if related_name not in seen_source:
                        seen_source.add(related_name)
                        etymological_source.append(related_name)
                if len(derived_from) >= max(1, int(limit)) and len(etymological_source) >= max(
                    1, int(limit)
                ):
                    break
            if len(derived_from) >= max(1, int(limit)) and len(etymological_source) >= max(
                1, int(limit)
            ):
                break
    return {
        "derived_from": derived_from[: max(1, int(limit))],
        "etymological_source": etymological_source[: max(1, int(limit))],
    }


def _read_local_thesaurus_synonyms(term: str) -> list[str]:
    config = ParserConfig()
    synonyms: list[str] = []
    for resource_name in ("openthesaurus", "thesaurus"):
        try:
            if not config.resource_exists(resource_name):
                continue
            path = config.get_full_resource_path(resource_name)
            synonyms.extend(get_openthesaurus_data(term, str(path)))
        except Exception:
            continue
    return _dedupe_strings(synonyms, limit=24)


def _aggregate_wordnet_relationships(term: str) -> dict[str, Any]:
    entries = get_wordnet_data(term)
    synsets = get_synsets(term)
    definitions: list[str] = []
    examples: list[str] = []
    pos_labels: list[str] = []
    relationships: dict[str, list[str]] = {
        "synonym": [],
        "antonym": [],
        "hypernym": [],
        "hyponym": [],
        "derived_from": [],
        "etymological_source": [],
    }
    for entry in entries:
        definitions.append(str(entry.get("definition") or ""))
        examples.extend(str(example or "") for example in (entry.get("examples") or []))
        raw_pos = str(entry.get("part_of_speech") or "").strip().lower()
        if raw_pos:
            pos_labels.append(_POS_LABELS.get(raw_pos, raw_pos))
        relationships["synonym"].extend(str(item or "") for item in (entry.get("synonyms") or []))
        relationships["antonym"].extend(str(item or "") for item in (entry.get("antonyms") or []))
    for synset in synsets:
        try:
            for target in synset.hypernyms() or []:
                for lemma in target.lemmas() or []:
                    relationships["hypernym"].append(str(lemma.name() or "").replace("_", " "))
        except Exception:
            pass
        try:
            for target in synset.hyponyms() or []:
                for lemma in target.lemmas() or []:
                    relationships["hyponym"].append(str(lemma.name() or "").replace("_", " "))
        except Exception:
            pass
    derivational = _wordnet_derivational_terms(term, limit=8)
    relationships["derived_from"].extend(
        str(item or "") for item in (derivational.get("derived_from") or [])
    )
    relationships["etymological_source"].extend(
        str(item or "") for item in (derivational.get("etymological_source") or [])
    )
    relationships["synonym"].extend(_read_local_thesaurus_synonyms(term))
    cleaned_relationships = {
        relation: [
            item
            for item in _dedupe_strings((_normalize_term(value) for value in values), limit=16)
            if _is_wordnet_candidate(item) and item != term
        ]
        for relation, values in relationships.items()
    }
    pos_counter = Counter(label for label in pos_labels if label)
    best_pos = pos_counter.most_common(1)[0][0] if pos_counter else ""
    best_definitions = _dedupe_strings(definitions, limit=1)
    return {
        "wordnet_entry_count": len(entries),
        "synset_count": len(synsets),
        "definition": best_definitions[0] if best_definitions else "",
        "examples": _dedupe_strings(examples, limit=4),
        "part_of_speech": best_pos,
        "relationships": cleaned_relationships,
    }


def _merge_examples(existing: list[str], incoming: list[str], *, limit: int = 4) -> list[str]:
    return _dedupe_strings([*existing, *incoming], limit=limit)


def _word_has_phonetics(conn: sqlite3.Connection, term: str) -> bool:
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM phonetics p
        JOIN words w ON w.id = p.word_id
        WHERE lower(w.term) = ?
        """,
        (_normalize_term(term),),
    ).fetchone()
    return bool(int(row[0] or 0)) if row else False


def _ensure_word_record(
    db: DBManager,
    term: str,
    *,
    definition: str = "",
    part_of_speech: str = "",
    usage_examples: list[str] | None = None,
) -> bool:
    normalized = _normalize_term(term)
    if not normalized:
        return False
    exists = db.word_exists(normalized)
    if exists:
        return False
    db.insert_or_update_word(
        term=normalized,
        definition=definition,
        part_of_speech=part_of_speech,
        usage_examples=usage_examples or [],
    )
    return True


def render_thesaurus_markdown(report: dict[str, Any]) -> str:
    latest = report.get("deltas") or {}
    types = report.get("relationship_type_deltas") or {}
    lines = [
        "# Word Forge Thesaurus Enrichment",
        "",
        f"- Generated: `{report.get('generated_at')}`",
        f"- Candidate count: `{report.get('candidate_count', 0)}`",
        f"- Processed count: `{report.get('processed_count', 0)}`",
        f"- WordNet hits: `{report.get('wordnet_hit_count', 0)}`",
        f"- Related words inserted: `{report.get('inserted_related_word_count', 0)}`",
        f"- OMW language hits: `{report.get('omw_language_hit_count', 0)}`",
        f"- OMW translation delta: `{(report.get('deltas') or {}).get('omw_translation_delta', 0)}`",
        f"- Relationship delta: `{latest.get('relationship_delta', 0)}`",
        f"- Phonetic delta: `{latest.get('phonetic_delta', 0)}`",
        "",
        "## Relationship Deltas",
        "",
    ]
    lines.extend([f"- `{key}`: `{value}`" for key, value in sorted(types.items())] or ["- None"])
    lines.extend(["", "## Processed Terms", ""])
    processed = report.get("processed_terms") or []
    lines.extend(
        [
            f"- `{row.get('term')}`: rel+`{row.get('relationship_delta', 0)}` phon+`{row.get('phonetic_delta', 0)}`"
            for row in processed[:24]
            if isinstance(row, dict)
        ]
        or ["- None"]
    )
    return "\n".join(lines) + "\n"


def write_thesaurus_report(repo_root: Path, report: dict[str, Any]) -> dict[str, str]:
    report_dir = _report_dir(repo_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = report_dir / f"word_forge_thesaurus_{stamp}.json"
    md_path = report_dir / f"word_forge_thesaurus_{stamp}.md"
    latest_json = report_dir / "latest.json"
    latest_md = report_dir / "latest.md"
    write_json(json_path, report)
    md_path.write_text(render_thesaurus_markdown(report), encoding="utf-8")
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    return {
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "latest_json": str(latest_json),
        "latest_markdown": str(latest_md),
    }


def build_thesaurus_report(
    *,
    repo_root: Path,
    db_path: Path,
    limit: int | None = None,
    phonetic_limit: int | None = None,
    seed_terms: Iterable[str] | None = None,
    seed_source: str = "operator",
    apply: bool = True,
    use_llm_phonetics: bool = True,
) -> dict[str, Any]:
    bounded_limit = max(1, min(int(limit or 24), 120))
    bounded_phonetic_limit = max(0, min(int(phonetic_limit or 0), bounded_limit))
    before = _term_counts(db_path)
    auto_candidates = _select_candidate_terms(db_path, limit=bounded_limit)
    explicit_seed_terms = _coerce_seed_terms(seed_terms, limit=min(bounded_limit, 64))
    report_seed_terms = explicit_seed_terms or _seed_terms_from_reports(
        repo_root, db_path, limit=min(16, bounded_limit)
    )
    candidate_map: dict[str, dict[str, Any]] = {row["term"]: row for row in auto_candidates}
    for term in report_seed_terms:
        candidate_map.setdefault(
            term,
            {
                "term": term,
                "definition": "",
                "part_of_speech": "",
                "relationship_count": 0,
                "phonetic_count": 0,
                "last_refreshed": 0.0,
            },
        )
    ordered_terms = [term for term in report_seed_terms if term in candidate_map]
    ordered_terms.extend(row["term"] for row in auto_candidates if row["term"] not in ordered_terms)
    candidates = [candidate_map[term] for term in ordered_terms[:bounded_limit]]
    processed_terms: list[dict[str, Any]] = []
    inserted_related_word_count = 0
    relationship_type_deltas: Counter[str] = Counter()
    relationship_delta = 0
    phonetic_delta = 0
    wordnet_hit_count = 0
    semantic_neighbor_hit_count = 0
    semantic_related_delta = 0
    omw_language_hit_count = 0
    omw_translation_delta = 0
    llm_attempts = 0
    llm_error = ""
    db = DBManager(db_path=db_path)
    db.create_tables()
    phonetics = PhoneticsManager(db)
    model_state: ModelState | None = None
    g2p_manager: G2PManager | None = None
    semantic_vector_store: Any | None = None

    def _ensure_g2p_manager() -> G2PManager:
        nonlocal model_state, g2p_manager
        if g2p_manager is None:
            model_state = ModelState()
            g2p_manager = G2PManager(primary_provider=LLMG2PProvider(model_state))
        return g2p_manager

    def _ensure_semantic_vector_store() -> Any | None:
        nonlocal semantic_vector_store
        if semantic_vector_store is None:
            try:
                from word_forge.configs.config_essentials import StorageType
                from word_forge.vectorizer.vector_store import VectorStore

                semantic_vector_store = VectorStore(storage_type=StorageType.DISK)
            except Exception:
                semantic_vector_store = False
        return None if semantic_vector_store is False else semantic_vector_store

    try:
        with sqlite3.connect(str(db_path)) as conn:
            for candidate in candidates:
                term = str(candidate.get("term") or "").strip().lower()
                if not term:
                    continue
                current_definition = str(candidate.get("definition") or "")
                current_pos = str(candidate.get("part_of_speech") or "")
                current_examples: list[str] = []
                try:
                    if db.word_exists(term):
                        current_entry = db.get_word_entry(term)
                        current_examples = [
                            str(item or "") for item in (current_entry.get("usage_examples") or [])
                        ]
                except Exception:
                    current_examples = []
                aggregate = _aggregate_wordnet_relationships(term)
                if int(aggregate.get("wordnet_entry_count") or 0) > 0:
                    wordnet_hit_count += 1
                merged_definition = (
                    str(aggregate.get("definition") or "")
                    if _should_enrich_definition(term, current_definition)
                    else current_definition
                )
                merged_pos = current_pos or str(aggregate.get("part_of_speech") or "")
                merged_examples = _merge_examples(
                    current_examples,
                    [str(item or "") for item in (aggregate.get("examples") or [])],
                )
                term_relationship_delta = 0
                term_phonetic_delta = 0
                term_semantic_related_delta = 0
                term_omw_translation_delta = 0
                omw_translations: dict[str, list[str]] = {}
                related_insertions: list[str] = []
                if apply:
                    db.insert_or_update_word(
                        term=term,
                        definition=merged_definition,
                        part_of_speech=merged_pos,
                        usage_examples=merged_examples,
                    )
                    for relation_type, inverse_type in (
                        ("synonym", "synonym"),
                        ("antonym", "antonym"),
                        ("hypernym", "hyponym"),
                        ("hyponym", "hypernym"),
                        ("derived_from", "etymological_source"),
                        ("etymological_source", "derived_from"),
                    ):
                        for related_term in list(
                            (aggregate.get("relationships") or {}).get(relation_type, [])
                        ):
                            normalized_related = _normalize_term(str(related_term or ""))
                            if (
                                not _is_wordnet_candidate(normalized_related)
                                or normalized_related == term
                            ):
                                continue
                            related_aggregate = _aggregate_wordnet_relationships(normalized_related)
                            if _ensure_word_record(
                                db,
                                normalized_related,
                                definition=str(related_aggregate.get("definition") or ""),
                                part_of_speech=str(related_aggregate.get("part_of_speech") or ""),
                                usage_examples=[
                                    str(item or "")
                                    for item in (related_aggregate.get("examples") or [])
                                ],
                            ):
                                inserted_related_word_count += 1
                                related_insertions.append(normalized_related)
                            if db.insert_relationship(term, normalized_related, relation_type):
                                term_relationship_delta += 1
                                relationship_delta += 1
                                relationship_type_deltas[relation_type] += 1
                            if db.word_exists(normalized_related) and db.insert_relationship(
                                normalized_related, term, inverse_type
                            ):
                                term_relationship_delta += 1
                                relationship_delta += 1
                                relationship_type_deltas[inverse_type] += 1
                    omw_translations = _omw_translation_terms(term, limit_per_language=4)
                    if omw_translations:
                        omw_language_hit_count += len(omw_translations)
                    for lang, translation_terms in omw_translations.items():
                        for translated_term in translation_terms:
                            normalized_related = _normalize_term(str(translated_term or ""))
                            if (
                                not _is_wordnet_candidate(normalized_related)
                                or normalized_related == term
                            ):
                                continue
                            if _ensure_word_record(
                                db,
                                normalized_related,
                                definition=f"A {lang} lexical translation aligned to '{term}'.",
                                part_of_speech="",
                                usage_examples=[f"{lang}:{normalized_related}"],
                            ):
                                inserted_related_word_count += 1
                                related_insertions.append(normalized_related)
                            if db.insert_relationship(term, normalized_related, "translation"):
                                term_relationship_delta += 1
                                term_omw_translation_delta += 1
                                relationship_delta += 1
                                omw_translation_delta += 1
                                relationship_type_deltas["translation"] += 1
                            if db.word_exists(normalized_related) and db.insert_relationship(
                                normalized_related, term, "translation"
                            ):
                                term_relationship_delta += 1
                                term_omw_translation_delta += 1
                                relationship_delta += 1
                                omw_translation_delta += 1
                                relationship_type_deltas["translation"] += 1

                    existing_relationship_count = (
                        int(candidate.get("relationship_count") or 0) + term_relationship_delta
                    )
                    if existing_relationship_count < 4:
                        semantic_related = _semantic_related_terms(
                            term, limit=4, vector_store=_ensure_semantic_vector_store()
                        )
                        if semantic_related:
                            semantic_neighbor_hit_count += 1
                        existing_related_terms = {
                            _normalize_term(str(item or ""))
                            for rows in (aggregate.get("relationships") or {}).values()
                            if isinstance(rows, list)
                            for item in rows
                        }
                        for related_row in semantic_related:
                            normalized_related = _normalize_term(str(related_row.get("term") or ""))
                            if (
                                not normalized_related
                                or normalized_related == term
                                or normalized_related in existing_related_terms
                            ):
                                continue
                            if _ensure_word_record(
                                db,
                                normalized_related,
                                definition=str(related_row.get("definition") or ""),
                                part_of_speech=str(related_row.get("part_of_speech") or ""),
                                usage_examples=[],
                            ):
                                inserted_related_word_count += 1
                                related_insertions.append(normalized_related)
                            if db.insert_relationship(term, normalized_related, "related"):
                                term_relationship_delta += 1
                                term_semantic_related_delta += 1
                                relationship_delta += 1
                                semantic_related_delta += 1
                                relationship_type_deltas["related"] += 1
                            if db.word_exists(normalized_related) and db.insert_relationship(
                                normalized_related, term, "related"
                            ):
                                term_relationship_delta += 1
                                term_semantic_related_delta += 1
                                relationship_delta += 1
                                semantic_related_delta += 1
                                relationship_type_deltas["related"] += 1
                            existing_related_terms.add(normalized_related)
                    if bounded_phonetic_limit > phonetic_delta and not _word_has_phonetics(
                        conn, term
                    ):
                        try:
                            llm_attempts += 1
                            g2p = _ensure_g2p_manager() if use_llm_phonetics else G2PManager()
                            result = g2p.convert(
                                term, context=merged_definition or "Lexical thesaurus enrichment"
                            )
                            if result.get("ipa"):
                                phonetics.upsert_phonetics(
                                    db.get_word_id(term),
                                    ipa=str(result.get("ipa") or ""),
                                    arpabet=str(result.get("arpabet") or "") or None,
                                    stress_pattern=str(result.get("stress_pattern") or "") or None,
                                    source="word_forge_thesaurus",
                                    is_exception=False,
                                )
                                term_phonetic_delta += 1
                                phonetic_delta += 1
                        except Exception as exc:
                            if not llm_error:
                                llm_error = str(exc)
                processed_terms.append(
                    {
                        "term": term,
                        "wordnet_entry_count": int(aggregate.get("wordnet_entry_count") or 0),
                        "definition": merged_definition,
                        "part_of_speech": merged_pos,
                        "relationship_delta": term_relationship_delta,
                        "phonetic_delta": term_phonetic_delta,
                        "semantic_related_delta": term_semantic_related_delta,
                        "omw_translation_delta": term_omw_translation_delta,
                        "omw_languages": sorted(omw_translations),
                        "relationship_types": {
                            key: len(value)
                            for key, value in (aggregate.get("relationships") or {}).items()
                            if isinstance(value, list) and value
                        },
                        "related_insertions": related_insertions[:12],
                    }
                )
    finally:
        db.close()
    after = _term_counts(db_path)
    return {
        "contract": "eidos.word_forge.thesaurus.report.v1",
        "generated_at": now_iso(),
        "repo_root": str(repo_root),
        "db_path": str(db_path),
        "apply": apply,
        "limit": bounded_limit,
        "phonetic_limit": bounded_phonetic_limit,
        "use_llm_phonetics": bool(use_llm_phonetics),
        "llm_attempts": llm_attempts,
        "llm_error": llm_error,
        "candidate_count": len(candidates),
        "seed_terms": report_seed_terms,
        "seed_source": seed_source,
        "processed_count": len(processed_terms),
        "wordnet_hit_count": wordnet_hit_count,
        "semantic_neighbor_hit_count": semantic_neighbor_hit_count,
        "omw_language_hit_count": omw_language_hit_count,
        "inserted_related_word_count": inserted_related_word_count,
        "before": before,
        "after": after,
        "deltas": {
            "word_delta": after["words"] - before["words"],
            "relationship_delta": relationship_delta,
            "phonetic_delta": phonetic_delta,
            "semantic_related_delta": semantic_related_delta,
            "omw_translation_delta": omw_translation_delta,
        },
        "relationship_type_deltas": dict(sorted(relationship_type_deltas.items())),
        "processed_terms": processed_terms[:48],
    }


def run_word_forge_thesaurus(
    *,
    repo_root: Path,
    db_path: Path,
    limit: int | None = None,
    phonetic_limit: int | None = None,
    seed_terms: Iterable[str] | None = None,
    seed_source: str = "operator",
    force: bool = False,
    use_llm_phonetics: bool = True,
) -> dict[str, Any]:
    status_path = _status_path(repo_root)
    history_path = _history_path(repo_root)
    status_payload = {
        "contract": "eidos.word_forge.thesaurus.status.v1",
        "status": "running",
        "phase": "enriching",
        "repo_root": str(repo_root),
        "db_path": str(db_path),
        "started_at": now_iso(),
        "limit": limit,
        "phonetic_limit": phonetic_limit,
        "force": force,
        "use_llm_phonetics": bool(use_llm_phonetics),
        "seed_terms": _coerce_seed_terms(seed_terms, limit=32),
        "seed_source": seed_source,
    }
    write_json(status_path, status_payload)
    try:
        report = build_thesaurus_report(
            repo_root=repo_root,
            db_path=db_path,
            limit=limit,
            phonetic_limit=phonetic_limit,
            seed_terms=seed_terms,
            seed_source=seed_source,
            apply=True,
            use_llm_phonetics=use_llm_phonetics,
        )
        artifacts = write_thesaurus_report(repo_root, report)
        status_payload.update(
            {
                "status": "completed",
                "phase": "completed",
                "finished_at": now_iso(),
                "report_path": artifacts["latest_json"],
                "candidate_count": int(report.get("candidate_count") or 0),
                "processed_count": int(report.get("processed_count") or 0),
                "relationship_delta": int(
                    (report.get("deltas") or {}).get("relationship_delta") or 0
                ),
                "phonetic_delta": int((report.get("deltas") or {}).get("phonetic_delta") or 0),
                "semantic_related_delta": int(
                    (report.get("deltas") or {}).get("semantic_related_delta") or 0
                ),
                "omw_translation_delta": int(
                    (report.get("deltas") or {}).get("omw_translation_delta") or 0
                ),
                "inserted_related_word_count": int(report.get("inserted_related_word_count") or 0),
                "wordnet_hit_count": int(report.get("wordnet_hit_count") or 0),
                "semantic_neighbor_hit_count": int(report.get("semantic_neighbor_hit_count") or 0),
                "omw_language_hit_count": int(report.get("omw_language_hit_count") or 0),
                "llm_error": str(report.get("llm_error") or ""),
                "seed_terms": list(report.get("seed_terms") or []),
                "seed_source": str(report.get("seed_source") or seed_source),
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
