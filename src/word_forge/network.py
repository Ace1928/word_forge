"""Efficient bounded graph projections for the live Word Forge interface."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import Any

from word_forge.database.database_manager import DBManager
from word_forge.models import NetworkEdge, NetworkNode, NetworkSnapshot


def _node_id(term: str, language: str) -> str:
    digest = hashlib.blake2b(f"{language}\0{term}".encode("utf-8"), digest_size=12).hexdigest()
    return f"n-{digest}"


def _edge_id(source: str, target: str, kind: str) -> str:
    digest = hashlib.blake2b(
        f"{source}\0{target}\0{kind}".encode("utf-8"), digest_size=12
    ).hexdigest()
    return f"e-{digest}"


def _chunks(values: Sequence[Any], size: int = 400) -> Iterable[Sequence[Any]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def _candidate_rows(
    database: DBManager,
    *,
    query: str,
    language: str | None,
    limit: int,
) -> list[Any]:
    escaped_query = query.replace("\\", r"\\").replace("%", r"\%").replace("_", r"\_")
    search = f"%{escaped_query}%"
    clauses: list[str] = []
    parameters: list[Any] = []
    if query:
        clauses.append("(lemma LIKE ? ESCAPE '\\' OR gloss LIKE ? ESCAPE '\\')")
        parameters.extend([search, search])
    if language:
        clauses.append("lang = ?")
        parameters.append(language)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    parameters.append(limit + 1)
    return database.execute_query(
        f"""
        SELECT id AS source_id, uid, lemma AS term, lang AS language,
               gloss AS definition, part_of_speech, last_refreshed
        FROM lexemes
        {where}
          {"AND" if where else "WHERE"} deleted_at IS NULL
        ORDER BY last_refreshed DESC, term, language
        LIMIT ?
        """,
        tuple(parameters),
    )


def build_network_snapshot(
    database: DBManager,
    *,
    query: str = "",
    language: str | None = None,
    limit: int = 400,
) -> NetworkSnapshot:
    """Build a compact graph without loading the full lexicon into memory."""

    if not 1 <= limit <= 2_000:
        raise ValueError("limit must be between 1 and 2000")
    rows = _candidate_rows(
        database,
        query=query.strip(),
        language=language,
        limit=limit,
    )
    truncated = len(rows) > limit
    selected = rows[:limit]
    nodes: dict[tuple[str, str], NetworkNode] = {}
    source_word_ids: list[int] = []
    source_lexeme_ids: list[int] = []

    for row in selected:
        term = str(row["term"])
        lang = str(row["language"])
        nodes[(lang, term)] = NetworkNode(
            id=str(row["uid"]),
            term=term,
            language=lang,
            definition=str(row["definition"] or ""),
            part_of_speech=str(row["part_of_speech"] or ""),
        )
        source_id = int(row["source_id"])
        source_lexeme_ids.append(source_id)

    # Semantic relationships retain their legacy integer word foreign key;
    # resolve it in one bounded query while graph identity remains the lexeme UUID.
    english_terms = [term for lang, term in nodes if lang == "en"]
    for term_chunk in _chunks(english_terms):
        placeholders = ",".join("?" for _ in term_chunk)
        if placeholders:
            rows_for_terms = database.execute_query(
                f"""
                SELECT id FROM words
                WHERE deleted_at IS NULL AND term IN ({placeholders})
                """,
                tuple(term_chunk),
            )
            source_word_ids.extend(int(row["id"]) for row in rows_for_terms)

    edge_specs: list[tuple[str, str, str, str, str]] = []
    for id_chunk in _chunks(source_word_ids):
        placeholders = ",".join("?" for _ in id_chunk)
        if not placeholders:
            continue
        rows_for_ids = database.execute_query(
            f"""
            SELECT w.term AS source_term, r.related_term AS target_term,
                   r.relationship_type AS kind
            FROM relationships r
            JOIN words w ON w.id = r.word_id
            WHERE w.id IN ({placeholders}) AND r.deleted_at IS NULL
            """,
            tuple(id_chunk),
        )
        edge_specs.extend(
            (str(row["source_term"]), "en", str(row["target_term"]), "en", str(row["kind"]))
            for row in rows_for_ids
        )

    for id_chunk in _chunks(source_lexeme_ids):
        placeholders = ",".join("?" for _ in id_chunk)
        if not placeholders:
            continue
        rows_for_ids = database.execute_query(
            f"""
            SELECT l.lemma AS source_term, l.lang AS source_language,
                   t.target_term, t.target_lang AS target_language,
                   t.relation AS kind
            FROM translations t
            JOIN lexemes l ON l.id = t.lexeme_id
            WHERE l.id IN ({placeholders}) AND t.deleted_at IS NULL
            """,
            tuple(id_chunk),
        )
        edge_specs.extend(
            (
                str(row["source_term"]),
                str(row["source_language"]),
                str(row["target_term"]),
                str(row["target_language"]),
                str(row["kind"]),
            )
            for row in rows_for_ids
        )

    target_records: dict[tuple[str, str], Any] = {}
    targets_by_language: dict[str, list[str]] = {}
    for _, _, target_term, target_lang, _ in edge_specs:
        if (target_lang, target_term) not in nodes:
            targets_by_language.setdefault(target_lang, []).append(target_term)
    for target_language, terms in targets_by_language.items():
        unique_terms = sorted(set(terms))
        for term_chunk in _chunks(unique_terms):
            placeholders = ",".join("?" for _ in term_chunk)
            if not placeholders:
                continue
            records = database.execute_query(
                f"""
                SELECT uid, lemma, lang, gloss, part_of_speech
                FROM lexemes
                WHERE lang = ? AND deleted_at IS NULL
                  AND lemma IN ({placeholders})
                """,
                (target_language, *term_chunk),
            )
            for record in records:
                target_records[(str(record["lang"]), str(record["lemma"]))] = record

    edges: dict[str, NetworkEdge] = {}
    degrees: dict[tuple[str, str], int] = {key: 0 for key in nodes}
    for source_term, source_lang, target_term, target_lang, kind in edge_specs:
        source_key = (source_lang, source_term)
        target_key = (target_lang, target_term)
        if source_key not in nodes:
            continue
        if target_key not in nodes:
            if len(nodes) >= limit:
                truncated = True
                continue
            target_record = target_records.get(target_key)
            nodes[target_key] = NetworkNode(
                id=(
                    str(target_record["uid"])
                    if target_record is not None
                    else _node_id(target_term, target_lang)
                ),
                term=target_term,
                language=target_lang,
                definition=(str(target_record["gloss"] or "") if target_record is not None else ""),
                part_of_speech=(
                    str(target_record["part_of_speech"] or "") if target_record is not None else ""
                ),
            )
            degrees[target_key] = 0
        source_node_id = nodes[source_key].id
        target_node_id = nodes[target_key].id
        identifier = _edge_id(source_node_id, target_node_id, kind)
        edges[identifier] = NetworkEdge(
            id=identifier,
            source=source_node_id,
            target=target_node_id,
            kind=kind,
        )
        degrees[source_key] += 1
        degrees[target_key] += 1

    weighted_nodes = tuple(
        replace(node, weight=max(1, degrees[key]))
        for key, node in sorted(nodes.items(), key=lambda item: (item[0][0], item[0][1].casefold()))
    )
    return NetworkSnapshot(
        nodes=weighted_nodes,
        edges=tuple(sorted(edges.values(), key=lambda edge: edge.id)),
        truncated=truncated,
    )


__all__ = ["build_network_snapshot"]
