"""Word Forge Web API — Word Detail & Search Endpoints.

Provides rich word detail (definitions, pronunciations, phonemes, emotion,
relationships, graphemes, usage examples) and fuzzy search with autocomplete.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_web(request: Request) -> Any:
    """Retrieve the WebApp instance from FastAPI state."""
    return request.app.state.web


@router.get("/words/search")
def search_words(
    request: Request,
    q: str = Query("", min_length=1, description="Search query"),
    limit: int = Query(12, ge=1, le=50),
) -> Dict[str, Any]:
    """Fuzzy-prefix search over the lexicon with metadata hints.

    Returns matching terms with availability flags for pronunciation
    and emotion data, enabling rich autocomplete UI.
    """
    web = _get_web(request)

    with web.db_manager.get_connection() as conn:
        cursor = conn.cursor()
        # Prefix match on normalized_term, fall back to LIKE
        pattern = q.lower().replace("%", "").replace("_", "") + "%"
        cursor.execute(
            """
            SELECT
                w.id,
                w.term,
                w.language,
                w.part_of_speech,
                w.is_stub,
                (SELECT COUNT(*) FROM pronunciations p WHERE p.word_id = w.id) AS pron_count,
                (SELECT COUNT(*) FROM word_emotion e WHERE e.word_id = w.id) AS emo_count,
                (SELECT COUNT(*) FROM relationships r WHERE r.word_id = w.id) AS rel_count
            FROM words w
            WHERE w.normalized_term LIKE ?
            ORDER BY
                LENGTH(w.term) ASC,
                w.is_stub ASC,
                w.term ASC
            LIMIT ?
            """,
            (pattern, limit),
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "term": row[1],
                    "language": row[2],
                    "part_of_speech": row[3] or "",
                    "is_stub": bool(row[4]),
                    "has_pronunciation": row[5] > 0,
                    "has_emotion": row[6] > 0,
                    "relationship_count": row[7],
                }
            )

    return {"query": q, "count": len(results), "results": results}


@router.get("/words/{term}")
def get_word_detail(
    request: Request,
    term: str,
    language: str = Query("en", description="Language filter"),
) -> Dict[str, Any]:
    """Return the full enrichment profile for a single word.

    Includes: definition, part of speech, pronunciations with phoneme
    decomposition, emotion scores, relationships grouped by type,
    grapheme analysis, and usage examples.
    """
    web = _get_web(request)

    with web.db_manager.get_connection() as conn:
        cursor = conn.cursor()

        # ── Resolve word ─────────────────────────────────────────────
        cursor.execute(
            """SELECT id, term, normalized_term, language, script,
                      definition, part_of_speech, usage_examples,
                      source, is_stub, last_refreshed
               FROM words
               WHERE (term = ? OR normalized_term = ?)
                 AND language = ?
               LIMIT 1""",
            (term, term.lower(), language),
        )
        word_row = cursor.fetchone()

        if not word_row:
            # Try without language filter
            cursor.execute(
                """SELECT id, term, normalized_term, language, script,
                          definition, part_of_speech, usage_examples,
                          source, is_stub, last_refreshed
                   FROM words
                   WHERE term = ? OR normalized_term = ?
                   LIMIT 1""",
                (term, term.lower()),
            )
            word_row = cursor.fetchone()

        if not word_row:
            raise HTTPException(status_code=404, detail=f"Word '{term}' not found")

        word_id = word_row[0]
        usage_raw = word_row[7] or ""
        usage_examples = [ex.strip() for ex in usage_raw.split("\n") if ex.strip()]

        # ── Pronunciations + Phonemes ────────────────────────────────
        pronunciations = _fetch_pronunciations(cursor, word_id)

        # ── Emotion ──────────────────────────────────────────────────
        emotion = _fetch_emotion(cursor, word_id)

        # ── Relationships (grouped by type) ──────────────────────────
        relationships = _fetch_relationships(cursor, word_id)

        # ── Graphemes ────────────────────────────────────────────────
        graphemes = _fetch_graphemes(cursor, word_id)

        # ── Definition enrichment from lexical_glosses ───────────────
        definition = word_row[5] or ""
        if not definition or definition.startswith("Auto-generated"):
            gloss = _fetch_best_gloss(cursor, word_id)
            if gloss:
                definition = gloss

        return {
            "word_id": word_id,
            "term": word_row[1],
            "normalized_term": word_row[2],
            "language": word_row[3],
            "script": word_row[4],
            "definition": definition,
            "part_of_speech": word_row[6] or "",
            "usage_examples": usage_examples,
            "source": word_row[8],
            "is_stub": bool(word_row[9]),
            "pronunciations": pronunciations,
            "emotion": emotion,
            "relationships": relationships,
            "graphemes": graphemes,
        }


# ── Helper: Pronunciations ───────────────────────────────────────────


def _fetch_pronunciations(
    cursor: sqlite3.Cursor, word_id: int
) -> List[Dict[str, Any]]:
    """Fetch all pronunciations with nested phoneme decomposition."""
    cursor.execute(
        """SELECT id, notation, transcription, language, dialect,
                  source, confidence, generated, syllable_count, stress_pattern
           FROM pronunciations
           WHERE word_id = ?
           ORDER BY notation, dialect""",
        (word_id,),
    )

    pronunciations: List[Dict[str, Any]] = []
    for prow in cursor.fetchall():
        pron_id = prow[0]

        # Parse stress_pattern JSON
        try:
            stress_pattern = json.loads(prow[9]) if prow[9] else []
        except (json.JSONDecodeError, TypeError):
            stress_pattern = []

        # Fetch phonemes for this pronunciation
        cursor.execute(
            """SELECT position, symbol, base_symbol, stress, syllabic
               FROM phonemes
               WHERE pronunciation_id = ?
               ORDER BY position""",
            (pron_id,),
        )
        phonemes = [
            {
                "position": ph[0],
                "symbol": ph[1],
                "base_symbol": ph[2],
                "stress": ph[3],
                "syllabic": bool(ph[4]),
            }
            for ph in cursor.fetchall()
        ]

        pronunciations.append(
            {
                "id": pron_id,
                "notation": prow[1],
                "transcription": prow[2],
                "language": prow[3],
                "dialect": prow[4] or "",
                "source": prow[5],
                "confidence": prow[6],
                "generated": bool(prow[7]),
                "syllable_count": prow[8],
                "stress_pattern": stress_pattern,
                "phonemes": phonemes,
            }
        )

    return pronunciations


# ── Helper: Emotion ──────────────────────────────────────────────────


def _fetch_emotion(
    cursor: sqlite3.Cursor, word_id: int
) -> Optional[Dict[str, float]]:
    """Fetch valence/arousal emotion profile."""
    cursor.execute(
        "SELECT valence, arousal FROM word_emotion WHERE word_id = ?",
        (word_id,),
    )
    row = cursor.fetchone()
    if row:
        return {"valence": row[0], "arousal": row[1]}
    return None


# ── Helper: Relationships ────────────────────────────────────────────


def _fetch_relationships(
    cursor: sqlite3.Cursor, word_id: int
) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch relationships grouped by type."""
    cursor.execute(
        """SELECT related_term, related_normalized_term, related_language,
                  relationship_type, source, confidence
           FROM relationships
           WHERE word_id = ?
           ORDER BY relationship_type, confidence DESC""",
        (word_id,),
    )

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in cursor.fetchall():
        rel_type = row[3]
        if rel_type not in grouped:
            grouped[rel_type] = []
        grouped[rel_type].append(
            {
                "term": row[0],
                "normalized_term": row[1],
                "language": row[2],
                "source": row[4],
                "confidence": row[5],
            }
        )

    return grouped


# ── Helper: Graphemes ────────────────────────────────────────────────


def _fetch_graphemes(
    cursor: sqlite3.Cursor, word_id: int
) -> List[Dict[str, Any]]:
    """Fetch Unicode grapheme decomposition."""
    try:
        cursor.execute(
            """SELECT position, text, normalized, codepoints,
                      unicode_names, categories, combining_classes, script
               FROM graphemes
               WHERE word_id = ?
               ORDER BY position""",
            (word_id,),
        )
    except sqlite3.OperationalError:
        # Table may not exist in all DB versions
        return []

    graphemes: List[Dict[str, Any]] = []
    for row in cursor.fetchall():
        try:
            codepoints = json.loads(row[3]) if row[3] else []
            unicode_names = json.loads(row[4]) if row[4] else []
            categories = json.loads(row[5]) if row[5] else []
            combining = json.loads(row[6]) if row[6] else []
        except (json.JSONDecodeError, TypeError):
            codepoints = []
            unicode_names = []
            categories = []
            combining = []

        graphemes.append(
            {
                "position": row[0],
                "text": row[1],
                "normalized": row[2],
                "codepoints": codepoints,
                "unicode_names": unicode_names,
                "categories": categories,
                "combining_classes": combining,
                "script": row[7],
            }
        )

    return graphemes


# ── Helper: Best Gloss ───────────────────────────────────────────────


def _fetch_best_gloss(cursor: sqlite3.Cursor, word_id: int) -> Optional[str]:
    """Attempt to find a definition from lexical_glosses."""
    try:
        cursor.execute(
            """SELECT lg.text
               FROM lexical_senses ls
               JOIN lexical_entries le ON le.id = ls.entry_id
               JOIN lexical_glosses lg ON lg.sense_id = ls.id
               WHERE le.word_id = ?
                 AND lg.kind IN ('definition', 'gloss')
               ORDER BY ls.position, lg.position
               LIMIT 1""",
            (word_id,),
        )
        row = cursor.fetchone()
        return row[0] if row else None
    except sqlite3.OperationalError:
        return None
