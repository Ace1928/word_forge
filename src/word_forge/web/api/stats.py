"""Word Forge Web API — Statistics Endpoints.

Provides database summary statistics for the dashboard HUD:
total words, relationships, pronunciation coverage, emotion coverage,
language distribution, and relationship type distribution.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_web(request: Request) -> Any:
    """Retrieve the WebApp instance from FastAPI state."""
    return request.app.state.web


@router.get("/stats")
def get_stats(request: Request) -> Dict[str, Any]:
    """Return aggregate database statistics."""
    web = _get_web(request)

    with web.db_manager.get_connection() as conn:
        cursor = conn.cursor()

        # ── Counts ───────────────────────────────────────────────────
        cursor.execute("SELECT COUNT(*) FROM words")
        total_words = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM words WHERE is_stub = 0")
        enriched_words = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM words WHERE is_stub = 1")
        stub_words = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM relationships")
        total_relationships = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM pronunciations")
        total_pronunciations = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM phonemes")
        total_phonemes = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM word_emotion")
        total_emotion = cursor.fetchone()[0]

        # ── Distinct words with pronunciations ───────────────────────
        cursor.execute(
            "SELECT COUNT(DISTINCT word_id) FROM pronunciations"
        )
        words_with_pronunciation = cursor.fetchone()[0]

        # ── Pronunciation coverage % ─────────────────────────────────
        pron_coverage = (
            round(words_with_pronunciation / total_words * 100, 1)
            if total_words > 0
            else 0.0
        )
        emotion_coverage = (
            round(total_emotion / total_words * 100, 1)
            if total_words > 0
            else 0.0
        )

        # ── Relationship type distribution ───────────────────────────
        cursor.execute(
            """SELECT relationship_type, COUNT(*)
               FROM relationships
               GROUP BY relationship_type
               ORDER BY COUNT(*) DESC"""
        )
        rel_types: List[Dict[str, Any]] = [
            {"type": row[0], "count": row[1]} for row in cursor.fetchall()
        ]

        # ── Language distribution ────────────────────────────────────
        cursor.execute(
            """SELECT language, COUNT(*)
               FROM words
               GROUP BY language
               ORDER BY COUNT(*) DESC
               LIMIT 20"""
        )
        languages: List[Dict[str, Any]] = [
            {"language": row[0], "count": row[1]} for row in cursor.fetchall()
        ]

        # ── Graph stats ──────────────────────────────────────────────
        graph_nodes = (
            web.graph_manager.graph.number_of_nodes()
            if web.graph_manager.graph
            else 0
        )
        graph_edges = (
            web.graph_manager.graph.number_of_edges()
            if web.graph_manager.graph
            else 0
        )

    return {
        "total_words": total_words,
        "enriched_words": enriched_words,
        "stub_words": stub_words,
        "total_relationships": total_relationships,
        "total_pronunciations": total_pronunciations,
        "total_phonemes": total_phonemes,
        "total_emotion": total_emotion,
        "words_with_pronunciation": words_with_pronunciation,
        "pronunciation_coverage_pct": pron_coverage,
        "emotion_coverage_pct": emotion_coverage,
        "relationship_types": rel_types,
        "languages": languages,
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
    }
