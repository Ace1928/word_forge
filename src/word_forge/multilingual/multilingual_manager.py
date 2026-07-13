"""
Multilingual manager for Word Forge.

Maintains a base-language (English) alignment layer while ingesting
multilingual lexical resources.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from word_forge.database.database_manager import DBManager, TermNotFoundError

LOGGER = logging.getLogger("word_forge.multilingual")

SQL_GET_EXISTING_WORD = """
    SELECT definition, part_of_speech, usage_examples
    FROM words
    WHERE term = ?
    LIMIT 1
"""


def _normalize_examples(examples: Iterable[str] | None, *, limit: int = 6) -> list[str]:
    seen: set[str] = set()
    rows: list[str] = []
    for example in examples or []:
        text = str(example or "").strip()
        if not text:
            continue
        normalized = text.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        rows.append(text)
        if len(rows) >= max(1, int(limit or 6)):
            break
    return rows


def _is_placeholder_definition(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return True
    return (
        "base alignment imported for multilingual lexeme" in normalized
        or "translation linked to" in normalized
        or "lexical translation aligned to" in normalized
    )


class MultilingualManager:
    """Coordinates multilingual ingestion with English as the base language."""

    def __init__(self, db_manager: DBManager | None = None) -> None:
        self.db = db_manager or DBManager()

    def upsert_lexeme(
        self,
        lemma: str,
        lang: str,
        part_of_speech: str = "",
        gloss: str = "",
        source: str = "",
        base_term: str = "",
    ) -> int:
        """Insert/update a lexeme and return its ID."""
        return self.db.insert_or_update_lexeme(
            lemma=lemma,
            lang=lang,
            part_of_speech=part_of_speech,
            gloss=gloss,
            source=source,
            base_term=base_term,
        )

    def add_translation(
        self,
        lemma: str,
        lang: str,
        target_lang: str,
        target_term: str,
        relation: str = "translation",
        source: str = "",
    ) -> None:
        """Add translation mapping for a lexeme."""
        lexeme_id = self.db.get_lexeme_id(lemma=lemma, lang=lang)
        self.db.add_translation(
            lexeme_id=lexeme_id,
            target_lang=target_lang,
            target_term=target_term,
            relation=relation,
            source=source,
        )

    def align_to_english(self, lemma: str, lang: str, english_term: str) -> None:
        """Force-align a lexeme to an English base term."""
        try:
            self.db.update_lexeme_base(lemma=lemma, lang=lang, base_term=english_term)
        except TermNotFoundError:
            self.db.insert_or_update_lexeme(lemma=lemma, lang=lang, base_term=english_term)

    def ensure_base_word(
        self,
        english_term: str,
        *,
        definition: str = "",
        part_of_speech: str = "",
    ) -> None:
        """Ensure an English base term exists in the core Word Forge word table."""
        term = str(english_term or "").strip()
        if not term:
            return
        if self.db.word_exists(term):
            return
        definition_text = (
            definition.strip() or f"Base alignment imported for multilingual lexeme '{term}'."
        )
        self.db.insert_or_update_word(
            term=term,
            definition=definition_text,
            part_of_speech=part_of_speech or "",
            usage_examples=[],
        )

    def ensure_word_entry(
        self,
        term: str,
        *,
        definition: str = "",
        part_of_speech: str = "",
        usage_examples: Iterable[str] | None = None,
    ) -> None:
        """Ensure a public word-layer entry exists and merge stronger multilingual details."""
        normalized_term = str(term or "").strip()
        if not normalized_term:
            return
        merged_examples = _normalize_examples(usage_examples)
        merged_definition = str(definition or "").strip()
        merged_pos = str(part_of_speech or "").strip()
        existing = self._get_existing_word_payload(normalized_term)
        if existing is None:
            self.db.insert_or_update_word(
                term=normalized_term,
                definition=merged_definition,
                part_of_speech=merged_pos,
                usage_examples=merged_examples,
            )
            return

        existing_definition = str(existing.get("definition") or "").strip()
        existing_pos = str(existing.get("part_of_speech") or "").strip()
        existing_examples = [
            str(item).strip()
            for item in (existing.get("usage_examples") or [])
            if str(item).strip()
        ]

        final_definition = existing_definition
        if merged_definition and (
            _is_placeholder_definition(existing_definition) or not existing_definition
        ):
            final_definition = merged_definition

        final_pos = existing_pos or merged_pos
        final_examples = _normalize_examples([*existing_examples, *merged_examples], limit=8)

        self.db.insert_or_update_word(
            term=normalized_term,
            definition=final_definition,
            part_of_speech=final_pos,
            usage_examples=final_examples,
        )

    def _get_existing_word_payload(self, term: str) -> dict[str, Any] | None:
        """Return existing word-layer data without noisy expected-miss exceptions."""
        rows = self.db.execute_query(SQL_GET_EXISTING_WORD, (term,))
        if not rows:
            return None
        result = rows[0]
        return {
            "definition": str(result["definition"] or "").strip(),
            "part_of_speech": str(result["part_of_speech"] or "").strip(),
            "usage_examples": _normalize_examples(
                str(result["usage_examples"] or "").split("\n"), limit=8
            ),
        }

    def link_word_translation_pair(self, source_term: str, target_term: str) -> int:
        """Create bidirectional translation relationships for public word entries when possible."""
        left = str(source_term or "").strip().lower()
        right = str(target_term or "").strip().lower()
        if not left or not right or left == right:
            return 0
        created = 0
        if self.db.word_exists(left) and self.db.insert_relationship(left, right, "translation"):
            created += 1
        if self.db.word_exists(right) and self.db.insert_relationship(right, left, "translation"):
            created += 1
        return created

    def ingest_translation_batch(
        self,
        lemma: str,
        lang: str,
        translations: Iterable[dict[str, Any]],
        source: str = "",
    ) -> None:
        """Bulk add translations for a lexeme."""
        for entry in translations:
            target_lang = str(entry.get("lang") or "").strip()
            target_term = str(entry.get("term") or "").strip()
            relation = str(entry.get("relation") or "translation").strip()
            if not target_lang or not target_term:
                continue
            self.add_translation(
                lemma=lemma,
                lang=lang,
                target_lang=target_lang,
                target_term=target_term,
                relation=relation,
                source=source,
            )
