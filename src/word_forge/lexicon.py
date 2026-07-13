"""Friendly, local-first public API for a multilingual living lexicon.

Most users should start with :class:`Lexicon`.  It provides a small stable
surface over Word Forge's detailed persistence, graph, vector, and enrichment
subsystems.  Core operations use only SQLite and never require a model download
or an internet connection.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from word_forge.database.database_manager import DBManager, TermNotFoundError
from word_forge.models import (
    LexicalEntry,
    LexiconStatistics,
    NetworkSnapshot,
    Relationship,
    Sense,
)
from word_forge.network import build_network_snapshot

if TYPE_CHECKING:
    from word_forge.jobs import JobQueue

_LANGUAGE_PATTERN = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")
_MAX_TERM_LENGTH = 512
_MAX_LANGUAGE_LENGTH = 35


def _normalize_term(term: str) -> str:
    """Normalize a user term while preserving its language-specific spelling."""

    value = unicodedata.normalize("NFC", str(term)).strip()
    if not value:
        raise ValueError("term must not be empty")
    if "\x00" in value:
        raise ValueError("term must not contain a NUL character")
    if len(value) > _MAX_TERM_LENGTH:
        raise ValueError(f"term must contain at most {_MAX_TERM_LENGTH} characters")
    return value


def _normalize_language(language: str) -> str:
    """Validate and normalize a compact BCP-47-style language tag."""

    value = str(language).strip().replace("_", "-")
    if not value:
        raise ValueError("language must not be empty")
    if len(value) > _MAX_LANGUAGE_LENGTH or not _LANGUAGE_PATTERN.fullmatch(value):
        raise ValueError("language must be a tag such as 'en', 'es', 'zh-Hans', or 'pt-BR'")
    components = value.split("-")
    return "-".join([components[0].lower(), *(item.lower() for item in components[1:])])


def _normalize_examples(examples: Iterable[str] | None) -> list[str]:
    """Return non-empty, de-duplicated examples in their original order."""

    result: list[str] = []
    seen: set[str] = set()
    for raw_example in examples or ():
        example = unicodedata.normalize("NFC", str(raw_example)).strip()
        key = example.casefold()
        if example and key not in seen:
            seen.add(key)
            result.append(example)
    return result


def _like_pattern(query: str) -> str:
    """Escape SQLite LIKE metacharacters and create a contains pattern."""

    escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


class Lexicon:
    """Open, query, and extend a Word Forge lexicon.

    Parameters
    ----------
    path:
        Optional SQLite file.  Omitting it uses Word Forge's cross-platform user
        data directory.  Passing a path is useful for portable collections,
        tests, and applications that manage several lexicons.

    Examples
    --------
    >>> from word_forge import Lexicon
    >>> with Lexicon("my-lexicon.sqlite") as lexicon:
    ...     lexicon.add("recursion", definition="A process that refers to itself.")
    ...     entry = lexicon.get("recursion")
    ...     print(entry.definition)
    A process that refers to itself.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._database = DBManager(db_path=path)
        self._closed = False

    @property
    def path(self) -> Path:
        """Return the SQLite file backing this lexicon."""

        return self._database.db_path

    @property
    def jobs(self) -> JobQueue:
        """Return a durable job queue sharing this lexicon's transaction store."""

        self._require_open()
        from word_forge.jobs import JobQueue

        return JobQueue(self._database)

    def __enter__(self) -> Lexicon:
        self._require_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("This Lexicon is closed; create or open a new Lexicon")

    def add(
        self,
        term: str,
        *,
        definition: str = "",
        part_of_speech: str = "",
        language: str = "en",
        examples: Iterable[str] | None = None,
        source: str = "manual",
        base_term: str = "",
        expected_revision: int | None = None,
    ) -> LexicalEntry:
        """Add or update an entry and return its stored representation.

        ``expected_revision`` enables optimistic concurrency. Use ``0`` when
        creating a record that must not already exist, or pass the revision
        returned by :meth:`get` when updating. A stale editor receives a clear
        conflict instead of silently overwriting newer work.
        """

        self._require_open()
        normalized_term = _normalize_term(term)
        normalized_language = _normalize_language(language)
        normalized_base = _normalize_term(base_term) if str(base_term).strip() else ""
        normalized_examples = _normalize_examples(examples)
        normalized_definition = unicodedata.normalize("NFC", str(definition)).strip()
        normalized_pos = str(part_of_speech).strip()
        normalized_source = str(source).strip()

        self._database.insert_or_update_lexeme(
            lemma=normalized_term,
            lang=normalized_language,
            part_of_speech=normalized_pos,
            gloss=normalized_definition,
            source=normalized_source,
            base_term=normalized_base,
            usage_examples=normalized_examples,
            expected_revision=expected_revision,
            sync_english_word=True,
        )

        if normalized_base and normalized_language != "en":
            if not self._database.word_exists(normalized_base):
                self.add(normalized_base, language="en", source=normalized_source)
            lexeme_id = self._database.get_lexeme_id(normalized_term, normalized_language)
            self._database.add_translation(
                lexeme_id,
                target_lang="en",
                target_term=normalized_base,
                source=normalized_source,
            )
        return self.get(normalized_term, language=normalized_language)

    def get(self, term: str, *, language: str = "en") -> LexicalEntry:
        """Return one exact entry or raise :class:`TermNotFoundError`."""

        self._require_open()
        normalized_term = _normalize_term(term)
        normalized_language = _normalize_language(language)

        if normalized_language == "en" and self._database.word_exists(normalized_term):
            record = self._database.get_word_entry(normalized_term)
            lexeme_rows = self._database.execute_query(
                """
                SELECT id, uid, normalized_lemma, script, source, base_term,
                       revision, confidence, review_state, created_at, updated_at
                FROM lexemes
                WHERE lemma = ? AND lang = ? AND deleted_at IS NULL
                LIMIT 1
                """,
                (normalized_term, normalized_language),
            )
            lexeme = lexeme_rows[0] if lexeme_rows else None
            word_metadata = self._database.execute_query(
                """
                SELECT uid, normalized_term, script, revision, confidence,
                       review_state, created_at, updated_at
                FROM words WHERE id = ? AND deleted_at IS NULL
                """,
                (record["id_int"],),
            )[0]
            relationship_rows = self._database.execute_query(
                """
                SELECT uid, related_term, relationship_type, revision,
                       source, confidence
                FROM relationships
                WHERE word_id = ? AND deleted_at IS NULL
                ORDER BY relationship_type, related_term
                """,
                (record["id_int"],),
            )
            translation_rows = (
                self._database.execute_query(
                    """
                    SELECT uid, target_term, target_lang, relation, revision,
                           source, confidence
                    FROM translations
                    WHERE lexeme_id = ? AND deleted_at IS NULL
                    ORDER BY target_lang, target_term
                    """,
                    (lexeme["id"],),
                )
                if lexeme is not None
                else []
            )
            identity = lexeme if lexeme is not None else word_metadata
            entry_uid = str(identity["uid"])
            return LexicalEntry(
                id=entry_uid,
                term=record["term"],
                language="en",
                normalized_term=str(
                    identity["normalized_lemma"]
                    if lexeme is not None
                    else identity["normalized_term"]
                ),
                script=str(identity["script"]),
                definition=record["definition"],
                part_of_speech=record["part_of_speech"],
                examples=tuple(record["usage_examples"]),
                relationships=tuple(
                    Relationship(
                        id=str(item["uid"]),
                        target=str(item["related_term"]),
                        kind=str(item["relationship_type"]),
                        revision=int(item["revision"]),
                        source=str(item["source"] or ""),
                        confidence=float(item["confidence"]),
                    )
                    for item in relationship_rows
                )
                + tuple(
                    Relationship(
                        id=str(item["uid"]),
                        target=str(item["target_term"]),
                        kind=str(item["relation"]),
                        language=str(item["target_lang"]),
                        revision=int(item["revision"]),
                        source=str(item["source"] or ""),
                        confidence=float(item["confidence"]),
                    )
                    for item in translation_rows
                ),
                senses=self._get_senses(entry_uid),
                source=str(lexeme["source"] or "") if lexeme is not None else "",
                base_term=str(lexeme["base_term"] or "") if lexeme is not None else "",
                revision=int(identity["revision"]),
                confidence=float(identity["confidence"]),
                review_state=str(identity["review_state"]),
                created_at=float(identity["created_at"]),
                updated_at=float(identity["updated_at"]),
                last_refreshed=record["last_refreshed"],
            )

        rows = self._database.execute_query(
            """
            SELECT id, uid, lemma, normalized_lemma, lang, script, part_of_speech,
                   gloss, base_term, source, revision, confidence, review_state,
                   created_at, updated_at, last_refreshed
            FROM lexemes
            WHERE lemma = ? AND lang = ? AND deleted_at IS NULL
            LIMIT 1
            """,
            (normalized_term, normalized_language),
        )
        if not rows:
            raise TermNotFoundError(f"{normalized_term} ({normalized_language})")
        row = rows[0]
        translations = self._database.execute_query(
            """
            SELECT uid, target_term, target_lang, relation, revision, source, confidence
            FROM translations
            WHERE lexeme_id = ? AND deleted_at IS NULL
            ORDER BY target_lang, target_term
            """,
            (row["id"],),
        )
        return LexicalEntry(
            id=str(row["uid"]),
            term=str(row["lemma"]),
            language=str(row["lang"]),
            normalized_term=str(row["normalized_lemma"]),
            script=str(row["script"]),
            definition=str(row["gloss"] or ""),
            part_of_speech=str(row["part_of_speech"] or ""),
            relationships=tuple(
                Relationship(
                    id=str(item["uid"]),
                    target=str(item["target_term"]),
                    kind=str(item["relation"]),
                    language=str(item["target_lang"]),
                    revision=int(item["revision"]),
                    source=str(item["source"] or ""),
                    confidence=float(item["confidence"]),
                )
                for item in translations
            ),
            senses=self._get_senses(str(row["uid"])),
            source=str(row["source"] or ""),
            base_term=str(row["base_term"] or ""),
            revision=int(row["revision"]),
            confidence=float(row["confidence"]),
            review_state=str(row["review_state"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            last_refreshed=float(row["last_refreshed"] or 0.0),
        )

    def _get_senses(self, lexeme_uid: str) -> tuple[Sense, ...]:
        """Return active senses for one immutable lexical identity."""

        rows = self._database.execute_query(
            """
            SELECT s.id, s.definition, s.part_of_speech, s.domain,
                   s.register_name, COALESCE(src.name, '') AS source,
                   s.confidence, s.review_state, s.revision
            FROM senses s
            LEFT JOIN sources src ON src.id = s.source_id
            WHERE s.lexeme_uid = ? AND s.deleted_at IS NULL
            ORDER BY s.part_of_speech, s.domain, s.definition, s.id
            """,
            (lexeme_uid,),
        )
        return tuple(
            Sense(
                id=str(row["id"]),
                definition=str(row["definition"]),
                part_of_speech=str(row["part_of_speech"]),
                domain=str(row["domain"]),
                register=str(row["register_name"]),
                source=str(row["source"]),
                confidence=float(row["confidence"]),
                review_state=str(row["review_state"]),
                revision=int(row["revision"]),
            )
            for row in rows
        )

    def find(
        self,
        query: str,
        *,
        language: str | None = None,
        limit: int = 20,
    ) -> list[LexicalEntry]:
        """Find entries whose term or definition contains a text fragment."""

        self._require_open()
        normalized_query = _normalize_term(query)
        if not 1 <= limit <= 1_000:
            raise ValueError("limit must be between 1 and 1000")
        normalized_language = _normalize_language(language) if language else None
        pattern = _like_pattern(normalized_query)
        search_watermark = self._database.invalidation_watermark("search")
        matches: list[tuple[str, str]] = []

        if normalized_language in {None, "en"}:
            word_rows = self._database.execute_query(
                """
                SELECT term
                FROM words
                WHERE deleted_at IS NULL
                  AND (term LIKE ? ESCAPE '\\' OR definition LIKE ? ESCAPE '\\')
                ORDER BY
                    CASE
                        WHEN term = ? THEN 0
                        WHEN term LIKE ? ESCAPE '\\' THEN 1
                        ELSE 2
                    END,
                    length(term),
                    term
                LIMIT ?
                """,
                (pattern, pattern, normalized_query, f"{normalized_query}%", limit),
            )
            matches.extend((str(row["term"]), "en") for row in word_rows)

        if normalized_language != "en":
            if normalized_language is None:
                language_clause = "lang <> 'en'"
                parameters: tuple[object, ...] = (pattern, pattern, limit)
            else:
                language_clause = "lang = ?"
                parameters = (pattern, pattern, normalized_language, limit)
            lexeme_rows = self._database.execute_query(
                f"""
                SELECT lemma, lang
                FROM lexemes
                WHERE (lemma LIKE ? ESCAPE '\\' OR gloss LIKE ? ESCAPE '\\')
                  AND {language_clause}
                  AND deleted_at IS NULL
                ORDER BY length(lemma), lemma, lang
                LIMIT ?
                """,
                parameters,
            )
            matches.extend((str(row["lemma"]), str(row["lang"])) for row in lexeme_rows)

        entries: list[LexicalEntry] = []
        seen: set[tuple[str, str]] = set()
        for matched_term, matched_language in matches:
            key = (matched_term, matched_language)
            if key in seen:
                continue
            seen.add(key)
            entries.append(self.get(matched_term, language=matched_language))
            if len(entries) >= limit:
                break
        self._database.complete_invalidations("search", through_id=search_watermark)
        return entries

    def relate(self, term: str, target: str, *, kind: str = "related") -> bool:
        """Create a typed relationship between two base-language terms."""

        self._require_open()
        normalized_term = _normalize_term(term)
        normalized_target = _normalize_term(target)
        normalized_kind = str(kind).strip().casefold().replace(" ", "_")
        if not normalized_kind:
            raise ValueError("kind must not be empty")
        if not self._database.word_exists(normalized_target):
            self.add(normalized_target, language="en", source="relationship")
        return self._database.insert_relationship(
            normalized_term, normalized_target, normalized_kind
        )

    def translate(
        self,
        term: str,
        *,
        source_language: str,
        target_language: str,
        target_term: str,
        source: str = "manual",
        bidirectional: bool = True,
    ) -> None:
        """Add a translation, optionally creating the reverse mapping too."""

        self._require_open()
        source_term = _normalize_term(term)
        translated_term = _normalize_term(target_term)
        source_lang = _normalize_language(source_language)
        target_lang = _normalize_language(target_language)
        if source_lang == target_lang and source_term == translated_term:
            raise ValueError("a translation must change the term or language")

        for lexical_term, lang in ((source_term, source_lang), (translated_term, target_lang)):
            try:
                self._database.get_lexeme_id(lexical_term, lang)
            except TermNotFoundError:
                self.add(lexical_term, language=lang, source=source)

        source_id = self._database.get_lexeme_id(source_term, source_lang)
        self._database.add_translation(
            source_id,
            target_lang=target_lang,
            target_term=translated_term,
            source=source,
        )
        if bidirectional:
            target_id = self._database.get_lexeme_id(translated_term, target_lang)
            self._database.add_translation(
                target_id,
                target_lang=source_lang,
                target_term=source_term,
                source=source,
            )

    def add_sense(
        self,
        term: str,
        definition: str,
        *,
        language: str = "en",
        part_of_speech: str = "",
        domain: str = "",
        register: str = "",
        source: str = "manual",
        confidence: float = 1.0,
        review_state: str = "unreviewed",
        sense_id: str | None = None,
        expected_revision: int | None = None,
    ) -> Sense:
        """Create or revise one independently reviewable meaning."""

        self._require_open()
        entry = self.get(_normalize_term(term), language=_normalize_language(language))
        normalized_definition = unicodedata.normalize("NFC", str(definition)).strip()
        identifier = self._database.upsert_sense(
            lexeme_uid=entry.id,
            definition=normalized_definition,
            part_of_speech=str(part_of_speech).strip(),
            domain=str(domain).strip(),
            register=str(register).strip(),
            source=str(source).strip(),
            confidence=confidence,
            review_state=str(review_state).strip().casefold(),
            sense_id=sense_id,
            expected_revision=expected_revision,
        )
        refreshed = self.get(entry.term, language=entry.language)
        return next(sense for sense in refreshed.senses if sense.id == identifier)

    def remove_sense(
        self,
        sense_id: str,
        *,
        expected_revision: int | None = None,
    ) -> bool:
        """Tombstone a sense while retaining its history for audit and recovery."""

        self._require_open()
        return self._database.delete_sense(
            str(sense_id).strip(), expected_revision=expected_revision
        )

    def statistics(self) -> LexiconStatistics:
        """Return exact record counts without loading full tables into memory."""

        self._require_open()
        query = self._database.execute_scalar
        return LexiconStatistics(
            words=int(query("SELECT COUNT(*) FROM words WHERE deleted_at IS NULL") or 0),
            lexemes=int(query("SELECT COUNT(*) FROM lexemes WHERE deleted_at IS NULL") or 0),
            languages=int(
                query("SELECT COUNT(DISTINCT lang) FROM lexemes WHERE deleted_at IS NULL") or 0
            ),
            relationships=int(
                query("SELECT COUNT(*) FROM relationships WHERE deleted_at IS NULL") or 0
            ),
            translations=int(
                query("SELECT COUNT(*) FROM translations WHERE deleted_at IS NULL") or 0
            ),
        )

    def network(
        self,
        *,
        query: str = "",
        language: str | None = None,
        limit: int = 400,
    ) -> NetworkSnapshot:
        """Return a bounded, live-rendering-friendly semantic graph snapshot."""

        self._require_open()
        normalized_language = _normalize_language(language) if language else None
        graph_watermark = self._database.invalidation_watermark("graph")
        snapshot = build_network_snapshot(
            self._database,
            query=str(query).strip(),
            language=normalized_language,
            limit=limit,
        )
        self._database.complete_invalidations("graph", through_id=graph_watermark)
        return snapshot

    def close(self) -> None:
        """Release pooled SQLite connections; calling this twice is safe."""

        if not self._closed:
            self._database.close()
            self._closed = True


__all__ = ["Lexicon"]
