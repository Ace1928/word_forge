"""Transactional persistence for normalized, source-distinct lexical records."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple, cast

from word_forge.database.database_manager import DatabaseError, DBManager
from word_forge.lexicon.records import (
    GlossKind,
    LexicalEntryRecord,
    LexicalExampleRecord,
    LexicalFormRecord,
    LexicalGlossRecord,
    LexicalPronunciationRecord,
    LexicalRelationRecord,
    LexicalSenseRecord,
    SourceSnapshot,
    canonical_json,
)
from word_forge.parser.linguistics import normalize_term, segment_graphemes


class LexicalRepositoryError(DatabaseError):
    """Raised when normalized lexical persistence cannot be completed safely."""


class LexicalEntryNotFoundError(LexicalRepositoryError):
    """Raised when a normalized lexical entry does not exist."""


@dataclass(frozen=True, slots=True)
class LexicalWriteReport:
    """Counts produced by one atomic lexical-entry batch."""

    attempted: int
    inserted: int
    updated: int
    forms: int
    senses: int
    glosses: int
    examples: int
    pronunciations: int
    relations: int


class LexicalRepository:
    """Store complete source entries while maintaining the legacy word facade."""

    def __init__(self, database: DBManager) -> None:
        self.database = database

    def register_snapshot(self, snapshot: SourceSnapshot) -> int:
        """Register provenance idempotently and return its persistent identifier."""

        digest = snapshot.artifact_sha256 or ""
        now = time.time()
        try:
            with self.database.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO source_snapshots (
                        source_id, source_version, source_url, retrieved_at,
                        artifact_sha256, artifact_bytes, license_name,
                        license_url, attribution, importer_version,
                        metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_id, source_version, artifact_sha256)
                    DO UPDATE SET
                        source_url=excluded.source_url,
                        retrieved_at=excluded.retrieved_at,
                        artifact_bytes=excluded.artifact_bytes,
                        license_name=excluded.license_name,
                        license_url=excluded.license_url,
                        attribution=excluded.attribution,
                        importer_version=excluded.importer_version,
                        metadata_json=excluded.metadata_json
                    """,
                    (
                        snapshot.source_id,
                        snapshot.source_version,
                        snapshot.source_url,
                        snapshot.retrieved_at,
                        digest,
                        snapshot.artifact_bytes,
                        snapshot.license_name,
                        snapshot.license_url,
                        snapshot.attribution,
                        snapshot.importer_version,
                        snapshot.metadata_json,
                        now,
                    ),
                )
                row = connection.execute(
                    """
                    SELECT id FROM source_snapshots
                    WHERE source_id = ? AND source_version = ?
                      AND artifact_sha256 = ?
                    """,
                    (snapshot.source_id, snapshot.source_version, digest),
                ).fetchone()
                if row is None:  # pragma: no cover - SQLite invariant
                    raise LexicalRepositoryError(
                        f"Snapshot {snapshot.source_id!r} could not be retrieved"
                    )
                return int(row[0])
        except LexicalRepositoryError:
            raise
        except DatabaseError as exc:
            raise LexicalRepositoryError(
                f"Failed to register source snapshot {snapshot.source_id!r}", exc
            ) from exc

    def upsert_entries(
        self, snapshot_id: int, entries: Iterable[LexicalEntryRecord]
    ) -> LexicalWriteReport:
        """Atomically insert or replace a batch of complete lexical entries."""

        records = tuple(entries)
        self._validate_batch(records)
        if not records:
            return LexicalWriteReport(0, 0, 0, 0, 0, 0, 0, 0, 0)

        inserted = 0
        updated = 0
        forms = 0
        senses = 0
        glosses = 0
        examples = 0
        pronunciations = 0
        relations = 0
        now = time.time()

        try:
            with self.database.transaction() as connection:
                source_id = self._snapshot_source_id(connection, snapshot_id)
                for record in records:
                    existed, entry_id, word_id = self._upsert_entry_header(
                        connection, snapshot_id, source_id, record, now
                    )
                    if existed:
                        updated += 1
                    else:
                        inserted += 1
                    self._clear_entry_children(connection, entry_id)
                    form_ids = self._insert_forms(connection, entry_id, record.forms)
                    sense_ids = self._insert_senses(connection, entry_id, record.senses)
                    self._insert_pronunciations(
                        connection,
                        entry_id,
                        record.pronunciations,
                        form_ids,
                    )
                    self._insert_relations(
                        connection,
                        entry_id,
                        word_id,
                        source_id,
                        record,
                        sense_ids,
                    )
                    self._ensure_graphemes(connection, word_id, record.lemma)

                    forms += len(record.forms)
                    senses += len(record.senses)
                    glosses += sum(len(sense.glosses) for sense in record.senses)
                    examples += sum(len(sense.examples) for sense in record.senses)
                    pronunciations += len(record.pronunciations)
                    relations += len(record.relations)
        except LexicalRepositoryError:
            raise
        except DatabaseError as exc:
            raise LexicalRepositoryError(
                f"Failed to persist lexical batch for snapshot {snapshot_id}", exc
            ) from exc

        return LexicalWriteReport(
            attempted=len(records),
            inserted=inserted,
            updated=updated,
            forms=forms,
            senses=senses,
            glosses=glosses,
            examples=examples,
            pronunciations=pronunciations,
            relations=relations,
        )

    def get_entry(self, entry_id: int) -> LexicalEntryRecord:
        """Reconstruct one complete normalized entry by database identifier."""

        rows = self.database.execute_query(
            "SELECT * FROM lexical_entries WHERE id = ?", (entry_id,)
        )
        if not rows:
            raise LexicalEntryNotFoundError(f"Lexical entry {entry_id} was not found")
        row = rows[0]
        form_rows = self.database.execute_query(
            "SELECT * FROM lexical_forms WHERE entry_id = ? ORDER BY position, id",
            (entry_id,),
        )
        forms = tuple(
            LexicalFormRecord(
                form=str(item["form"]),
                language=str(item["language"]),
                position=int(item["position"]),
                source_form_id=str(item["source_form_id"]),
                script=str(item["script"]),
                features=_json_string_tuple(item["features_json"]),
                tags=_json_string_tuple(item["tags_json"]),
            )
            for item in form_rows
        )
        form_sources = {
            int(item["id"]): str(item["source_form_id"]) for item in form_rows
        }

        sense_rows = self.database.execute_query(
            "SELECT * FROM lexical_senses WHERE entry_id = ? ORDER BY position, id",
            (entry_id,),
        )
        senses: List[LexicalSenseRecord] = []
        sense_sources: Dict[int, str] = {}
        for item in sense_rows:
            sense_id = int(item["id"])
            source_sense_id = str(item["source_sense_id"])
            sense_sources[sense_id] = source_sense_id
            gloss_rows = self.database.execute_query(
                """
                SELECT * FROM lexical_glosses
                WHERE sense_id = ? ORDER BY position, id
                """,
                (sense_id,),
            )
            example_rows = self.database.execute_query(
                """
                SELECT * FROM lexical_examples
                WHERE sense_id = ? ORDER BY position, id
                """,
                (sense_id,),
            )
            senses.append(
                LexicalSenseRecord(
                    source_sense_id=source_sense_id,
                    position=int(item["position"]),
                    glosses=tuple(
                        LexicalGlossRecord(
                            text=str(gloss["text"]),
                            language=str(gloss["language"]),
                            kind=cast(GlossKind, str(gloss["kind"])),
                            generated=bool(gloss["generated"]),
                        )
                        for gloss in gloss_rows
                    ),
                    examples=tuple(
                        LexicalExampleRecord(
                            text=str(example["text"]),
                            language=str(example["language"]),
                            source_example_id=str(example["source_example_id"]),
                            translation=str(example["translation"]),
                            translation_language=str(example["translation_language"]),
                            reference=str(example["reference"]),
                            generated=bool(example["generated"]),
                        )
                        for example in example_rows
                    ),
                    concept_id=str(item["concept_id"]),
                    tags=_json_string_tuple(item["tags_json"]),
                    metadata_json=str(item["metadata_json"]),
                    confidence=float(item["confidence"]),
                    generated=bool(item["generated"]),
                )
            )

        pronunciation_rows = self.database.execute_query(
            """
            SELECT * FROM lexical_pronunciations
            WHERE entry_id = ? ORDER BY position, id
            """,
            (entry_id,),
        )
        pronunciations = tuple(
            LexicalPronunciationRecord(
                transcription=str(item["transcription"]),
                notation=str(item["notation"]),
                language=str(item["language"]),
                position=int(item["position"]),
                source_record_id=str(item["source_record_id"]),
                form_source_id=(
                    form_sources.get(int(item["form_id"]), "")
                    if item["form_id"] is not None
                    else ""
                ),
                dialect=str(item["dialect"]),
                tags=_json_string_tuple(item["tags_json"]),
                audio_url=str(item["audio_url"]),
                confidence=float(item["confidence"]),
                generated=bool(item["generated"]),
            )
            for item in pronunciation_rows
        )

        relation_rows = self.database.execute_query(
            """
            SELECT * FROM lexical_relations
            WHERE entry_id = ? ORDER BY position, id
            """,
            (entry_id,),
        )
        relations = tuple(
            LexicalRelationRecord(
                relationship_type=str(item["relationship_type"]),
                target_term=str(item["target_term"]),
                target_language=str(item["target_language"]),
                position=int(item["position"]),
                source_sense_id=(
                    sense_sources.get(int(item["sense_id"]))
                    if item["sense_id"] is not None
                    else None
                ),
                target_source_entry_id=str(item["target_source_entry_id"]),
                source_record_id=str(item["source_record_id"]),
                confidence=float(item["confidence"]),
            )
            for item in relation_rows
        )

        return LexicalEntryRecord(
            source_entry_id=str(row["source_entry_id"]),
            lemma=str(row["lemma"]),
            language=str(row["language"]),
            part_of_speech=str(row["part_of_speech"]),
            lexical_category=str(row["lexical_category"]),
            script=str(row["script"]),
            etymology=str(row["etymology"]),
            tags=_json_string_tuple(row["tags_json"]),
            metadata_json=str(row["metadata_json"]),
            confidence=float(row["confidence"]),
            generated=bool(row["generated"]),
            forms=forms,
            senses=tuple(senses),
            pronunciations=pronunciations,
            relations=relations,
        )

    def get_entry_by_source(
        self, snapshot_id: int, source_entry_id: str
    ) -> LexicalEntryRecord:
        """Return an entry by its reproducible source identity."""

        row = self.database.execute_query(
            """
            SELECT id FROM lexical_entries
            WHERE snapshot_id = ? AND source_entry_id = ?
            """,
            (snapshot_id, source_entry_id),
        )
        if not row:
            raise LexicalEntryNotFoundError(
                f"Source entry {source_entry_id!r} was not found in snapshot "
                f"{snapshot_id}"
            )
        return self.get_entry(int(row[0]["id"]))

    def find_entry_ids(
        self, term: str, language: str, *, limit: int = 100
    ) -> Tuple[int, ...]:
        """Find normalized entry identifiers for a term-language identity."""

        if limit <= 0:
            raise ValueError("limit must be positive")
        from word_forge.parser.linguistics import canonicalize_language_tag

        rows = self.database.execute_query(
            """
            SELECT id FROM lexical_entries
            WHERE normalized_lemma = ? AND language = ?
            ORDER BY id LIMIT ?
            """,
            (normalize_term(term), canonicalize_language_tag(language), limit),
        )
        return tuple(int(row["id"]) for row in rows)

    @staticmethod
    def _validate_batch(records: Sequence[LexicalEntryRecord]) -> None:
        source_ids = [record.source_entry_id for record in records]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_entry_id values must be unique within a batch")

    @staticmethod
    def _snapshot_source_id(connection: sqlite3.Connection, snapshot_id: int) -> str:
        row = connection.execute(
            "SELECT source_id FROM source_snapshots WHERE id = ?", (snapshot_id,)
        ).fetchone()
        if row is None:
            raise LexicalRepositoryError(
                f"Source snapshot {snapshot_id} does not exist"
            )
        return str(row[0])

    @staticmethod
    def _upsert_entry_header(
        connection: sqlite3.Connection,
        snapshot_id: int,
        source_id: str,
        record: LexicalEntryRecord,
        now: float,
    ) -> Tuple[bool, int, int]:
        existing = connection.execute(
            """
            SELECT id FROM lexical_entries
            WHERE snapshot_id = ? AND source_entry_id = ?
            """,
            (snapshot_id, record.source_entry_id),
        ).fetchone()
        definition = _first_definition(record)
        examples = _example_texts(record)
        normalized = normalize_term(record.lemma)
        connection.execute(
            """
            INSERT INTO words (
                term, normalized_term, language, script, definition,
                part_of_speech, usage_examples, source, is_stub,
                last_refreshed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            ON CONFLICT(normalized_term, language) DO UPDATE SET
                term=excluded.term,
                script=excluded.script,
                definition=CASE WHEN excluded.definition <> ''
                    THEN excluded.definition ELSE words.definition END,
                part_of_speech=CASE WHEN excluded.part_of_speech <> ''
                    THEN excluded.part_of_speech ELSE words.part_of_speech END,
                usage_examples=CASE WHEN excluded.usage_examples <> ''
                    THEN excluded.usage_examples ELSE words.usage_examples END,
                source=excluded.source,
                is_stub=0,
                last_refreshed=excluded.last_refreshed
            """,
            (
                record.lemma,
                normalized,
                record.language,
                record.script,
                definition,
                record.part_of_speech,
                "\n".join(examples),
                source_id,
                now,
            ),
        )
        word_row = connection.execute(
            "SELECT id FROM words WHERE normalized_term = ? AND language = ?",
            (normalized, record.language),
        ).fetchone()
        if word_row is None:  # pragma: no cover - SQLite invariant
            raise LexicalRepositoryError(f"Facade word {record.lemma!r} is missing")
        word_id = int(word_row[0])

        connection.execute(
            """
            INSERT INTO lexical_entries (
                word_id, snapshot_id, source_entry_id, lemma,
                normalized_lemma, language, script, part_of_speech,
                lexical_category, etymology, tags_json, metadata_json,
                confidence, generated, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(snapshot_id, source_entry_id) DO UPDATE SET
                word_id=excluded.word_id,
                lemma=excluded.lemma,
                normalized_lemma=excluded.normalized_lemma,
                language=excluded.language,
                script=excluded.script,
                part_of_speech=excluded.part_of_speech,
                lexical_category=excluded.lexical_category,
                etymology=excluded.etymology,
                tags_json=excluded.tags_json,
                metadata_json=excluded.metadata_json,
                confidence=excluded.confidence,
                generated=excluded.generated,
                updated_at=excluded.updated_at
            """,
            (
                word_id,
                snapshot_id,
                record.source_entry_id,
                record.lemma,
                normalized,
                record.language,
                record.script,
                record.part_of_speech,
                record.lexical_category,
                record.etymology,
                canonical_json(record.tags),
                record.metadata_json,
                record.confidence,
                int(record.generated),
                now,
                now,
            ),
        )
        entry_row = connection.execute(
            """
            SELECT id FROM lexical_entries
            WHERE snapshot_id = ? AND source_entry_id = ?
            """,
            (snapshot_id, record.source_entry_id),
        ).fetchone()
        if entry_row is None:  # pragma: no cover - SQLite invariant
            raise LexicalRepositoryError(
                f"Lexical entry {record.source_entry_id!r} is missing after upsert"
            )
        return existing is not None, int(entry_row[0]), word_id

    @staticmethod
    def _clear_entry_children(connection: sqlite3.Connection, entry_id: int) -> None:
        connection.execute(
            "DELETE FROM lexical_relations WHERE entry_id = ?", (entry_id,)
        )
        connection.execute(
            "DELETE FROM lexical_pronunciations WHERE entry_id = ?", (entry_id,)
        )
        connection.execute("DELETE FROM lexical_senses WHERE entry_id = ?", (entry_id,))
        connection.execute("DELETE FROM lexical_forms WHERE entry_id = ?", (entry_id,))

    @staticmethod
    def _insert_forms(
        connection: sqlite3.Connection,
        entry_id: int,
        forms: Sequence[LexicalFormRecord],
    ) -> Dict[str, int]:
        form_ids: Dict[str, int] = {}
        for form in forms:
            cursor = connection.execute(
                """
                INSERT INTO lexical_forms (
                    entry_id, source_form_id, position, form,
                    normalized_form, language, script, features_json,
                    tags_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry_id,
                    form.source_form_id,
                    form.position,
                    form.form,
                    normalize_term(form.form),
                    form.language,
                    form.script,
                    canonical_json(form.features),
                    canonical_json(form.tags),
                ),
            )
            if form.source_form_id:
                form_ids[form.source_form_id] = _last_row_id(cursor, "lexical form")
        return form_ids

    @staticmethod
    def _insert_senses(
        connection: sqlite3.Connection,
        entry_id: int,
        senses: Sequence[LexicalSenseRecord],
    ) -> Dict[str, int]:
        sense_ids: Dict[str, int] = {}
        for sense in senses:
            cursor = connection.execute(
                """
                INSERT INTO lexical_senses (
                    entry_id, source_sense_id, position, concept_id,
                    tags_json, metadata_json, confidence, generated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry_id,
                    sense.source_sense_id,
                    sense.position,
                    sense.concept_id,
                    canonical_json(sense.tags),
                    sense.metadata_json,
                    sense.confidence,
                    int(sense.generated),
                ),
            )
            sense_id = _last_row_id(cursor, "lexical sense")
            sense_ids[sense.source_sense_id] = sense_id
            connection.executemany(
                """
                INSERT INTO lexical_glosses (
                    sense_id, position, text, language, kind, generated
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        sense_id,
                        position,
                        gloss.text,
                        gloss.language,
                        gloss.kind,
                        int(gloss.generated),
                    )
                    for position, gloss in enumerate(sense.glosses)
                ],
            )
            connection.executemany(
                """
                INSERT INTO lexical_examples (
                    sense_id, source_example_id, position, text, language,
                    translation, translation_language, reference, generated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        sense_id,
                        example.source_example_id,
                        position,
                        example.text,
                        example.language,
                        example.translation,
                        example.translation_language,
                        example.reference,
                        int(example.generated),
                    )
                    for position, example in enumerate(sense.examples)
                ],
            )
        return sense_ids

    @staticmethod
    def _insert_pronunciations(
        connection: sqlite3.Connection,
        entry_id: int,
        pronunciations: Sequence[LexicalPronunciationRecord],
        form_ids: Dict[str, int],
    ) -> None:
        connection.executemany(
            """
            INSERT INTO lexical_pronunciations (
                entry_id, form_id, source_record_id, position, notation,
                transcription, language, dialect, tags_json, audio_url,
                confidence, generated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    entry_id,
                    form_ids.get(pronunciation.form_source_id),
                    pronunciation.source_record_id,
                    pronunciation.position,
                    pronunciation.notation,
                    pronunciation.transcription,
                    pronunciation.language,
                    pronunciation.dialect,
                    canonical_json(pronunciation.tags),
                    pronunciation.audio_url,
                    pronunciation.confidence,
                    int(pronunciation.generated),
                )
                for pronunciation in pronunciations
            ],
        )

    @staticmethod
    def _insert_relations(
        connection: sqlite3.Connection,
        entry_id: int,
        word_id: int,
        source_id: str,
        record: LexicalEntryRecord,
        sense_ids: Dict[str, int],
    ) -> None:
        for relation in record.relations:
            sense_id = (
                sense_ids[relation.source_sense_id]
                if relation.source_sense_id is not None
                else None
            )
            scope_key = (
                f"sense:{relation.source_sense_id}"
                if relation.source_sense_id is not None
                else "entry"
            )
            target_normalized = normalize_term(relation.target_term)
            connection.execute(
                """
                INSERT INTO lexical_relations (
                    entry_id, sense_id, scope_key, source_record_id,
                    position, relationship_type, target_term,
                    target_normalized_term, target_language,
                    target_source_entry_id, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry_id,
                    sense_id,
                    scope_key,
                    relation.source_record_id,
                    relation.position,
                    relation.relationship_type,
                    relation.target_term,
                    target_normalized,
                    relation.target_language,
                    relation.target_source_entry_id,
                    relation.confidence,
                ),
            )
            if not (
                normalize_term(record.lemma) == target_normalized
                and record.language == relation.target_language
            ):
                connection.execute(
                    """
                    INSERT OR IGNORE INTO relationships (
                        word_id, related_term, related_normalized_term,
                        related_language, relationship_type, source,
                        confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        word_id,
                        relation.target_term,
                        target_normalized,
                        relation.target_language,
                        relation.relationship_type,
                        source_id,
                        relation.confidence,
                    ),
                )

    @staticmethod
    def _ensure_graphemes(
        connection: sqlite3.Connection, word_id: int, lemma: str
    ) -> None:
        exists = connection.execute(
            "SELECT 1 FROM graphemes WHERE word_id = ? LIMIT 1", (word_id,)
        ).fetchone()
        if exists is not None:
            return
        connection.executemany(
            """
            INSERT INTO graphemes (
                word_id, position, text, normalized, codepoints,
                unicode_names, categories, combining_classes, script
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    word_id,
                    grapheme.position,
                    grapheme.text,
                    grapheme.normalized,
                    canonical_json(grapheme.codepoints),
                    canonical_json(grapheme.unicode_names),
                    canonical_json(grapheme.categories),
                    canonical_json(grapheme.combining_classes),
                    grapheme.script,
                )
                for grapheme in segment_graphemes(lemma)
            ],
        )


def _first_definition(record: LexicalEntryRecord) -> str:
    for sense in sorted(record.senses, key=lambda item: item.position):
        for gloss in sense.glosses:
            if gloss.kind in {"definition", "gloss"}:
                return gloss.text
    return ""


def _example_texts(record: LexicalEntryRecord) -> Tuple[str, ...]:
    values = []
    for sense in sorted(record.senses, key=lambda item: item.position):
        for example in sense.examples:
            if example.text not in values:
                values.append(example.text)
    return tuple(values)


def _json_string_tuple(value: object) -> Tuple[str, ...]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:  # pragma: no cover - database corruption
        raise LexicalRepositoryError(f"Invalid persisted JSON array: {exc}") from exc
    if not isinstance(parsed, list) or not all(
        isinstance(item, str) for item in parsed
    ):
        raise LexicalRepositoryError("Persisted value is not a string array")
    return tuple(parsed)


def _last_row_id(cursor: sqlite3.Cursor, context: str) -> int:
    """Return SQLite's generated identifier or raise an invariant error."""

    if cursor.lastrowid is None:  # pragma: no cover - SQLite invariant
        raise LexicalRepositoryError(f"SQLite did not return an id for {context}")
    return int(cursor.lastrowid)


__all__ = [
    "LexicalEntryNotFoundError",
    "LexicalRepository",
    "LexicalRepositoryError",
    "LexicalWriteReport",
]
