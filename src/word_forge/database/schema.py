"""Versioned SQLite schema and lossless lexical-data migrations."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

from word_forge.parser.linguistics import (
    LanguageTagError,
    canonicalize_language_tag,
    infer_script,
    normalize_term,
)

CURRENT_SCHEMA_VERSION = 2

SQL_CREATE_WORDS_TABLE = """
CREATE TABLE IF NOT EXISTS words (
    id INTEGER PRIMARY KEY,
    term TEXT NOT NULL,
    normalized_term TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'en',
    script TEXT NOT NULL DEFAULT 'Zzzz',
    definition TEXT NOT NULL DEFAULT '',
    part_of_speech TEXT NOT NULL DEFAULT '',
    usage_examples TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'unknown',
    is_stub INTEGER NOT NULL DEFAULT 0 CHECK (is_stub IN (0, 1)),
    last_refreshed REAL NOT NULL,
    UNIQUE(normalized_term, language)
)
"""

SQL_CREATE_RELATIONSHIPS_TABLE = """
CREATE TABLE IF NOT EXISTS relationships (
    id INTEGER PRIMARY KEY,
    word_id INTEGER NOT NULL,
    related_term TEXT NOT NULL,
    related_normalized_term TEXT NOT NULL,
    related_language TEXT NOT NULL DEFAULT 'en',
    relationship_type TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'unknown',
    confidence REAL NOT NULL DEFAULT 1.0
        CHECK (confidence >= 0.0 AND confidence <= 1.0),
    FOREIGN KEY(word_id) REFERENCES words(id) ON DELETE CASCADE,
    UNIQUE(
        word_id,
        related_normalized_term,
        related_language,
        relationship_type,
        source
    )
)
"""

SQL_CREATE_EMOTIONAL_RELATIONSHIPS_TABLE = """
CREATE TABLE IF NOT EXISTS emotional_relationships (
    id INTEGER PRIMARY KEY,
    word_id INTEGER NOT NULL,
    related_term TEXT NOT NULL,
    related_language TEXT NOT NULL DEFAULT 'en',
    relationship_type TEXT NOT NULL,
    valence REAL NOT NULL,
    arousal REAL NOT NULL,
    last_updated REAL NOT NULL,
    FOREIGN KEY(word_id) REFERENCES words(id) ON DELETE CASCADE,
    UNIQUE(word_id, related_term, related_language, relationship_type)
)
"""

SQL_CREATE_GRAPHEMES_TABLE = """
CREATE TABLE IF NOT EXISTS graphemes (
    word_id INTEGER NOT NULL,
    position INTEGER NOT NULL CHECK (position >= 0),
    text TEXT NOT NULL,
    normalized TEXT NOT NULL,
    codepoints TEXT NOT NULL,
    unicode_names TEXT NOT NULL,
    categories TEXT NOT NULL,
    combining_classes TEXT NOT NULL,
    script TEXT NOT NULL,
    PRIMARY KEY(word_id, position),
    FOREIGN KEY(word_id) REFERENCES words(id) ON DELETE CASCADE
)
"""

SQL_CREATE_PRONUNCIATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS pronunciations (
    id INTEGER PRIMARY KEY,
    word_id INTEGER NOT NULL,
    notation TEXT NOT NULL CHECK (notation IN ('arpabet', 'ipa')),
    transcription TEXT NOT NULL,
    language TEXT NOT NULL,
    dialect TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0
        CHECK (confidence >= 0.0 AND confidence <= 1.0),
    generated INTEGER NOT NULL DEFAULT 0 CHECK (generated IN (0, 1)),
    syllable_count INTEGER NOT NULL DEFAULT 0 CHECK (syllable_count >= 0),
    stress_pattern TEXT NOT NULL DEFAULT '[]',
    FOREIGN KEY(word_id) REFERENCES words(id) ON DELETE CASCADE,
    UNIQUE(word_id, notation, transcription, dialect, source)
)
"""

SQL_CREATE_PHONEMES_TABLE = """
CREATE TABLE IF NOT EXISTS phonemes (
    pronunciation_id INTEGER NOT NULL,
    position INTEGER NOT NULL CHECK (position >= 0),
    symbol TEXT NOT NULL,
    base_symbol TEXT NOT NULL,
    stress INTEGER CHECK (stress IN (0, 1, 2)),
    syllabic INTEGER NOT NULL DEFAULT 0 CHECK (syllabic IN (0, 1)),
    PRIMARY KEY(pronunciation_id, position),
    FOREIGN KEY(pronunciation_id) REFERENCES pronunciations(id) ON DELETE CASCADE
)
"""

SQL_CREATE_GRAPH_METADATA_TABLE = """
CREATE TABLE IF NOT EXISTS graph_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
)
"""

_TABLE_STATEMENTS: Tuple[str, ...] = (
    SQL_CREATE_WORDS_TABLE,
    SQL_CREATE_RELATIONSHIPS_TABLE,
    SQL_CREATE_EMOTIONAL_RELATIONSHIPS_TABLE,
    SQL_CREATE_GRAPHEMES_TABLE,
    SQL_CREATE_PRONUNCIATIONS_TABLE,
    SQL_CREATE_PHONEMES_TABLE,
    SQL_CREATE_GRAPH_METADATA_TABLE,
)

_INDEX_STATEMENTS: Tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_words_term ON words(term)",
    "CREATE INDEX IF NOT EXISTS idx_words_language ON words(language)",
    "CREATE INDEX IF NOT EXISTS idx_words_refreshed ON words(last_refreshed)",
    "CREATE INDEX IF NOT EXISTS idx_relationships_word ON relationships(word_id)",
    "CREATE INDEX IF NOT EXISTS idx_relationships_target "
    "ON relationships(related_normalized_term, related_language)",
    "CREATE INDEX IF NOT EXISTS idx_emotional_word "
    "ON emotional_relationships(word_id)",
    "CREATE INDEX IF NOT EXISTS idx_pronunciations_word " "ON pronunciations(word_id)",
)


@dataclass(frozen=True, slots=True)
class MigrationReport:
    """Summary of a completed schema check or migration."""

    previous_version: int
    current_version: int
    migrated: bool
    merged_word_collisions: int = 0


class SchemaMigrationError(RuntimeError):
    """Raised when a database cannot be migrated without risking data loss."""


def ensure_schema(connection: sqlite3.Connection) -> MigrationReport:
    """Create or migrate the Word Forge schema on an open connection."""

    original_row_factory = connection.row_factory
    connection.row_factory = sqlite3.Row
    try:
        return _ensure_schema(connection)
    finally:
        connection.row_factory = original_row_factory


def _ensure_schema(connection: sqlite3.Connection) -> MigrationReport:
    """Implement schema management with mapping-capable SQLite rows."""

    previous_version = _database_version(connection)
    if previous_version > CURRENT_SCHEMA_VERSION:
        raise SchemaMigrationError(
            f"Database schema version {previous_version} is newer than supported "
            f"version {CURRENT_SCHEMA_VERSION}. Upgrade Word Forge before opening it."
        )

    words_exist = _table_exists(connection, "words")
    migrated = False
    merged_collisions = 0
    if words_exist and not _core_schema_is_current(connection):
        merged_collisions = _migrate_core_schema(connection)
        migrated = True
    else:
        _create_schema_transactionally(connection)

    return MigrationReport(
        previous_version=previous_version,
        current_version=CURRENT_SCHEMA_VERSION,
        migrated=migrated,
        merged_word_collisions=merged_collisions,
    )


def _create_schema_transactionally(connection: sqlite3.Connection) -> None:
    """Create or complete the current schema as one validated transaction."""

    connection.commit()
    try:
        connection.execute("BEGIN IMMEDIATE")
        _create_current_schema(connection)
        _validate_foreign_keys(connection)
        _set_database_version(connection, CURRENT_SCHEMA_VERSION)
        connection.commit()
    except Exception as exc:
        connection.rollback()
        raise SchemaMigrationError(f"Failed to create lexical schema: {exc}") from exc


def _create_current_schema(connection: sqlite3.Connection) -> None:
    """Create every current table and index idempotently."""

    for statement in _TABLE_STATEMENTS:
        connection.execute(statement)
    for statement in _INDEX_STATEMENTS:
        connection.execute(statement)


def _migrate_core_schema(connection: sqlite3.Connection) -> int:
    """Rebuild legacy core tables while preserving identifiers and records."""

    connection.commit()
    connection.execute("PRAGMA foreign_keys = OFF")
    legacy_tables = tuple(
        table
        for table in ("emotional_relationships", "relationships", "words")
        if _table_exists(connection, table)
    )
    try:
        connection.execute("BEGIN IMMEDIATE")
        for index_name in (
            "idx_word_term",
            "idx_unique_relationship",
            "idx_unique_emotional_relationship",
        ):
            connection.execute(f'DROP INDEX IF EXISTS "{index_name}"')
        for table in legacy_tables:
            connection.execute(f'ALTER TABLE "{table}" RENAME TO "{table}_legacy_v1"')

        _create_current_schema(connection)
        id_mapping, merged_collisions = _copy_legacy_words(connection)
        _copy_legacy_relationships(connection, id_mapping)
        _copy_legacy_emotional_relationships(connection, id_mapping)

        for table in legacy_tables:
            connection.execute(f'DROP TABLE "{table}_legacy_v1"')
        # Recreate any indexes that were temporarily attached to renamed tables.
        _create_current_schema(connection)
        _validate_foreign_keys(connection)
        _set_database_version(connection, CURRENT_SCHEMA_VERSION)
        connection.commit()
        return merged_collisions
    except Exception as exc:
        connection.rollback()
        raise SchemaMigrationError(f"Failed to migrate lexical schema: {exc}") from exc
    finally:
        connection.execute("PRAGMA foreign_keys = ON")


def _copy_legacy_words(
    connection: sqlite3.Connection,
) -> Tuple[Dict[int, int], int]:
    """Copy legacy words and merge Unicode-normalization collisions safely."""

    if not _table_exists(connection, "words_legacy_v1"):
        return {}, 0
    rows = connection.execute(
        'SELECT * FROM "words_legacy_v1" ORDER BY id ASC'
    ).fetchall()
    id_mapping: Dict[int, int] = {}
    identity_to_id: Dict[Tuple[str, str], int] = {}
    merged_collisions = 0
    now = time.time()

    for row in rows:
        values = _row_mapping(row)
        old_id = _required_int(values.get("id"), "words.id")
        term = str(values.get("term") or "").strip()
        if not term:
            raise SchemaMigrationError(f"Legacy word {old_id} has no term")
        language = _safe_language(values.get("language"), "en")
        normalized = normalize_term(term)
        identity = (normalized, language)
        existing_id = identity_to_id.get(identity)
        if existing_id is not None:
            id_mapping[old_id] = existing_id
            merged_collisions += 1
            _merge_word_values(connection, existing_id, values)
            continue

        script = str(values.get("script") or infer_script(term))
        connection.execute(
            """
            INSERT INTO words (
                id, term, normalized_term, language, script, definition,
                part_of_speech, usage_examples, source, is_stub, last_refreshed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                old_id,
                term,
                normalized,
                language,
                script,
                str(values.get("definition") or ""),
                str(values.get("part_of_speech") or ""),
                str(values.get("usage_examples") or ""),
                str(values.get("source") or "legacy"),
                _safe_boolean(values.get("is_stub")),
                _safe_float(values.get("last_refreshed"), now),
            ),
        )
        identity_to_id[identity] = old_id
        id_mapping[old_id] = old_id
    return id_mapping, merged_collisions


def _merge_word_values(
    connection: sqlite3.Connection,
    target_id: int,
    legacy_values: Mapping[str, object],
) -> None:
    """Merge useful fields from a colliding legacy spelling."""

    current = connection.execute(
        "SELECT definition, part_of_speech, usage_examples, last_refreshed "
        "FROM words WHERE id = ?",
        (target_id,),
    ).fetchone()
    if current is None:
        raise SchemaMigrationError(f"Collision target word {target_id} is missing")
    current_values = _row_mapping(current)
    definition = _merge_text(
        str(current_values.get("definition") or ""),
        str(legacy_values.get("definition") or ""),
        " | ",
    )
    examples = _merge_text(
        str(current_values.get("usage_examples") or ""),
        str(legacy_values.get("usage_examples") or ""),
        "\n",
    )
    part_of_speech = str(current_values.get("part_of_speech") or "") or str(
        legacy_values.get("part_of_speech") or ""
    )
    refreshed = max(
        _safe_float(current_values.get("last_refreshed"), 0.0),
        _safe_float(legacy_values.get("last_refreshed"), 0.0),
    )
    connection.execute(
        """
        UPDATE words
        SET definition = ?, part_of_speech = ?, usage_examples = ?,
            is_stub = 0, last_refreshed = ?
        WHERE id = ?
        """,
        (definition, part_of_speech, examples, refreshed, target_id),
    )


def _copy_legacy_relationships(
    connection: sqlite3.Connection, id_mapping: Mapping[int, int]
) -> None:
    """Copy lexical relationships using remapped source word identifiers."""

    if not _table_exists(connection, "relationships_legacy_v1"):
        return
    rows = connection.execute('SELECT * FROM "relationships_legacy_v1"').fetchall()
    word_languages = _word_languages(connection)
    for row in rows:
        values = _row_mapping(row)
        old_word_id = _required_int(values.get("word_id"), "relationships.word_id")
        word_id = id_mapping.get(old_word_id)
        if word_id is None:
            continue
        related_term = str(values.get("related_term") or "").strip()
        relationship_type = str(values.get("relationship_type") or "").strip()
        if not related_term or not relationship_type:
            continue
        related_language = _safe_language(
            values.get("related_language"), word_languages.get(word_id, "en")
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO relationships (
                word_id, related_term, related_normalized_term, related_language,
                relationship_type, source, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                word_id,
                related_term,
                normalize_term(related_term),
                related_language,
                relationship_type,
                str(values.get("source") or "legacy"),
                _safe_confidence(values.get("confidence")),
            ),
        )


def _copy_legacy_emotional_relationships(
    connection: sqlite3.Connection, id_mapping: Mapping[int, int]
) -> None:
    """Copy emotional relationships using remapped source identifiers."""

    if not _table_exists(connection, "emotional_relationships_legacy_v1"):
        return
    rows = connection.execute(
        'SELECT * FROM "emotional_relationships_legacy_v1"'
    ).fetchall()
    word_languages = _word_languages(connection)
    for row in rows:
        values = _row_mapping(row)
        old_word_id = _required_int(
            values.get("word_id"), "emotional_relationships.word_id"
        )
        word_id = id_mapping.get(old_word_id)
        if word_id is None:
            continue
        related_term = str(values.get("related_term") or "").strip()
        relationship_type = str(values.get("relationship_type") or "").strip()
        if not related_term or not relationship_type:
            continue
        connection.execute(
            """
            INSERT OR IGNORE INTO emotional_relationships (
                word_id, related_term, related_language, relationship_type,
                valence, arousal, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                word_id,
                related_term,
                _safe_language(
                    values.get("related_language"), word_languages.get(word_id, "en")
                ),
                relationship_type,
                _safe_float(values.get("valence"), 0.0),
                _safe_float(values.get("arousal"), 0.0),
                _safe_float(values.get("last_updated"), time.time()),
            ),
        )


def _core_schema_is_current(connection: sqlite3.Connection) -> bool:
    """Return whether core tables expose every version-two column."""

    required = {
        "words": {"normalized_term", "language", "script", "source", "is_stub"},
        "relationships": {
            "related_normalized_term",
            "related_language",
            "source",
            "confidence",
        },
        "emotional_relationships": {"related_language"},
    }
    return all(
        _table_exists(connection, table)
        and columns <= _table_columns(connection, table)
        for table, columns in required.items()
    )


def _database_version(connection: sqlite3.Connection) -> int:
    """Return SQLite's application schema version."""

    row = connection.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row else 0


def _set_database_version(connection: sqlite3.Connection, version: int) -> None:
    """Set SQLite's application schema version using a trusted integer."""

    connection.execute(f"PRAGMA user_version = {int(version)}")


def _validate_foreign_keys(connection: sqlite3.Connection) -> None:
    """Raise when persisted rows violate a declared foreign-key constraint."""

    violations = connection.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise SchemaMigrationError(
            f"Schema contains {len(violations)} foreign-key violation(s)"
        )


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    """Return whether ``table`` exists."""

    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()
    return row is not None


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    """Return column names for a trusted internal table name."""

    return {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')}


def _row_mapping(row: sqlite3.Row) -> Dict[str, object]:
    """Convert a configured SQLite row to a plain mapping."""

    return {str(key): row[key] for key in row.keys()}


def _safe_language(value: object, default: str) -> str:
    """Canonicalize a migrated language value with a safe fallback."""

    try:
        return str(canonicalize_language_tag(str(value or default)))
    except LanguageTagError:
        return str(canonicalize_language_tag(default))


def _safe_boolean(value: object) -> int:
    """Convert a migrated boolean-ish value to SQLite's integer form."""

    return int(value in (True, 1, "1", "true", "True"))


def _safe_float(value: object, default: float) -> float:
    """Convert a migrated numeric value with a finite default."""

    if not isinstance(value, (int, float, str, bytes)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _required_int(value: object, field: str) -> int:
    """Convert a required migrated integer or raise a contextual error."""

    if not isinstance(value, (int, str, bytes)):
        raise SchemaMigrationError(f"Legacy field {field} is not an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SchemaMigrationError(f"Legacy field {field} is not an integer") from exc


def _safe_confidence(value: object) -> float:
    """Clamp migrated confidence into its persisted invariant."""

    return max(0.0, min(1.0, _safe_float(value, 1.0)))


def _merge_text(left: str, right: str, separator: str) -> str:
    """Merge delimited source text without repeating identical values."""

    values = []
    for source in (left, right):
        for value in source.split(separator):
            stripped = value.strip()
            if stripped and stripped not in values:
                values.append(stripped)
    return separator.join(values)


def _word_languages(connection: sqlite3.Connection) -> Dict[int, str]:
    """Return word-language values keyed by identifier."""

    return {
        int(row[0]): str(row[1])
        for row in connection.execute("SELECT id, language FROM words")
    }


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "MigrationReport",
    "SQL_CREATE_EMOTIONAL_RELATIONSHIPS_TABLE",
    "SQL_CREATE_GRAPH_METADATA_TABLE",
    "SQL_CREATE_GRAPHEMES_TABLE",
    "SQL_CREATE_PHONEMES_TABLE",
    "SQL_CREATE_PRONUNCIATIONS_TABLE",
    "SQL_CREATE_RELATIONSHIPS_TABLE",
    "SQL_CREATE_WORDS_TABLE",
    "SchemaMigrationError",
    "ensure_schema",
]
