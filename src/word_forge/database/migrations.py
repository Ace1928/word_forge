"""Crash-safe schema migrations for Word Forge's local SQLite lexicon.

The original Word Forge schema grew organically and did not record which
schema revision a database used.  This module provides a small, dependency-free
migration runner with two guarantees that matter to users:

* every migration is committed atomically, so interruption leaves the previous
  schema usable; and
* migrations are additive, so older Word Forge tools can still read their core
  tables while users move to the canonical production API.

Applications normally do not call this module directly.  :class:`DBManager`
runs it whenever a database is opened.
"""

from __future__ import annotations

import sqlite3
import time
import unicodedata
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass

LATEST_SCHEMA_VERSION = 2


class MigrationError(RuntimeError):
    """Raised when a database cannot be migrated safely."""


@dataclass(frozen=True, slots=True)
class Migration:
    """One ordered, atomic database schema transition."""

    version: int
    name: str
    apply: Callable[[sqlite3.Connection], None]


_SCRIPT_MARKERS: tuple[tuple[str, str], ...] = (
    ("LATIN", "Latn"),
    ("CYRILLIC", "Cyrl"),
    ("GREEK", "Grek"),
    ("ARABIC", "Arab"),
    ("HEBREW", "Hebr"),
    ("DEVANAGARI", "Deva"),
    ("BENGALI", "Beng"),
    ("HIRAGANA", "Hira"),
    ("KATAKANA", "Kana"),
    ("HANGUL", "Hang"),
    ("CJK", "Hani"),
    ("IDEOGRAPH", "Hani"),
    ("THAI", "Thai"),
    ("ARMENIAN", "Armn"),
    ("GEORGIAN", "Geor"),
    ("ETHIOPIC", "Ethi"),
)


def normalize_lemma(value: str) -> str:
    """Return the stable Unicode key used for exact lexical identity."""

    return unicodedata.normalize("NFC", value).casefold()


def detect_script(value: str) -> str:
    """Infer a compact ISO 15924 script tag without external dependencies.

    The first letter with a recognized Unicode name determines the primary
    script.  ``Zyyy`` (Common) is returned for punctuation, emoji, digits, or
    scripts not yet in the conservative mapping.  The original spelling is
    always retained, so this hint can be corrected later without data loss.
    """

    for character in unicodedata.normalize("NFC", value):
        if not character.isalpha():
            continue
        name = unicodedata.name(character, "")
        for marker, script in _SCRIPT_MARKERS:
            if marker in name:
                return script
    return "Zyyy"


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    """Return column names for one trusted, internally supplied table name."""

    if not table.replace("_", "").isalnum():
        raise MigrationError(f"Unsafe internal table name: {table!r}")
    return {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')}


def _add_columns(
    connection: sqlite3.Connection,
    table: str,
    declarations: Iterable[tuple[str, str]],
) -> None:
    existing = _columns(connection, table)
    for name, declaration in declarations:
        if name not in existing:
            connection.execute(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {declaration}')


def _new_uid() -> str:
    """Return an opaque, immutable RFC 4122 identifier."""

    return str(uuid.uuid4())


def _execute_script_atomic(connection: sqlite3.Connection, script: str) -> None:
    """Execute a simple DDL script without ``executescript``'s implicit commit."""

    for statement in script.split(";"):
        sql = statement.strip()
        if sql:
            connection.execute(sql)


def _backfill_identity(
    connection: sqlite3.Connection,
    *,
    table: str,
    term_column: str,
    normalized_column: str,
    refreshed_column: str | None,
) -> None:
    columns = ["rowid", term_column, "uid", normalized_column, "script", "created_at"]
    if refreshed_column:
        columns.append(refreshed_column)
    rows = connection.execute(
        f'SELECT {", ".join(columns)} FROM "{table}" '
        f"WHERE uid IS NULL OR uid = '' OR \"{normalized_column}\" = '' OR created_at = 0"
    ).fetchall()
    now = time.time()
    for row in rows:
        row_id = int(row[0])
        term = str(row[1])
        uid = str(row[2] or _new_uid())
        normalized = str(row[3] or normalize_lemma(term))
        stored_script = str(row[4] or "")
        script = detect_script(term) if stored_script in {"", "Zyyy"} else stored_script
        created_at = float(row[5] or 0.0)
        refreshed_at = float(row[6] or 0.0) if refreshed_column else 0.0
        timestamp = created_at or refreshed_at or now
        connection.execute(
            f'UPDATE "{table}" SET uid = ?, "{normalized_column}" = ?, script = ?, '
            "created_at = ?, updated_at = CASE WHEN updated_at = 0 THEN ? ELSE updated_at END "
            "WHERE rowid = ?",
            (uid, normalized, script, timestamp, refreshed_at or timestamp, row_id),
        )


def _backfill_uids(connection: sqlite3.Connection, table: str) -> None:
    rows = connection.execute(
        f"SELECT rowid FROM \"{table}\" WHERE uid IS NULL OR uid = ''"
    ).fetchall()
    for row in rows:
        connection.execute(
            f'UPDATE "{table}" SET uid = ? WHERE rowid = ?',
            (_new_uid(), int(row[0])),
        )


def _backfill_english_lexemes(connection: sqlite3.Connection) -> None:
    """Give legacy English-only words a canonical language-aware lexeme."""

    rows = connection.execute(
        """
        SELECT term, definition, part_of_speech, source, last_refreshed,
               normalized_term, script, confidence, review_state,
               created_at, updated_at, deleted_at
        FROM words w
        WHERE NOT EXISTS (
            SELECT 1 FROM lexemes l WHERE l.lemma = w.term AND l.lang = 'en'
        )
        """
    ).fetchall()
    for row in rows:
        connection.execute(
            """
            INSERT INTO lexemes(
                lemma, lang, part_of_speech, gloss, base_term, source,
                last_refreshed, uid, normalized_lemma, script, revision,
                confidence, review_state, created_at, updated_at, deleted_at
            ) VALUES (?, 'en', ?, ?, '', ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)
            """,
            (
                str(row[0]),
                str(row[2] or ""),
                str(row[1] or ""),
                str(row[3] or ""),
                float(row[4] or 0.0),
                _new_uid(),
                str(row[5]),
                str(row[6]),
                float(row[7]),
                str(row[8]),
                float(row[9]),
                float(row[10]),
                row[11],
            ),
        )


def _migration_1_identity_and_provenance(connection: sqlite3.Connection) -> None:
    """Add immutable identity, revision, provenance, and tombstone metadata."""

    identity_columns = (
        ("uid", "TEXT"),
        ("normalized_lemma", "TEXT NOT NULL DEFAULT ''"),
        ("script", "TEXT NOT NULL DEFAULT 'Zyyy'"),
        ("revision", "INTEGER NOT NULL DEFAULT 1"),
        ("confidence", "REAL NOT NULL DEFAULT 1.0"),
        ("review_state", "TEXT NOT NULL DEFAULT 'unreviewed'"),
        ("created_at", "REAL NOT NULL DEFAULT 0"),
        ("updated_at", "REAL NOT NULL DEFAULT 0"),
        ("deleted_at", "REAL"),
    )
    _add_columns(connection, "lexemes", identity_columns)

    word_columns = tuple(
        ("normalized_term" if name == "normalized_lemma" else name, declaration)
        for name, declaration in identity_columns
    ) + (("source", "TEXT NOT NULL DEFAULT ''"),)
    _add_columns(connection, "words", word_columns)

    assertion_columns = (
        ("uid", "TEXT"),
        ("revision", "INTEGER NOT NULL DEFAULT 1"),
        ("source", "TEXT NOT NULL DEFAULT ''"),
        ("confidence", "REAL NOT NULL DEFAULT 1.0"),
        ("review_state", "TEXT NOT NULL DEFAULT 'unreviewed'"),
        ("created_at", "REAL NOT NULL DEFAULT 0"),
        ("updated_at", "REAL NOT NULL DEFAULT 0"),
        ("deleted_at", "REAL"),
    )
    _add_columns(connection, "relationships", assertion_columns)
    _add_columns(
        connection,
        "translations",
        tuple(item for item in assertion_columns if item[0] != "source"),
    )

    _backfill_identity(
        connection,
        table="lexemes",
        term_column="lemma",
        normalized_column="normalized_lemma",
        refreshed_column="last_refreshed",
    )
    _backfill_identity(
        connection,
        table="words",
        term_column="term",
        normalized_column="normalized_term",
        refreshed_column="last_refreshed",
    )
    _backfill_english_lexemes(connection)
    _backfill_uids(connection, "relationships")
    _backfill_uids(connection, "translations")
    now = time.time()
    for table in ("relationships", "translations"):
        connection.execute(
            f'UPDATE "{table}" SET created_at = CASE WHEN created_at = 0 THEN ? ELSE created_at END, '
            "updated_at = CASE WHEN updated_at = 0 THEN ? ELSE updated_at END",
            (now, now),
        )

    _execute_script_atomic(
        connection,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_lexemes_uid ON lexemes(uid);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_words_uid ON words(uid);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_relationships_uid ON relationships(uid);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_translations_uid ON translations(uid);
        CREATE INDEX IF NOT EXISTS idx_lexemes_normalized_live
            ON lexemes(normalized_lemma, lang, script) WHERE deleted_at IS NULL;
        CREATE INDEX IF NOT EXISTS idx_words_normalized_live
            ON words(normalized_term) WHERE deleted_at IS NULL;
        """,
    )


def _migration_2_lifecycle_tables(connection: sqlite3.Connection) -> None:
    """Add senses, source records, audit history, derived state, and jobs."""

    _execute_script_atomic(
        connection,
        """
        CREATE TABLE IF NOT EXISTS sources (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            origin TEXT NOT NULL DEFAULT '',
            license TEXT NOT NULL DEFAULT '',
            retrieved_at REAL,
            checksum TEXT NOT NULL DEFAULT '',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS senses (
            id TEXT PRIMARY KEY,
            lexeme_uid TEXT NOT NULL,
            definition TEXT NOT NULL DEFAULT '',
            part_of_speech TEXT NOT NULL DEFAULT '',
            domain TEXT NOT NULL DEFAULT '',
            register_name TEXT NOT NULL DEFAULT '',
            source_id TEXT,
            confidence REAL NOT NULL DEFAULT 1.0 CHECK (confidence >= 0 AND confidence <= 1),
            review_state TEXT NOT NULL DEFAULT 'unreviewed',
            revision INTEGER NOT NULL DEFAULT 1 CHECK (revision >= 1),
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            deleted_at REAL,
            FOREIGN KEY(source_id) REFERENCES sources(id)
        );
        CREATE INDEX IF NOT EXISTS idx_senses_lexeme_live
            ON senses(lexeme_uid, part_of_speech) WHERE deleted_at IS NULL;

        CREATE TABLE IF NOT EXISTS change_log (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            revision INTEGER NOT NULL,
            operation TEXT NOT NULL,
            actor TEXT NOT NULL DEFAULT 'local',
            occurred_at REAL NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            checksum TEXT NOT NULL DEFAULT '',
            UNIQUE(entity_type, entity_id, revision)
        );
        CREATE INDEX IF NOT EXISTS idx_change_log_occurred
            ON change_log(occurred_at, sequence);

        CREATE TABLE IF NOT EXISTS derived_artifacts (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            source_revision INTEGER NOT NULL,
            model TEXT NOT NULL DEFAULT '',
            dimensions INTEGER,
            configuration_hash TEXT NOT NULL DEFAULT '',
            checksum TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'current',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            UNIQUE(kind, entity_id, model, configuration_hash)
        );

        CREATE TABLE IF NOT EXISTS invalidation_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            source_revision INTEGER NOT NULL,
            reason TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'pending',
            created_at REAL NOT NULL,
            claimed_at REAL,
            completed_at REAL,
            UNIQUE(kind, entity_id, source_revision, reason)
        );
        CREATE INDEX IF NOT EXISTS idx_invalidation_pending
            ON invalidation_queue(state, created_at) WHERE state = 'pending';

        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            state TEXT NOT NULL DEFAULT 'pending',
            idempotency_key TEXT NOT NULL UNIQUE,
            attempts INTEGER NOT NULL DEFAULT 0,
            max_attempts INTEGER NOT NULL DEFAULT 3,
            lease_owner TEXT,
            lease_expires_at REAL,
            last_error TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            completed_at REAL
        );
        CREATE INDEX IF NOT EXISTS idx_jobs_claimable
            ON jobs(state, lease_expires_at, created_at);
        """,
    )


_MIGRATIONS: tuple[Migration, ...] = (
    Migration(1, "identity-and-provenance", _migration_1_identity_and_provenance),
    Migration(2, "lifecycle-tables", _migration_2_lifecycle_tables),
)


def apply_migrations(connection: sqlite3.Connection) -> int:
    """Upgrade ``connection`` and return its resulting schema version.

    A database created by a newer Word Forge is rejected rather than opened
    with potentially incompatible code.  Every version is committed in its own
    ``BEGIN IMMEDIATE`` transaction, which serializes competing initializers and
    guarantees that the recorded version changes only after all schema work.
    """

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at REAL NOT NULL
        )
        """
    )
    row = connection.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations").fetchone()
    current = int(row[0]) if row else 0
    pragma_row = connection.execute("PRAGMA user_version").fetchone()
    pragma_version = int(pragma_row[0]) if pragma_row else 0
    current = max(current, pragma_version)
    if current > LATEST_SCHEMA_VERSION:
        raise MigrationError(
            f"Database schema {current} is newer than this Word Forge "
            f"(supports up to {LATEST_SCHEMA_VERSION})"
        )

    for migration in _MIGRATIONS:
        if migration.version <= current:
            continue
        try:
            connection.execute("BEGIN IMMEDIATE")
            migration.apply(connection)
            connection.execute(
                "INSERT OR REPLACE INTO schema_migrations(version, name, applied_at) "
                "VALUES (?, ?, ?)",
                (migration.version, migration.name, time.time()),
            )
            connection.execute(f"PRAGMA user_version = {migration.version}")
            connection.commit()
        except Exception as exc:
            connection.rollback()
            raise MigrationError(
                f"Migration {migration.version} ({migration.name}) failed; "
                "the previous schema was preserved"
            ) from exc
        current = migration.version
    return current


__all__ = [
    "LATEST_SCHEMA_VERSION",
    "MigrationError",
    "apply_migrations",
    "detect_script",
    "normalize_lemma",
]
