"""Database Manager Module.

This module provides a comprehensive interface for managing a SQLite database
that stores lexical data including words, definitions, and their various
relationships (lexical, semantic, emotive, affective).

It implements a robust architecture for:
- Creating and maintaining database schema
- Inserting and updating lexical entries
- Managing complex relationship networks between terms
- Providing type-safe interfaces for database operations
- Connection pooling for efficient resource management
- Transaction management with automatic rollback on errors

The DBManager class serves as the central access point for all database operations,
ensuring data integrity, proper error handling, and efficient query execution.
The module uses SQLite as its database backend and provides robust error
handling, connection management, and performance optimization techniques
like connection pooling and prepared statements.

Key Classes:
    DBManager: Core class handling all database operations with connection pooling
    RelationshipTypeManager: Manages and validates relationship type definitions

Data Structures:
    WordEntryDict: Complete word entry with all metadata and relationships
    RelationshipDict: Represents relationships between terms
    WordDataDict: Simplified word data for listing operations

Exception Hierarchy:
    DatabaseError: Base exception for all database-related errors
        - ConnectionError: Database connection failures
        - QueryError: SQL query execution errors
        - SchemaError: Database schema issues
        - TransactionError: Transaction management failures
        - TermNotFoundError: Word lookup failures

Usage Examples:
    >>> # Initialize the database manager
    >>> db = DBManager()
    >>>
    >>> # Create schema if needed
    >>> db.create_tables()
    >>>
    >>> # Add or update a word
    >>> db.insert_or_update_word(
    ...     "algorithm",
    ...     "A step-by-step procedure for solving a problem",
    ...     "noun",
    ...     ["The sorting algorithm runs in O(n log n) time"]
    ... )
    >>>
    >>> # Create relationships between words
    >>> db.insert_relationship("algorithm", "procedure", "synonym")
    >>>
    >>> # Retrieve complete word information
    >>> word_entry = db.get_word_entry("algorithm")
    >>>
    >>> # Use a transaction for multiple operations
    >>> with db.transaction() as conn:
    ...     conn.execute("INSERT INTO words (term) VALUES (?)", ("lexicon",))
    ...     conn.execute("INSERT INTO words (term) VALUES (?)", ("syntax",))
"""

import json
import sqlite3
import time
import unicodedata
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from threading import Lock, local
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

from word_forge.config import config
from word_forge.database.schema import (
    CURRENT_SCHEMA_VERSION,
    MigrationReport,
    SchemaMigrationError,
    ensure_schema,
)
from word_forge.parser.linguistics import (
    Grapheme,
    Pronunciation,
    canonicalize_language_tag,
    infer_script,
    normalize_term,
)


class DatabaseError(Exception):
    """
    Base exception for database operations.

    Provides a consistent foundation for all database-related exceptions
    with support for capturing the original cause for detailed diagnostics.

    Attributes:
        message: Detailed error description
        cause: Original exception that triggered this error
    """

    def __init__(self, message: str, cause: Optional[Exception] = None) -> None:
        """
        Initialize with detailed error message and optional cause.

        Args:
            message: Error description with context
            cause: Original exception that caused this error (if applicable)
        """
        super().__init__(message)
        self.__cause__ = cause
        self.message = message
        self.cause = cause

    def __str__(self) -> str:
        """Provide detailed error message including cause if available."""
        error_msg = self.message
        if self.cause:
            error_msg += f" | Cause: {str(self.cause)}"
        return error_msg


class TermNotFoundError(DatabaseError):
    """
    Raised when a term cannot be found in the database.

    Provides clear context about which specific term was not found,
    allowing for precise error handling in calling code.

    Attributes:
        term: The specific term that could not be found
    """

    def __init__(self, term: str) -> None:
        """
        Initialize with specific term that was not found.

        Args:
            term: The term that could not be found in the database
        """
        super().__init__(f"Term '{term}' not found in database")
        self.term = term


class ConnectionError(DatabaseError):
    """
    Raised when database connection cannot be established or maintained.

    Occurs during connection initialization, pool exhaustion, or when
    an existing connection is unexpectedly terminated.

    Attributes:
        db_path: Path to the database that failed to connect
    """

    def __init__(
        self,
        message: str,
        cause: Optional[Exception] = None,
        db_path: Optional[str] = None,
    ) -> None:
        """
        Initialize connection error with context details.

        Args:
            message: Error description with context
            cause: Original exception that caused this error
            db_path: Database path that failed to connect
        """
        super().__init__(message, cause)
        self.db_path = db_path


class QueryError(DatabaseError):
    """
    Raised when a database query fails to execute.

    Typically occurs due to syntax errors, constraint violations,
    or invalid parameters.

    Attributes:
        query: The SQL query that failed
        params: Parameters passed to the query
    """

    def __init__(
        self,
        message: str,
        cause: Optional[Exception] = None,
        query: Optional[str] = None,
        params: Optional[Union[Tuple[Any, ...], Dict[str, Any]]] = None,
    ) -> None:
        """
        Initialize query error with context details.

        Args:
            message: Error description
            cause: Original exception that caused this error
            query: SQL query that failed
            params: Parameters passed to the query
        """
        super().__init__(message, cause)
        self.query = query
        self.params = params

    def __str__(self) -> str:
        """Provide detailed error message including query and parameters if available."""
        error_msg = super().__str__()
        if self.query:
            error_msg += f"\nQuery: {self.query}"
        if self.params:
            error_msg += f"\nParameters: {self.params}"
        return error_msg


class TransactionError(DatabaseError):
    """
    Raised when transaction operations fail.

    Occurs when commits or rollbacks fail, or when transaction
    boundaries are violated.
    """

    pass


class SchemaError(DatabaseError):
    """
    Raised when database schema operations fail.

    Occurs during schema creation, migration, or validation when the
    database structure doesn't match expected specifications.

    Attributes:
        table: The table with schema issues (if applicable)
    """

    def __init__(
        self,
        message: str,
        cause: Optional[Exception] = None,
        table: Optional[str] = None,
    ) -> None:
        """
        Initialize schema error with context details.

        Args:
            message: Error description
            cause: Original exception that caused this error
            table: Relevant table name with schema issues
        """
        super().__init__(message, cause)
        self.table = table


class RelationshipDict(TypedDict):
    """
    Type definition for relationship dictionary structure.

    Represents the standardized format for relationship data
    across the application, ensuring type consistency.

    Attributes:
        related_term: The term related to the base term
        relationship_type: Type of semantic relationship (e.g., synonym, antonym)
    """

    related_term: str
    related_normalized_term: str
    related_language: str
    relationship_type: str
    source: str
    confidence: float


class GraphemeDict(TypedDict):
    """Persisted Unicode grapheme-cluster metadata."""

    position: int
    text: str
    normalized: str
    codepoints: List[str]
    unicode_names: List[str]
    categories: List[str]
    combining_classes: List[int]
    script: str


class PhonemeDict(TypedDict):
    """Persisted phonetic segment metadata."""

    position: int
    symbol: str
    base_symbol: str
    stress: Optional[int]
    syllabic: bool


class PronunciationDict(TypedDict):
    """Persisted pronunciation and ordered phoneme records."""

    id: int
    notation: str
    text: str
    language: str
    dialect: Optional[str]
    source: str
    confidence: float
    generated: bool
    syllable_count: int
    stress_pattern: List[int]
    phonemes: List[PhonemeDict]


class WordEntryDict(TypedDict):
    """
    Type definition for word entry dictionary structure.

    Represents the complete structure of a word entry including
    its relationships and metadata.

    Attributes:
        id: Unique identifier for the word (str for chroma db compat)
        id_int: Unique int ID for
        term: The actual word or phrase
        definition: Meaning or explanation of the term
        part_of_speech: Grammatical category (noun, verb, etc.)
        usage_examples: List of example sentences using the term
        language: str
        normalized_term: Unicode-normalized lexical identity key
        script: ISO 15924 script code
        source: Primary source responsible for the current entry
        is_stub: Whether the word awaits lexical enrichment
        last_refreshed: Timestamp of last update (epoch time)
        relationships: List of relationships to other terms
    """

    id: str
    id_int: int
    term: str
    definition: str
    part_of_speech: str
    usage_examples: List[str]
    language: str
    normalized_term: str
    script: str
    last_refreshed: float
    source: str
    is_stub: bool
    relationships: List[RelationshipDict]
    graphemes: List[GraphemeDict]
    pronunciations: List[PronunciationDict]


class WordDataDict(TypedDict):
    """
    Type definition for word data returned by get_all_words.

    Provides a simplified view of word data for listing and
    bulk operations.

    Attributes:
        id: Unique identifier for the word
        term: The actual word or phrase
        definition: Meaning or explanation of the term
        usage_examples: Example sentences (as serialized string)
    """

    id: int
    term: str
    definition: str
    usage_examples: str
    language: str
    script: str
    last_refreshed: float


class SQLExecutor(Protocol):
    """
    Protocol for objects that can execute SQL queries.

    Defines the minimal interface required for SQL execution,
    allowing for type-safe dependency injection and testing.
    """

    def execute(
        self,
        sql: str,
        parameters: Union[Tuple[Any, ...], List[Any], Dict[str, Any]] = (),
    ) -> Any: ...

    def fetchone(self) -> Optional[Tuple[Any, ...]]: ...

    def fetchall(self) -> List[Tuple[Any, ...]]: ...


# Type variables for return type annotations
T = TypeVar("T")
Row = sqlite3.Row
Connection = sqlite3.Connection
Cursor = sqlite3.Cursor
QueryParams = Union[Tuple[Any, ...], Dict[str, Any]]


SQL_CHECK_WORDS_TABLE = (
    "SELECT name FROM sqlite_master WHERE type='table' AND name='words'"
)

# SQL query constants for data operations
SQL_INSERT_OR_UPDATE_WORD = """
INSERT INTO words (
    term, normalized_term, language, script, definition, part_of_speech,
    usage_examples, source, is_stub, last_refreshed
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(normalized_term, language)
DO UPDATE SET
    term=excluded.term,
    script=excluded.script,
    definition=CASE
        WHEN excluded.definition <> '' THEN excluded.definition
        ELSE words.definition
    END,
    part_of_speech=CASE
        WHEN excluded.part_of_speech <> '' THEN excluded.part_of_speech
        ELSE words.part_of_speech
    END,
    usage_examples=CASE
        WHEN excluded.usage_examples <> '' THEN excluded.usage_examples
        ELSE words.usage_examples
    END,
    source=CASE
        WHEN excluded.source <> 'unknown' THEN excluded.source
        ELSE words.source
    END,
    is_stub=CASE
        WHEN excluded.is_stub = 0 THEN 0
        ELSE words.is_stub
    END,
    last_refreshed=excluded.last_refreshed
"""

SQL_INSERT_RELATIONSHIP = """
INSERT OR IGNORE INTO relationships
(word_id, related_term, related_normalized_term, related_language,
 relationship_type, source, confidence)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""

SQL_GET_WORD_ENTRY = """
SELECT id, term, normalized_term, language, script, definition,
       part_of_speech, usage_examples, source, is_stub, last_refreshed
FROM words WHERE normalized_term = ? AND language = ?
"""

SQL_GET_RELATIONSHIPS = """
SELECT related_term, related_normalized_term, related_language,
       relationship_type, source, confidence
FROM relationships
WHERE word_id = ?
"""

SQL_GET_WORD_ID = """
SELECT id FROM words WHERE normalized_term = ? AND language = ?
"""

SQL_GET_ALL_WORDS = """
SELECT id, term, definition, usage_examples, language, script, last_refreshed
FROM words
"""

SQL_GET_UPDATED_WORDS = """
SELECT id, term, definition, usage_examples, language, script, last_refreshed
FROM words
WHERE last_refreshed > ?
"""

SQL_CHECK_RELATIONSHIPS_TABLE = """
SELECT name FROM sqlite_master WHERE type='table' AND name='relationships'
"""

# SQL statements for database setup and schema validation
SQL_PRAGMA_FOREIGN_KEYS = "PRAGMA foreign_keys = ON"
SQL_PRAGMA_JOURNAL_MODE = "PRAGMA journal_mode = WAL"
SQL_PRAGMA_SYNCHRONOUS = "PRAGMA synchronous = NORMAL"
SQL_CHECK_TABLE_EXISTS = """
SELECT name FROM sqlite_master
WHERE type='table' AND name=?
"""
SQL_GET_TABLE_INFO = """
PRAGMA table_info(?)
"""


class DBManager:
    """
    Manages the SQLite database for terms, definitions, relationships, etc.

    Thread-safe implementation that automatically handles connection lifecycle
    across different execution contexts including multithreaded environments.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize the database manager with an optional custom path."""
        self.db_path = Path(db_path) if db_path else Path(config.database.db_path)
        self._thread_local = local()  # Thread-local storage for connections
        self._thread_local.conn_pool = []  # Initialize connection pool
        self._lock = Lock()
        self._schema_lock = Lock()
        self._max_pool_size = getattr(config.database, "max_connections", 5)
        self.last_migration_report: Optional[MigrationReport] = None

        try:
            self._ensure_database_directory()
        except OSError as e:
            raise ConnectionError(
                f"Failed to create database directory for {self.db_path}",
                e,
                str(self.db_path),
            ) from e

        # Ensure schema exists on initialization for convenience. Schema and
        # database errors retain their precise public exception types.
        self.create_tables()

    @property
    def connection(self) -> Optional[sqlite3.Connection]:
        """Get the current thread's connection, if any."""
        return getattr(self._thread_local, "connection", None)

    @connection.setter
    def connection(self, conn: Optional[sqlite3.Connection]) -> None:
        """Set the current thread's connection."""
        self._thread_local.connection = conn

    @property
    def _conn_pool(self) -> List[Connection]:
        """Get the connection pool for the current thread."""
        if not hasattr(self._thread_local, "conn_pool"):
            self._thread_local.conn_pool = []  # Connection list is created empty
        return cast(List[Connection], self._thread_local.conn_pool)

    def _ensure_database_directory(self) -> None:
        """Ensure the database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection with proper configuration."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(str(self.db_path), timeout=60.0)
            connection.row_factory = sqlite3.Row

            # Configure connection for optimal performance and safety
            connection.execute(SQL_PRAGMA_FOREIGN_KEYS)
            connection.execute(SQL_PRAGMA_JOURNAL_MODE)
            connection.execute(SQL_PRAGMA_SYNCHRONOUS)

            return connection
        except sqlite3.Error as e:
            raise ConnectionError(
                f"Failed to connect to database at {self.db_path}", e, str(self.db_path)
            )

    def create_connection(self) -> sqlite3.Connection:
        """Create a new database connection."""
        with self._lock:
            conn = self._create_connection()
            self._conn_pool.append(conn)
            return conn

    def create_tables(self) -> None:
        """Create or migrate every database table transactionally."""
        try:
            with self._schema_lock:
                with self.get_connection() as conn:
                    self.last_migration_report = ensure_schema(conn)
                    conn.execute(SQL_PRAGMA_FOREIGN_KEYS)
                    conn.execute(SQL_PRAGMA_JOURNAL_MODE)
                    conn.execute(SQL_PRAGMA_SYNCHRONOUS)
        except (ConnectionError, SchemaMigrationError, sqlite3.Error) as e:
            raise SchemaError("Failed to create database schema", e)

    def ensure_tables_exist(self) -> None:
        """Ensure that all required tables exist in the database."""
        if (
            not self.table_exists("words")
            or self.schema_version < CURRENT_SCHEMA_VERSION
        ):
            self.create_tables()

    @property
    def schema_version(self) -> int:
        """Return the persisted application schema version."""

        try:
            value = self.execute_scalar("PRAGMA user_version")
            return int(value or 0)
        except (QueryError, TypeError, ValueError):
            return 0

    @contextmanager
    def get_connection(self) -> Iterator[Connection]:
        """
        Thread-safe retrieval of a database connection.

        Each thread receives its own dedicated connection, ensuring SQLite's
        thread requirements are respected while maintaining optimal performance.

        Yields:
            An active SQLite database connection for the current thread
        """
        conn = None
        try:
            # First try to get connection from thread-local pool
            if self._conn_pool:
                conn = self._conn_pool.pop()
            else:
                conn = self._create_connection()

            # Yield connection to caller
            yield conn

            # Return connection to thread-local pool if still valid
            if conn and len(self._conn_pool) < self._max_pool_size:
                self._conn_pool.append(conn)
                conn = None  # Prevent closing outside

        except sqlite3.Error as e:
            raise ConnectionError(
                "Failed to get database connection", e, str(self.db_path)
            )

        finally:
            # Close connection if not returned to pool
            if conn:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass  # Already closing due to error, ignore

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get the current thread's database connection or create a new one.

        This method ensures each thread has its own dedicated connection,
        maintaining SQLite's thread affinity requirements.
        """
        if self.connection is None:
            self.connection = self._create_connection()
        return self.connection

    @contextmanager
    def transaction(self) -> Iterator[Connection]:
        """
        Thread-safe transaction context manager.

        Creates a transaction context that automatically commits on successful
        completion or rolls back on error, using the current thread's connection.
        """
        with self.get_connection() as conn:
            try:
                conn.execute("BEGIN")
                yield conn
                conn.commit()
            except Exception as e:
                # Roll back on any error
                try:
                    conn.rollback()
                except sqlite3.Error as rollback_error:
                    raise TransactionError(
                        "Failed to roll back transaction after error",
                        rollback_error,
                    ) from e

                # Re-raise original error with context
                if isinstance(e, DatabaseError):
                    raise
                elif isinstance(e, sqlite3.Error):
                    raise QueryError("SQL error during transaction", e) from e
                else:
                    raise TransactionError(
                        "Error during database transaction", e
                    ) from e

    def execute_query(
        self, query: str, params: Optional[QueryParams] = None
    ) -> List[Row]:
        """
        Execute a query and return all results.

        Handles parameter binding and error handling for SELECT queries,
        returning results as a list of Row objects.

        Args:
            query: SQL query to execute
            params: Parameters for query (tuple or dict)

        Returns:
            List of Row objects containing query results

        Raises:
            QueryError: If query execution fails

        Examples:
            >>> rows = db.execute_query("SELECT * FROM words WHERE term LIKE ?", ("lex%",))
            >>> for row in rows:
            ...     print(dict(row))
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(query, params or ())
                return cursor.fetchall()
        except sqlite3.Error as e:
            raise QueryError("Query execution failed", e, query, params or ())

    def execute_scalar(self, query: str, params: Optional[QueryParams] = None) -> Any:
        """
        Execute a query and return a single scalar value.

        Optimized for queries that return a single value, such as
        COUNT, SUM, or single column/row lookups.

        Args:
            query: SQL query to execute
            params: Parameters for query (tuple or dict)

        Returns:
            The first column of the first row, or None if no results

        Raises:
            QueryError: If query execution fails

        Examples:
            >>> count = db.execute_scalar("SELECT COUNT(*) FROM words")
            >>> word_id = db.execute_scalar("SELECT id FROM words WHERE term = ?", ("lexicon",))
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(query, params or ())
                row = cursor.fetchone()
                return row[0] if row else None
        except sqlite3.Error as e:
            raise QueryError("Scalar query execution failed", e, query, params or ())

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.

        Args:
            table_name: Name of the table to check

        Returns:
            True if the table exists, False otherwise

        Examples:
            >>> if not db.table_exists("words"):
            ...     print("Words table doesn't exist")
        """
        try:
            count = self.execute_scalar(SQL_CHECK_TABLE_EXISTS, (table_name,))
            return bool(count)
        except QueryError:
            return False

    def insert_or_update_word(
        self,
        term: str,
        definition: str = "",
        part_of_speech: str = "",
        usage_examples: Optional[List[str]] = None,
        *,
        language: str = "en",
        script: Optional[str] = None,
        source: str = "unknown",
        is_stub: bool = False,
    ) -> int:
        """
        Insert a new word or update an existing word in the database.

        Args:
            term: The word or phrase to store
            definition: The word's meaning or description
            part_of_speech: Grammatical category (noun, verb, etc.)
            usage_examples: List of example sentences using the term
            language: Structurally valid BCP 47 language tag
            script: Optional ISO 15924 script code; inferred when omitted
            source: Provenance label for the lexical record
            is_stub: Whether this record awaits lexical enrichment

        Returns:
            Persistent word identifier.

        Raises:
            DatabaseError: If the insertion or update fails
            ValueError: If term is empty

        Examples:
            >>> db.insert_or_update_word(
            ...     "algorithm",
            ...     "A step-by-step procedure for solving a problem",
            ...     "noun",
            ...     ["The sorting algorithm runs in O(n log n) time"]
            ... )
        """
        if not isinstance(term, str) or not term.strip():
            raise ValueError("Term cannot be empty")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")

        # Ensure tables exist before attempting operations
        self.ensure_tables_exist()

        display_term = unicodedata.normalize("NFC", term.strip())
        normalized = normalize_term(display_term)
        canonical_language = canonicalize_language_tag(language)
        resolved_script = script or infer_script(display_term)
        # Handle optional usage examples
        examples = usage_examples if usage_examples else []
        serialized_examples = "\n".join(examples)
        current_time = time.time()

        try:
            with self.transaction() as conn:
                conn.execute(
                    SQL_INSERT_OR_UPDATE_WORD,
                    (
                        display_term,
                        normalized,
                        canonical_language,
                        resolved_script,
                        definition,
                        part_of_speech,
                        serialized_examples,
                        source.strip(),
                        int(is_stub),
                        current_time,
                    ),
                )
                row = conn.execute(
                    SQL_GET_WORD_ID, (normalized, canonical_language)
                ).fetchone()
                if row is None:
                    raise DatabaseError(
                        f"Upserted word '{display_term}' could not be retrieved"
                    )
                return int(row[0])
        except (sqlite3.Error, TransactionError) as e:
            raise DatabaseError(f"Failed to insert or update word '{display_term}'", e)

    def get_word_id(self, term: str, language: str = "en") -> int:
        """
        Get the database ID for a specific term.

        Args:
            term: The word to look up
            language: BCP 47 language tag distinguishing homographs

        Returns:
            The numeric ID of the word in the database

        Raises:
            TermNotFoundError: If the term doesn't exist in the database
            QueryError: If database query fails

        Examples:
            >>> try:
            ...     word_id = db.get_word_id("algorithm")
            ...     print(f"ID for 'algorithm': {word_id}")
            ... except TermNotFoundError:
            ...     print("Term not found")
        """
        try:
            normalized = normalize_term(term)
            canonical_language = canonicalize_language_tag(language)
            result = self.execute_scalar(
                SQL_GET_WORD_ID, (normalized, canonical_language)
            )
            if result is None:
                raise TermNotFoundError(term)
            return cast(int, result)
        except QueryError as e:
            raise QueryError(f"Database error while retrieving ID for term '{term}'", e)

    def word_exists(self, term: str, language: str = "en") -> bool:
        """Return ``True`` if the given term already exists in the database."""

        self.ensure_tables_exist()
        try:
            result = self.execute_scalar(
                SQL_GET_WORD_ID,
                (normalize_term(term), canonicalize_language_tag(language)),
            )
            return result is not None
        except (QueryError, ValueError):
            return False

    def insert_relationship(
        self,
        base_term: str,
        related_term: str,
        relationship_type: str,
        *,
        base_language: str = "en",
        related_language: Optional[str] = None,
        source: str = "unknown",
        confidence: float = 1.0,
    ) -> bool:
        """
        Create a relationship between two terms.

        Args:
            base_term: The source term in the relationship
            related_term: The target term in the relationship
            relationship_type: The type of relationship (e.g., synonym, antonym)
            base_language: Language tag for the source term
            related_language: Language tag for the target term
            source: Provenance label for this assertion
            confidence: Source confidence between zero and one

        Returns:
            True if a new relationship was created, False if it already existed

        Raises:
            DatabaseError: If the relationship cannot be created
            TermNotFoundError: If the base_term doesn't exist in the database
            ValueError: If any parameters are invalid

        Examples:
            >>> success = db.insert_relationship("algorithm", "procedure", "synonym")
            >>> if success:
            ...     print("New relationship created")
            ... else:
            ...     print("Relationship already existed")
        """
        # Validate inputs
        self._validate_relationship_params(base_term, related_term, relationship_type)
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        canonical_base_language = canonicalize_language_tag(base_language)
        canonical_related_language = canonicalize_language_tag(
            related_language or canonical_base_language
        )
        if (
            normalize_term(base_term) == normalize_term(related_term)
            and canonical_base_language == canonical_related_language
        ):
            raise ValueError("Cannot create relationship to self")

        try:
            # Get the word ID (will raise TermNotFoundError if term not found)
            word_id = self.get_word_id(base_term, canonical_base_language)

            # Insert the relationship
            with self.transaction() as conn:
                cursor = conn.execute(
                    SQL_INSERT_RELATIONSHIP,
                    (
                        word_id,
                        unicodedata.normalize("NFC", related_term.strip()),
                        normalize_term(related_term),
                        canonical_related_language,
                        relationship_type.strip(),
                        source.strip(),
                        confidence,
                    ),
                )
                if cursor.rowcount > 0:
                    conn.execute(
                        "UPDATE words SET last_refreshed = ? WHERE id = ?",
                        (time.time(), word_id),
                    )
                # Return True if a new row was inserted
                return cursor.rowcount > 0
        except (sqlite3.Error, TransactionError) as e:
            raise DatabaseError(
                f"Failed to create relationship from '{base_term}' to '{related_term}'",
                e,
            )

    def _validate_relationship_params(
        self, base_term: str, related_term: str, relationship_type: str
    ) -> None:
        """
        Validate parameters for relationship creation.

        Args:
            base_term: The source term
            related_term: The target term
            relationship_type: The relationship type

        Raises:
            ValueError: If any parameters are invalid
        """
        if not isinstance(base_term, str) or not base_term.strip():
            raise ValueError("Base term cannot be empty")
        if not isinstance(related_term, str) or not related_term.strip():
            raise ValueError("Related term cannot be empty")
        if not isinstance(relationship_type, str) or not relationship_type.strip():
            raise ValueError("Relationship type cannot be empty")

    def get_word_entry(self, term: str, language: str = "en") -> WordEntryDict:
        """
        Get complete information about a word including its relationships.

        Retrieves a word entry by term and enriches it with relationship data
        from connected terms. Handles database interactions with proper error
        management and type guarantees.

        Args:
            term: The word or phrase to retrieve
            language: BCP 47 language tag distinguishing homographs

        Returns:
            WordEntryDict: Complete dictionary containing the word's data and relationships
            with structure:
            {
                "id": int,                         # Word identifier
                "language": str,                   # Language code (e.g., "en")
                "term": str,                       # The word itself
                "definition": str,                 # Word definition
                "part_of_speech": str,             # Grammatical category
                "usage_examples": List[str],       # Usage examples list
                "last_refreshed": float,           # Timestamp of last update
                "relationships": List[RelationshipDict]  # Related terms
            }

        Raises:
            TermNotFoundError: If the term doesn't exist in the database
            DatabaseError: For any database access or processing errors

        Examples:
            >>> try:
            ...     entry = db.get_word_entry("algorithm")
            ...     print(f"Definition: {entry['definition']}")
            ...     print(f"Related terms: {len(entry['relationships'])}")
            ... except TermNotFoundError:
            ...     print("Term not found in database")
        """
        try:
            # Get basic word information
            normalized = normalize_term(term)
            canonical_language = canonicalize_language_tag(language)
            row = self.execute_query(
                SQL_GET_WORD_ENTRY, (normalized, canonical_language)
            )
            if not row:
                raise TermNotFoundError(term)

            # Extract word data with proper type safety
            result = row[0]
            word_id_int: int = result["id"]
            term_value: str = result["term"]
            normalized_term: str = result["normalized_term"]
            language_value: str = result["language"]
            script: str = result["script"]
            definition: str = result["definition"] or ""
            part_of_speech: str = result["part_of_speech"] or ""
            usage_examples_str: str = result["usage_examples"] or ""
            source: str = result["source"] or "unknown"
            is_stub = bool(result["is_stub"])
            last_refreshed: float = result["last_refreshed"] or time.time()

            # Parse usage examples with guaranteed type safety
            usage_examples: List[str] = self._parse_usage_examples(usage_examples_str)

            # Get relationships
            relationships = self.get_relationships(str(word_id_int))
            graphemes = self.get_graphemes(word_id_int)
            pronunciations = self.get_pronunciations(word_id_int)

            # Construct and return the complete word entry
            return {
                "id": str(word_id_int),
                "id_int": word_id_int,
                "language": language_value,
                "term": term_value,
                "normalized_term": normalized_term,
                "script": script,
                "definition": definition,
                "part_of_speech": part_of_speech,
                "usage_examples": usage_examples,
                "last_refreshed": last_refreshed,
                "source": source,
                "is_stub": is_stub,
                "relationships": relationships,
                "graphemes": graphemes,
                "pronunciations": pronunciations,
            }
        except QueryError as e:
            raise DatabaseError(f"Database error while retrieving term '{term}'", e)

    def _parse_usage_examples(self, examples_str: str) -> List[str]:
        """
        Parse newline-separated usage examples into a list.

        Args:
            examples_str: String containing newline-separated examples

        Returns:
            List[str]: Individual usage examples as strings

        Examples:
            >>> self._parse_usage_examples("Example one.\\nExample two.")
            ['Example one.', 'Example two.']
            >>> self._parse_usage_examples("")
            []
        """
        return examples_str.split("\n") if examples_str else []

    def get_relationships(self, word_id: str) -> List[RelationshipDict]:
        """
        Get all relationships for a word identified by its ID.

        Args:
            word_id: The database ID of the word

        Returns:
            A list of relationship dictionaries with structure:
            [
                {
                    "related_term": str,         # Term related to the base word
                    "relationship_type": str,    # Type of relationship (e.g., synonym)
                },
                ...
            ]

        Raises:
            QueryError: If database query fails

        Examples:
            >>> relationships = db.get_relationships(42)
            >>> for rel in relationships:
            ...     print(f"{rel['relationship_type']}: {rel['related_term']}")
        """
        try:
            rows = self.execute_query(SQL_GET_RELATIONSHIPS, (word_id,))
            return [
                {
                    "related_term": row["related_term"],
                    "related_normalized_term": row["related_normalized_term"],
                    "related_language": row["related_language"],
                    "relationship_type": row["relationship_type"],
                    "source": row["source"],
                    "confidence": float(row["confidence"]),
                }
                for row in rows
            ]
        except QueryError as e:
            raise QueryError(
                f"Failed to retrieve relationships for word ID {word_id}",
                e,
                SQL_GET_RELATIONSHIPS,
                (word_id,),
            )

    def replace_graphemes(self, word_id: int, graphemes: Sequence[Grapheme]) -> int:
        """Atomically replace ordered grapheme records for a word."""

        try:
            with self.transaction() as conn:
                conn.execute("DELETE FROM graphemes WHERE word_id = ?", (word_id,))
                conn.executemany(
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
                            json.dumps(
                                grapheme.codepoints,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                            json.dumps(
                                grapheme.unicode_names,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                            json.dumps(
                                grapheme.categories,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                            json.dumps(
                                grapheme.combining_classes,
                                separators=(",", ":"),
                            ),
                            grapheme.script,
                        )
                        for grapheme in graphemes
                    ],
                )
            return len(graphemes)
        except (sqlite3.Error, TransactionError) as exc:
            raise DatabaseError(
                f"Failed to replace graphemes for word {word_id}", exc
            ) from exc

    def get_graphemes(self, word_id: int) -> List[GraphemeDict]:
        """Return ordered grapheme records for a word identifier."""

        try:
            rows = self.execute_query(
                """
                SELECT position, text, normalized, codepoints, unicode_names,
                       categories, combining_classes, script
                FROM graphemes
                WHERE word_id = ?
                ORDER BY position ASC
                """,
                (word_id,),
            )
            return [
                {
                    "position": int(row["position"]),
                    "text": str(row["text"]),
                    "normalized": str(row["normalized"]),
                    "codepoints": _json_string_list(row["codepoints"]),
                    "unicode_names": _json_string_list(row["unicode_names"]),
                    "categories": _json_string_list(row["categories"]),
                    "combining_classes": _json_int_list(row["combining_classes"]),
                    "script": str(row["script"]),
                }
                for row in rows
            ]
        except QueryError as exc:
            raise QueryError(
                f"Failed to retrieve graphemes for word {word_id}",
                exc,
                "SELECT ... FROM graphemes WHERE word_id = ?",
                (word_id,),
            ) from exc

    def replace_pronunciations(
        self, word_id: int, pronunciations: Sequence[Pronunciation]
    ) -> int:
        """Atomically replace pronunciations and their phoneme segments."""

        try:
            with self.transaction() as conn:
                conn.execute("DELETE FROM pronunciations WHERE word_id = ?", (word_id,))
                for pronunciation in pronunciations:
                    cursor = conn.execute(
                        """
                        INSERT INTO pronunciations (
                            word_id, notation, transcription, language, dialect,
                            source, confidence, generated, syllable_count,
                            stress_pattern
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            word_id,
                            pronunciation.notation,
                            pronunciation.text,
                            pronunciation.language,
                            pronunciation.dialect or "",
                            pronunciation.source,
                            pronunciation.confidence,
                            int(pronunciation.generated),
                            pronunciation.syllable_count,
                            json.dumps(
                                pronunciation.stress_pattern, separators=(",", ":")
                            ),
                        ),
                    )
                    if cursor.lastrowid is None:
                        raise DatabaseError(
                            f"Pronunciation insert for word {word_id} returned no ID"
                        )
                    pronunciation_id = int(cursor.lastrowid)
                    conn.executemany(
                        """
                        INSERT INTO phonemes (
                            pronunciation_id, position, symbol, base_symbol,
                            stress, syllabic
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                pronunciation_id,
                                phoneme.position,
                                phoneme.symbol,
                                phoneme.base_symbol,
                                phoneme.stress,
                                int(phoneme.syllabic),
                            )
                            for phoneme in pronunciation.phonemes
                        ],
                    )
            return len(pronunciations)
        except (sqlite3.Error, TransactionError) as exc:
            raise DatabaseError(
                f"Failed to replace pronunciations for word {word_id}", exc
            ) from exc

    def get_pronunciations(self, word_id: int) -> List[PronunciationDict]:
        """Return pronunciations with ordered phoneme segments."""

        try:
            rows = self.execute_query(
                """
                SELECT id, notation, transcription, language, dialect, source,
                       confidence, generated, syllable_count, stress_pattern
                FROM pronunciations
                WHERE word_id = ?
                ORDER BY id ASC
                """,
                (word_id,),
            )
            result: List[PronunciationDict] = []
            for row in rows:
                pronunciation_id = int(row["id"])
                phoneme_rows = self.execute_query(
                    """
                    SELECT position, symbol, base_symbol, stress, syllabic
                    FROM phonemes
                    WHERE pronunciation_id = ?
                    ORDER BY position ASC
                    """,
                    (pronunciation_id,),
                )
                phonemes: List[PhonemeDict] = [
                    {
                        "position": int(phoneme["position"]),
                        "symbol": str(phoneme["symbol"]),
                        "base_symbol": str(phoneme["base_symbol"]),
                        "stress": (
                            int(phoneme["stress"])
                            if phoneme["stress"] is not None
                            else None
                        ),
                        "syllabic": bool(phoneme["syllabic"]),
                    }
                    for phoneme in phoneme_rows
                ]
                result.append(
                    {
                        "id": pronunciation_id,
                        "notation": str(row["notation"]),
                        "text": str(row["transcription"]),
                        "language": str(row["language"]),
                        "dialect": str(row["dialect"]) or None,
                        "source": str(row["source"]),
                        "confidence": float(row["confidence"]),
                        "generated": bool(row["generated"]),
                        "syllable_count": int(row["syllable_count"]),
                        "stress_pattern": _json_int_list(row["stress_pattern"]),
                        "phonemes": phonemes,
                    }
                )
            return result
        except QueryError as exc:
            raise QueryError(
                f"Failed to retrieve pronunciations for word {word_id}",
                exc,
            ) from exc

    def get_all_words(self) -> List[WordDataDict]:
        """
        Get a list of all words in the database.

        Returns:
            A list of word data dictionaries containing basic information
            with structure:
            [
                {
                    "id": int,               # Word identifier
                    "term": str,             # The word itself
                    "definition": str,       # Word definition
                    "usage_examples": str,   # Serialized usage examples
                },
                ...
            ]

        Raises:
            QueryError: If retrieving the word list fails

        Examples:
            >>> words = db.get_all_words()
            >>> print(f"Database contains {len(words)} words")
            >>> for word in words[:5]:  # Print first 5 words
            ...     print(f"- {word['term']}")
        """
        try:
            rows = self.execute_query(SQL_GET_ALL_WORDS)
            return [
                {
                    "id": row["id"],
                    "term": row["term"],
                    "definition": row["definition"] or "",
                    "usage_examples": row["usage_examples"] or "",
                    "language": row["language"],
                    "script": row["script"],
                    "last_refreshed": float(row["last_refreshed"]),
                }
                for row in rows
            ]
        except QueryError as e:
            raise QueryError("Failed to retrieve word list", e, SQL_GET_ALL_WORDS)

    def get_updated_words(self, since: float) -> List[WordDataDict]:
        """Return words updated after the given timestamp."""
        try:
            rows = self.execute_query(SQL_GET_UPDATED_WORDS, (since,))
            return [
                {
                    "id": row["id"],
                    "term": row["term"],
                    "definition": row["definition"] or "",
                    "usage_examples": row["usage_examples"] or "",
                    "language": row["language"],
                    "script": row["script"],
                    "last_refreshed": float(row["last_refreshed"]),
                }
                for row in rows
            ]
        except QueryError as e:
            raise QueryError(
                "Failed to retrieve updated words",
                e,
                SQL_GET_UPDATED_WORDS,
                (since,),
            )

    def close(self) -> None:
        """Close the database connection for the current thread."""
        # Close thread-local connection
        if self.connection is not None:
            try:
                self.connection.close()
            except sqlite3.Error:
                pass  # Ignore errors during cleanup
            finally:
                self.connection = None

        # Close thread-local pooled connections
        for conn in self._conn_pool:
            try:
                conn.close()
            except sqlite3.Error:
                pass  # Ignore errors during cleanup
        self._conn_pool.clear()


def _json_string_list(value: object) -> List[str]:
    """Decode a persisted JSON list, retaining only string values."""

    decoded = _json_list(value)
    return [item for item in decoded if isinstance(item, str)]


def _json_int_list(value: object) -> List[int]:
    """Decode a persisted JSON list, retaining non-boolean integers."""

    decoded = _json_list(value)
    return [
        item for item in decoded if isinstance(item, int) and not isinstance(item, bool)
    ]


def _json_list(value: object) -> List[object]:
    """Decode a JSON list defensively for database read paths."""

    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return list(decoded) if isinstance(decoded, list) else []


class RelationshipTypeManager:
    """
    Manages relationship type definitions and operations.

    Provides a layer of abstraction for working with relationship types,
    including validation, normalization, and categorization. Maintains
    a cache for optimized performance during lookups.

    Attributes:
        db_manager: The database manager used for storage operations
        _cache: Internal cache for relationship types organized by category
    """

    def __init__(self, db_manager: DBManager) -> None:
        """
        Initialize with a database manager.

        Args:
            db_manager: The database manager to use for storage

        Examples:
            >>> db = DBManager()
            >>> rel_manager = RelationshipTypeManager(db)
        """
        self.db_manager = db_manager
        # Proper type definition matching actual usage pattern
        self._cache: Dict[str, List[str]] = {}

    @lru_cache(maxsize=128)
    def is_valid_relationship_type(self, relationship_type: str) -> bool:
        """
        Check if a relationship type is valid.

        Validates a relationship type against predefined types or naming
        convention rules. Results are cached for performance optimization.

        Args:
            relationship_type: The relationship type to validate

        Returns:
            bool: True if the relationship type is valid, False otherwise

        Examples:
            >>> if rel_manager.is_valid_relationship_type("synonym"):
            ...     print("Valid relationship type")
            >>> else:
            ...     print("Invalid relationship type")
        """
        # Normalize relationship type for comparison
        normalized_type = relationship_type.lower().strip()

        # Valid if it's in the predefined types or follows naming convention
        return (
            normalized_type in self.get_all_relationship_types()
            or self._follows_naming_convention(normalized_type)
        )

    def _follows_naming_convention(self, relationship_type: str) -> bool:
        """
        Check if a relationship type follows the naming convention.

        Args:
            relationship_type: The relationship type to check

        Returns:
            True if the type follows the convention, False otherwise
        """
        # Allow custom types with appropriate prefixes
        valid_prefixes = ["custom_", "domain_", "project_"]
        return any(relationship_type.startswith(prefix) for prefix in valid_prefixes)

    def get_all_relationship_types(self) -> List[str]:
        """
        Get all defined relationship types.

        Returns:
            A list of all valid relationship types

        Examples:
            >>> types = rel_manager.get_all_relationship_types()
            >>> print(f"Available relationship types: {', '.join(types)}")
        """
        # Ensure the cache is populated
        if not self._cache:
            self._refresh_cache()

        # Flatten the dictionary of categories into a single list
        return [
            relationship_type
            for category in self._cache.values()
            for relationship_type in category
        ]

    def _refresh_cache(self) -> None:
        """
        Refresh the relationship type cache from the database.
        """
        try:
            # Query all distinct relationship types
            rows = self.db_manager.execute_query(
                "SELECT DISTINCT relationship_type FROM relationships"
            )
            # Group by categories
            self._cache = self._categorize_relationship_types(
                [row["relationship_type"] for row in rows]
            )
        except (DatabaseError, sqlite3.Error):
            # Non-fatal error - continue with empty cache
            self._cache = {"other": []}

    def _categorize_relationship_types(self, types: List[str]) -> Dict[str, List[str]]:
        """
        Categorize relationship types into semantic groups.

        Args:
            types: List of relationship types to categorize

        Returns:
            Dictionary mapping categories to lists of relationship types
        """
        categories: Dict[str, List[str]] = {
            "lexical": [],
            "semantic": [],
            "emotional": [],
            "affective": [],
            "other": [],
        }

        for rel_type in types:
            if rel_type.startswith(("synonym", "antonym", "hypernym", "hyponym")):
                categories["lexical"].append(rel_type)
            elif rel_type.startswith(("related_to", "part_of", "has_part")):
                categories["semantic"].append(rel_type)
            elif rel_type.startswith(("evokes", "emotional_")):
                categories["emotional"].append(rel_type)
            elif rel_type.startswith(("positive_", "negative_", "high_", "low_")):
                categories["affective"].append(rel_type)
            else:
                categories["other"].append(rel_type)

        return categories


# Export public elements
__all__ = [
    "DBManager",
    "WordEntryDict",
    "WordDataDict",
    "RelationshipDict",
    "GraphemeDict",
    "PhonemeDict",
    "PronunciationDict",
    "CURRENT_SCHEMA_VERSION",
    "DatabaseError",
    "ConnectionError",
    "QueryError",
    "SchemaError",
    "TransactionError",
    "TermNotFoundError",
    "RelationshipTypeManager",
    "Row",
]
