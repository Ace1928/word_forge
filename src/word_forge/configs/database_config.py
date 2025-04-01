"""
Database configuration system for Word Forge.

This module defines the configuration schema for database connections,
performance parameters, SQL templates, and connection pooling used
throughout the Word Forge system.

Architecture:
    ┌─────────────────────┐
    │   DatabaseConfig    │
    └───────────┬─────────┘
                │
    ┌───────────┴─────────┐
    │     Components      │
    └─────────────────────┘
    ┌─────┬─────┬─────┬───────┬─────┐
    │Conn │Pool │SQL  │Pragmas│Trans│
    └─────┴─────┴─────┴───────┴─────┘
"""

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Dict, Union

from word_forge.configs.config_essentials import (
    DATA_ROOT,
    ConnectionPoolMode,
    DatabaseConfigError,
    DatabaseDialect,
    PathLike,
    SQLitePragmas,
    SQLTemplates,
    TransactionIsolationLevel,
)
from word_forge.configs.config_types import EnvMapping


@dataclass
class DatabaseConfig:
    """
    Database configuration for SQL database connections.

    Controls SQLite database location, connection parameters, SQL query templates,
    performance optimizations, and connection pooling settings.

    Attributes:
        db_path: Path to the SQLite database file
        dialect: Database dialect (currently supports SQLite)
        pragmas: SQLite pragmas for performance optimization
        sql_templates: SQL query templates for database operations
        pool_mode: Connection pooling mode (fixed, dynamic, none)
        pool_size: Connection pool size (for fixed/dynamic modes)
        pool_timeout: Connection pool timeout in seconds
        pool_recycle: Time in seconds between connection recycling
        isolation_level: Transaction isolation level
        enable_foreign_keys: Whether to enforce foreign key constraints
        enable_wal_mode: Whether to use WAL journal mode (SQLite only)
        page_size: Database page size in bytes (SQLite only)
        cache_size: Memory pages to use for database cache (SQLite only)

    Usage:
        from word_forge.config import config

        # Get database path
        db_path = config.database.get_db_path

        # Get connection URI with pragmas
        uri = config.database.get_connection_uri()

        # Create a new configuration with a different path
        test_config = config.database.with_path("path/to/test.db")
    """

    # Core database settings
    db_path: str = str(DATA_ROOT / "word_forge.sqlite")
    dialect: DatabaseDialect = DatabaseDialect.SQLITE

    # SQLite pragmas for performance optimization
    pragmas: SQLitePragmas = field(
        default_factory=lambda: {
            "foreign_keys": "ON",
            "journal_mode": "WAL",
            "synchronous": "NORMAL",
            "cache_size": "-2000",  # 2MB cache
            "temp_store": "MEMORY",
        }
    )

    # SQL query templates for database operations
    sql_templates: SQLTemplates = field(
        default_factory=lambda: {
            "create_words_table": """
                CREATE TABLE IF NOT EXISTS words (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    term TEXT UNIQUE NOT NULL,
                    definition TEXT,
                    part_of_speech TEXT,
                    usage_examples TEXT,
                    last_refreshed REAL
                );
            """,
            "create_relationships_table": """
                CREATE TABLE IF NOT EXISTS relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    word_id INTEGER NOT NULL,
                    related_term TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    FOREIGN KEY(word_id) REFERENCES words(id)
                );
            """,
            "create_word_id_index": """
                CREATE INDEX IF NOT EXISTS idx_relationships_word_id
                ON relationships(word_id);
            """,
            "create_unique_relationship_index": """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_relationship
                ON relationships(word_id, related_term, relationship_type);
            """,
        }
    )

    # Connection pool settings
    pool_mode: ConnectionPoolMode = "fixed"
    pool_size: int = 5
    pool_timeout: float = 30.0
    pool_recycle: int = 3600  # Recycle connections after 1 hour

    # Transaction settings
    isolation_level: TransactionIsolationLevel = "SERIALIZABLE"

    # Performance settings
    enable_foreign_keys: bool = True
    enable_wal_mode: bool = True
    page_size: int = 4096  # 4KB pages
    cache_size: int = 2000  # 2000 pages (~8MB)

    # Environment variable mapping for configuration overrides
    ENV_VARS: ClassVar[EnvMapping] = {
        "WORD_FORGE_DB_PATH": ("db_path", str),
        "WORD_FORGE_DB_DIALECT": ("dialect", DatabaseDialect),
        "WORD_FORGE_DB_POOL_SIZE": ("pool_size", int),
        "WORD_FORGE_DB_POOL_MODE": ("pool_mode", str),
        "WORD_FORGE_DB_ISOLATION": ("isolation_level", str),
        "WORD_FORGE_DB_FOREIGN_KEYS": ("enable_foreign_keys", bool),
    }

    # ==========================================
    # Cached Properties
    # ==========================================

    @cached_property
    def db_path_object(self) -> Path:
        """
        Database path as a Path object.

        Returns:
            Path: Object representing the database location.
        """
        return Path(self.db_path)

    @cached_property
    def get_db_path(self) -> Path:
        """
        Get database path as a validated Path object.

        Returns:
            Path: Database file location.

        Raises:
            DatabaseConfigError: If path is invalid
        """
        path = self.db_path_object

        # For SQLite, ensure parent directory exists
        if self.dialect == DatabaseDialect.SQLITE:
            parent_dir = path.parent
            if not parent_dir.exists():
                raise DatabaseConfigError(
                    f"Database parent directory does not exist: {parent_dir}"
                )

        return path

    @cached_property
    def effective_pragmas(self) -> Dict[str, str]:
        """
        Get the effective pragmas with computed values.

        Ensures all explicitly set performance parameters are
        reflected in the pragmas dictionary.

        Returns:
            Dict[str, str]: Effective pragmas dictionary
        """
        # Start with user-defined pragmas
        result = dict(self.pragmas)

        # Apply explicit settings if not present in pragmas
        if self.enable_foreign_keys and "foreign_keys" not in result:
            result["foreign_keys"] = "ON"

        if self.enable_wal_mode and "journal_mode" not in result:
            result["journal_mode"] = "WAL"

        # Apply cache size setting
        if "cache_size" not in result:
            result["cache_size"] = str(-self.cache_size)  # Negative means KB

        # Apply page size setting
        if "page_size" not in result:
            result["page_size"] = str(self.page_size)

        return result

    # ==========================================
    # Public Methods
    # ==========================================

    def get_connection_uri(self) -> str:
        """
        Build SQLite connection URI with pragmas.

        Returns:
            str: SQLite connection string with pragmas as query parameters.

        Raises:
            DatabaseConfigError: If dialect is not SQLite
        """
        if self.dialect != DatabaseDialect.SQLITE:
            raise DatabaseConfigError(
                f"Connection URI for {self.dialect.name} is not implemented"
            )

        params = "&".join(f"{k}={v}" for k, v in self.effective_pragmas.items())
        return f"file:{self.db_path}?{params}"

    def get_pool_settings(self) -> Dict[str, Union[str, int, float]]:
        """
        Get connection pool settings as a dictionary.

        Returns:
            Dict[str, Union[str, int, float]]: Pool configuration for database engines
        """
        return {
            "mode": self.pool_mode,
            "size": self.pool_size if self.pool_mode != "none" else 0,
            "timeout": self.pool_timeout,
            "recycle": self.pool_recycle,
        }

    def with_path(self, path: PathLike) -> "DatabaseConfig":
        """
        Create a new instance with a modified path.

        Args:
            path: New database path (string or Path object)

        Returns:
            DatabaseConfig: New instance with updated path
        """
        path_str = str(path)
        return DatabaseConfig(
            db_path=path_str,
            dialect=self.dialect,
            pragmas=self.pragmas,
            sql_templates=self.sql_templates,
            pool_mode=self.pool_mode,
            pool_size=self.pool_size,
            pool_timeout=self.pool_timeout,
            pool_recycle=self.pool_recycle,
            isolation_level=self.isolation_level,
            enable_foreign_keys=self.enable_foreign_keys,
            enable_wal_mode=self.enable_wal_mode,
            page_size=self.page_size,
            cache_size=self.cache_size,
        )

    def with_dialect(self, dialect: DatabaseDialect) -> "DatabaseConfig":
        """
        Create a new instance with a different database dialect.

        Args:
            dialect: New database dialect

        Returns:
            DatabaseConfig: New instance with updated dialect
        """
        return DatabaseConfig(
            db_path=self.db_path,
            dialect=dialect,
            pragmas=self.pragmas,
            sql_templates=self.sql_templates,
            pool_mode=self.pool_mode,
            pool_size=self.pool_size,
            pool_timeout=self.pool_timeout,
            pool_recycle=self.pool_recycle,
            isolation_level=self.isolation_level,
            enable_foreign_keys=self.enable_foreign_keys,
            enable_wal_mode=self.enable_wal_mode,
            page_size=self.page_size,
            cache_size=self.cache_size,
        )

    def optimize_for_reads(self) -> "DatabaseConfig":
        """
        Create a new instance optimized for read operations.

        Returns:
            DatabaseConfig: New read-optimized configuration
        """
        read_pragmas = dict(self.pragmas)
        read_pragmas.update(
            {
                "synchronous": "NORMAL",  # Less durability, more speed
                "journal_mode": "WAL",  # Allow concurrent reads/writes
                "cache_size": str(-8000),  # Larger cache for read-heavy workloads
                "temp_store": "MEMORY",  # Keep temp tables in memory
            }
        )

        return DatabaseConfig(
            db_path=self.db_path,
            dialect=self.dialect,
            pragmas=read_pragmas,
            sql_templates=self.sql_templates,
            pool_mode=self.pool_mode,
            pool_size=max(self.pool_size, 8),  # Larger pool for more concurrent reads
            pool_timeout=self.pool_timeout,
            pool_recycle=self.pool_recycle,
            isolation_level="READ_COMMITTED",  # Lower isolation level for better concurrency
            enable_foreign_keys=self.enable_foreign_keys,
            enable_wal_mode=True,  # WAL is critical for read performance
            page_size=self.page_size,
            cache_size=8000,  # Larger cache size
        )

    def optimize_for_writes(self) -> "DatabaseConfig":
        """
        Create a new instance optimized for write operations.

        Returns:
            DatabaseConfig: New write-optimized configuration
        """
        write_pragmas = dict(self.pragmas)
        write_pragmas.update(
            {
                "synchronous": "NORMAL",  # Balance between durability and performance
                "journal_mode": "WAL",  # Better for writes than DELETE
                "cache_size": str(-4000),  # Moderate cache
                "mmap_size": str(1073741824),  # 1GB memory map
            }
        )

        return DatabaseConfig(
            db_path=self.db_path,
            dialect=self.dialect,
            pragmas=write_pragmas,
            sql_templates=self.sql_templates,
            pool_mode=self.pool_mode,
            pool_size=max(self.pool_size, 4),  # Moderate pool size for writes
            pool_timeout=self.pool_timeout,
            pool_recycle=1800,  # More frequent recycling for write connections
            isolation_level=self.isolation_level,
            enable_foreign_keys=self.enable_foreign_keys,
            enable_wal_mode=True,  # WAL is also good for writes
            page_size=8192,  # Larger page size for bulk writes
            cache_size=4000,  # Moderate cache size
        )

    def validate(self) -> None:
        """
        Validate the entire configuration for consistency and correctness.

        Raises:
            DatabaseConfigError: If any validation fails
        """
        errors = []

        # Validate dialect-specific settings
        if self.dialect == DatabaseDialect.SQLITE:
            # Verify db_path is set for SQLite
            if not self.db_path:
                errors.append("SQLite database path cannot be empty")

            # Validate pool settings
            if self.pool_size <= 0 and self.pool_mode != "none":
                errors.append(
                    f"Pool size must be positive when mode is {self.pool_mode}, got {self.pool_size}"
                )

            # Validate timeout
            if self.pool_timeout <= 0:
                errors.append(f"Pool timeout must be positive, got {self.pool_timeout}")

        if errors:
            raise DatabaseConfigError(
                f"Configuration validation failed: {'; '.join(errors)}"
            )


# ==========================================
# Module Exports
# ==========================================

__all__ = [
    # Configuration class
    "DatabaseConfig",
    # Type definitions
    "DatabaseDialect",
    "TransactionIsolationLevel",
    "ConnectionPoolMode",
    "SQLitePragmas",
    "SQLTemplates",
    # Error types
    "DatabaseConfigError",
]
