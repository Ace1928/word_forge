"""
Unified configuration system for Word Forge.

This module centralizes all configuration settings used throughout
the Word Forge system, ensuring consistency across components.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    Final,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
)

# ==========================================
# Type Definitions
# ==========================================

T = TypeVar("T")  # Generic type for configuration values
R = TypeVar("R")  # Return type for type conversion functions

# Refined type definitions for more precise type checking
ConfigValue = Union[Dict[str, Any], List[Any], str, int, float, bool, None]
SerializedConfig = Dict[str, ConfigValue]
PathLike = Union[str, Path]
EnvMapping = Dict[str, Tuple[str, Type[Any]]]


class JSONSerializable(Protocol):
    """Protocol for objects that can be serialized to JSON."""

    def __str__(self) -> str: ...


class ConfigError(Exception):
    """Base exception for configuration errors."""

    pass


class PathError(ConfigError):
    """Raised when a path operation fails."""

    pass


class EnvVarError(ConfigError):
    """Raised when an environment variable cannot be processed."""

    pass


class StorageType(Enum):
    """Storage strategy for vector embeddings."""

    MEMORY = "memory"
    DISK = "disk"

    def __repr__(self) -> str:
        """Provide a clean representation for debugging."""
        return f"StorageType.{self.name}"


# ==========================================
# Core Path Configuration
# ==========================================

# Define project paths with explicit typing for better IDE support
PROJECT_ROOT: Final[Path] = Path("/home/lloyd/eidosian_forge/word_forge")
DATA_ROOT: Final[Path] = PROJECT_ROOT / "data"
LOGS_ROOT: Final[Path] = PROJECT_ROOT / "logs"


# ==========================================
# Configuration Dataclasses
# ==========================================


@dataclass
class DatabaseConfig:
    """
    Database configuration settings.

    Controls SQLite database location, connection parameters,
    performance optimizations, and SQL query templates.
    """

    # Main database path
    db_path: str = str(DATA_ROOT / "word_forge.sqlite")

    # SQLite pragmas for performance optimization
    pragmas: Dict[str, str] = field(
        default_factory=lambda: {
            "foreign_keys": "ON",
            "journal_mode": "WAL",
            "synchronous": "NORMAL",
            "cache_size": "-2000",  # 2MB cache
            "temp_store": "MEMORY",
        }
    )

    # SQL query templates for database operations
    sql_templates: Dict[str, str] = field(
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

    # Connection pool settings (for future scalability)
    pool_size: int = 5
    pool_timeout: float = 30.0

    # Environment variable mapping for configuration overrides
    ENV_VARS: ClassVar[EnvMapping] = {
        "WORD_FORGE_DB_PATH": ("db_path", str),
    }

    @cached_property
    def get_db_path(self) -> Path:
        """Return database path as Path object with caching for performance."""
        return Path(self.db_path)

    def get_connection_uri(self) -> str:
        """Return SQLite connection URI with pragmas attached."""
        params = "&".join(f"{k}={v}" for k, v in self.pragmas.items())
        return f"file:{self.db_path}?{params}"


@dataclass
class VectorizerConfig:
    """
    Vector store configuration settings.

    Controls embedding models, storage parameters, and query templates
    for the vector storage system.
    """

    # Vector embedding model
    model_name: str = "intfloat/multilingual-e5-large-instruct"

    # Optional specific dimension (None = use model's default dimension)
    dimension: Optional[int] = None

    # Path for vector index storage
    index_path: str = str(DATA_ROOT / "vector.index")

    # Storage strategy (memory or persistent disk)
    storage_type: StorageType = StorageType.DISK

    # Default collection name (None = derive from index_path)
    collection_name: Optional[str] = None

    # SQL query templates for content retrieval
    sql_templates: Dict[str, str] = field(
        default_factory=lambda: {
            "get_term_by_id": """
                SELECT term, definition FROM words WHERE id = ?
            """,
            "get_message_text": """
                SELECT text, speaker, conversation_id, timestamp
                FROM conversation_messages WHERE id = ?
            """,
        }
    )

    # Instruction templates for different query types
    instruction_templates: Dict[str, Dict[str, Optional[str]]] = field(
        default_factory=lambda: {
            "search": {
                "task": "Given a web search query, retrieve relevant passages that answer the query",
                "query_prefix": "Instruct: {task}\nQuery: ",
                "document_prefix": None,
            },
            "definition": {
                "task": "Find the definition that best matches this term",
                "query_prefix": "Instruct: {task}\nQuery: ",
                "document_prefix": None,
            },
            "similarity": {
                "task": "Measure semantic similarity between these text passages",
                "query_prefix": "Instruct: {task}\nQuery: ",
                "document_prefix": None,
            },
        }
    )

    # Error handling configuration
    max_retries: int = 3
    failure_cooldown_seconds: float = 60.0

    # Environment variable mapping for configuration overrides
    ENV_VARS: ClassVar[EnvMapping] = {
        "WORD_FORGE_MODEL": ("model_name", str),
        "WORD_FORGE_INDEX_PATH": ("index_path", str),
        "WORD_FORGE_STORAGE_TYPE": ("storage_type", StorageType),
    }

    @cached_property
    def get_index_path(self) -> Path:
        """Return index path as Path object with caching for performance."""
        return Path(self.index_path)


@dataclass
class ParserConfig:
    """
    Configuration for lexical data parser.

    Controls data sources, model settings, and resource paths for
    the system's parsing components.
    """

    # Base directory for lexical resources
    data_dir: str = str(DATA_ROOT)

    # Control language model usage for examples
    enable_model: bool = True

    # Custom language model name (None = use vectorizer's model)
    model_name: Optional[str] = None

    # Resource paths relative to data_dir
    resource_paths: Dict[str, str] = field(
        default_factory=lambda: {
            "openthesaurus": "openthesaurus.jsonl",
            "odict": "odict.json",
            "dbnary": "dbnary.ttl",
            "opendict": "opendict.json",
            "thesaurus": "thesaurus.jsonl",
        }
    )

    # Environment variable mapping for configuration overrides
    ENV_VARS: ClassVar[EnvMapping] = {
        "WORD_FORGE_DATA_DIR": ("data_dir", str),
        "WORD_FORGE_ENABLE_MODEL": ("enable_model", bool),
    }

    def get_full_resource_path(self, resource_name: str) -> Path:
        """
        Get absolute path for a resource.

        Args:
            resource_name: The name of the resource as defined in resource_paths

        Returns:
            Absolute path to the resource

        Raises:
            ConfigError: If the resource name is unknown
        """
        if resource_name not in self.resource_paths:
            raise ConfigError(f"Unknown resource: {resource_name}")
        return Path(self.data_dir) / self.resource_paths[resource_name]


@dataclass
class EmotionConfig:
    """
    Configuration for emotion analysis.

    Controls sentiment analysis parameters, emotion classification rules,
    and database schema for emotion data.
    """

    # VADER configuration
    enable_vader: bool = True

    # Default mixing weights for hybrid sentiment analysis
    vader_weight: float = 0.7
    textblob_weight: float = 0.3

    # Emotion range constraints
    valence_range: Tuple[float, float] = (-1.0, 1.0)  # Negative to positive
    arousal_range: Tuple[float, float] = (0.0, 1.0)  # Calm to excited
    confidence_range: Tuple[float, float] = (0.0, 1.0)  # Certainty level

    # SQL templates for emotion tables
    sql_templates: Dict[str, str] = field(
        default_factory=lambda: {
            "create_word_emotion_table": """
                CREATE TABLE IF NOT EXISTS word_emotion (
                    word_id INTEGER PRIMARY KEY,
                    valence REAL NOT NULL,
                    arousal REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY(word_id) REFERENCES words(id)
                );
            """,
            "create_message_emotion_table": """
                CREATE TABLE IF NOT EXISTS message_emotion (
                    message_id INTEGER PRIMARY KEY,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp REAL NOT NULL
                );
            """,
            "insert_word_emotion": """
                INSERT OR REPLACE INTO word_emotion
                (word_id, valence, arousal, timestamp)
                VALUES (?, ?, ?, ?)
            """,
            "get_word_emotion": """
                SELECT word_id, valence, arousal, timestamp
                FROM word_emotion
                WHERE word_id = ?
            """,
            "insert_message_emotion": """
                INSERT OR REPLACE INTO message_emotion
                (message_id, label, confidence, timestamp)
                VALUES (?, ?, ?, ?)
            """,
            "get_message_emotion": """
                SELECT message_id, label, confidence, timestamp
                FROM message_emotion
                WHERE message_id = ?
            """,
        }
    )

    # Emotion category keywords for classification
    emotion_keywords: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "happiness": ["happy", "joy", "delight", "pleased", "glad", "excited"],
            "sadness": ["sad", "unhappy", "depressed", "down", "miserable", "gloomy"],
            "anger": ["angry", "furious", "enraged", "mad", "irritated", "annoyed"],
            "fear": [
                "afraid",
                "scared",
                "frightened",
                "terrified",
                "anxious",
                "worried",
            ],
            "surprise": ["surprised", "astonished", "amazed", "shocked", "startled"],
            "disgust": ["disgusted", "revolted", "repulsed", "sickened", "appalled"],
            "neutral": ["okay", "fine", "neutral", "indifferent", "average"],
        }
    )

    # Analysis parameters
    min_keyword_confidence: float = 0.3  # Minimum confidence when no keywords found
    keyword_match_weight: float = (
        0.6  # Weight given to keyword matches in classification
    )

    # Environment variable mapping for configuration overrides
    ENV_VARS: ClassVar[EnvMapping] = {
        "WORD_FORGE_ENABLE_VADER": ("enable_vader", bool),
    }


@dataclass
class GraphConfig:
    """
    Configuration for graph visualization and storage.

    Controls layout, rendering, database connections, and sample data
    for the graph visualization system.
    """

    # Default graph export path
    default_export_path: str = str(DATA_ROOT / "word_graph.gexf")

    # Visualization settings
    visualization_path: str = str(DATA_ROOT / "graph_vis.html")
    vis_height: str = "600px"
    vis_width: str = "100%"

    # SQL query templates for graph operations
    sql_templates: Dict[str, str] = field(
        default_factory=lambda: {
            "check_words_table": """
                SELECT name FROM sqlite_master WHERE type='table' AND name='words'
            """,
            "check_relationships_table": """
                SELECT name FROM sqlite_master WHERE type='table' AND name='relationships'
            """,
            "fetch_all_words": """
                SELECT id, term FROM words
            """,
            "fetch_all_relationships": """
                SELECT word_id, related_term, relationship_type FROM relationships
            """,
            "insert_sample_word": """
                INSERT INTO words (term, definition, part_of_speech) VALUES (?, ?, ?)
            """,
            "insert_sample_relationship": """
                INSERT INTO relationships (word_id, related_term, relationship_type) VALUES (?, ?, ?)
            """,
        }
    )

    # Sample data for ensuring graph has content
    sample_words: List[Tuple[str, str, str]] = field(
        default_factory=lambda: [
            ("example", "a representative form or pattern", "noun"),
            ("test", "a procedure for critical evaluation", "noun"),
            (
                "sample",
                "a small part of something intended as representative of the whole",
                "noun",
            ),
            ("word", "a unit of language", "noun"),
            (
                "graph",
                "a diagram showing the relation between variable quantities",
                "noun",
            ),
        ]
    )

    sample_relationships: List[Tuple[str, str, str]] = field(
        default_factory=lambda: [
            ("example", "sample", "synonym"),
            ("example", "test", "related"),
            ("test", "sample", "related"),
            ("word", "term", "synonym"),
            ("graph", "diagram", "synonym"),
        ]
    )

    @cached_property
    def get_export_path(self) -> Path:
        """Return export path as Path object with caching for performance."""
        return Path(self.default_export_path)

    @cached_property
    def get_vis_path(self) -> Path:
        """Return visualization path as Path object with caching for performance."""
        return Path(self.visualization_path)


@dataclass
class QueueConfig:
    """
    Configuration for word queue processing.

    Controls batch sizes, throttling, threading, and other performance
    parameters for the asynchronous word processing queue.
    """

    # Default batch size for processing
    batch_size: int = 50

    # Processing throttle (seconds)
    throttle_seconds: float = 0.1

    # Cache size for seen items lookup optimization
    lru_cache_size: int = 128

    # Queue management settings
    max_queue_size: Optional[int] = None  # None = unlimited

    # Normalization settings
    apply_default_normalization: bool = True

    # Thread safety
    use_threading: bool = True
    lock_type: str = "reentrant"  # "reentrant" or "standard"

    # Performance monitoring
    track_metrics: bool = False

    # Sample size limits
    max_sample_size: int = 100


@dataclass
class ConversationConfig:
    """
    Configuration for conversations.

    Controls status values, metadata, and other settings for
    the conversation management system.
    """

    # Status values
    status_values: Dict[str, str] = field(
        default_factory=lambda: {"active": "ACTIVE", "completed": "COMPLETED"}
    )


@dataclass
class LoggingConfig:
    """
    Configuration for logging.

    Controls log levels, formats, and destinations for
    the Word Forge logging system.
    """

    # Log level
    level: str = "INFO"

    # Log file path (None = log to console only)
    file_path: Optional[str] = str(LOGS_ROOT / "word_forge.log")

    # Log format
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Environment variable mapping for configuration overrides
    ENV_VARS: ClassVar[EnvMapping] = {
        "WORD_FORGE_LOG_LEVEL": ("level", str),
        "WORD_FORGE_LOG_FILE": ("file_path", str),
    }

    @cached_property
    def get_log_path(self) -> Optional[Path]:
        """Return log file path as Path object if set, with caching for performance."""
        return Path(self.file_path) if self.file_path else None


# ==========================================
# Configuration Manager
# ==========================================


class Config:
    """
    Unified configuration for all Word Forge components.

    This class centralizes configuration for database, vectorizer, parser,
    emotion analysis, graph management, queue processing, conversation management,
    and logging systems.

    Attributes:
        database (DatabaseConfig): Database configuration
        vectorizer (VectorizerConfig): Vector embedding configuration
        parser (ParserConfig): Parser configuration
        emotion (EmotionConfig): Emotion analysis configuration
        graph (GraphConfig): Graph visualization configuration
        queue (QueueConfig): Queue processing configuration
        conversation (ConversationConfig): Conversation management configuration
        logging (LoggingConfig): Logging configuration

    Usage:
        from word_forge.config import config

        # Access settings directly
        db_path = config.database.db_path

        # Get Path objects
        db_path_obj = config.database.get_db_path
    """

    def __init__(self) -> None:
        """Initialize configuration with defaults and environment overrides."""
        self.database = DatabaseConfig()
        self.vectorizer = VectorizerConfig()
        self.parser = ParserConfig()
        self.emotion = EmotionConfig()
        self.graph = GraphConfig()
        self.queue = QueueConfig()
        self.conversation = ConversationConfig()
        self.logging = LoggingConfig()

        # Apply environment variable overrides
        self._load_from_env()

        # Ensure data directories exist
        self._ensure_directories()

    def _load_from_env(self) -> None:
        """
        Load configuration from environment variables.

        Processes environment variables based on ENV_VARS mapping
        defined in each configuration class.
        """
        # Process env vars for each config object that has ENV_VARS defined
        for config_name, config_obj in self._get_config_objects():
            if not hasattr(config_obj.__class__, "ENV_VARS"):
                continue

            env_vars = getattr(config_obj.__class__, "ENV_VARS", {})
            for env_var, (attr_name, value_type) in env_vars.items():
                self._set_from_env(env_var, config_obj, attr_name, value_type)

    def _get_config_objects(self) -> List[Tuple[str, Any]]:
        """
        Get all configuration objects with their names.

        Returns:
            List of (name, object) tuples for all configuration components
        """
        return [
            (name, obj)
            for name, obj in self.__dict__.items()
            if name
            in [
                "database",
                "vectorizer",
                "parser",
                "emotion",
                "graph",
                "queue",
                "conversation",
                "logging",
            ]
        ]

    def _set_from_env(
        self,
        env_var: str,
        config_obj: object,
        attr_name: str,
        value_type: Type[T],
    ) -> None:
        """
        Set configuration attribute from environment variable if it exists.

        Args:
            env_var: Environment variable name
            config_obj: Configuration object to modify
            attr_name: Attribute name to set
            value_type: Type to convert value to (str, bool, etc.)

        Raises:
            EnvVarError: If the environment variable has an invalid format
        """
        if env_var not in os.environ:
            return

        value = os.environ[env_var]

        try:
            if value_type is bool:
                # Convert string to boolean
                typed_value: Any = value.lower() in ("true", "yes", "1", "y")
            elif issubclass(value_type, Enum):
                # Convert string to Enum value
                typed_value = value_type(value)
            else:
                # General type conversion
                typed_value = value_type(value)

            setattr(config_obj, attr_name, typed_value)
        except (ValueError, TypeError) as e:
            raise EnvVarError(f"Invalid value '{value}' for {env_var}: {str(e)}") from e

    def _ensure_directories(self) -> None:
        """
        Ensure all required directories exist.

        Creates paths for data storage, logs, and any other required
        directories defined in configuration.

        Raises:
            PathError: If directory creation fails
        """
        try:
            # Ensure base directories exist
            DATA_ROOT.mkdir(parents=True, exist_ok=True)
            LOGS_ROOT.mkdir(parents=True, exist_ok=True)

            # Ensure subdirectories exist for specific data paths
            self._ensure_directory(self.vectorizer.index_path)
            self._ensure_directory(self.graph.default_export_path)
            self._ensure_directory(self.graph.visualization_path)

            # Ensure log directory exists
            if self.logging.file_path:
                self._ensure_directory(self.logging.file_path)
        except (OSError, PermissionError) as e:
            raise PathError(f"Failed to create directory: {str(e)}") from e

    @staticmethod
    def _ensure_directory(file_path: PathLike) -> None:
        """
        Ensure parent directory exists for a given file path.

        Args:
            file_path: Path to a file whose parent directory should exist
        """
        os.makedirs(os.path.dirname(str(file_path)), exist_ok=True)

    def get_full_path(self, path: str) -> Path:
        """
        Convert relative path to absolute using project data directory.

        Args:
            path: Relative path to convert

        Returns:
            Absolute path based on the configured data directory
        """
        return Path(self.parser.data_dir) / path

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the entire configuration to a dictionary.

        Returns:
            Dictionary representation of all configuration components
        """
        return {
            "database": asdict(self.database),
            "vectorizer": serialize_dataclass(self.vectorizer),
            "parser": asdict(self.parser),
            "emotion": serialize_dataclass(self.emotion),
            "graph": serialize_dataclass(self.graph),
            "queue": asdict(self.queue),
            "conversation": asdict(self.conversation),
            "logging": asdict(self.logging),
        }


# ==========================================
# Serialization Utilities
# ==========================================


def serialize_dataclass(obj: Any) -> Dict[str, Any]:
    """
    Serialize a dataclass to a dictionary, handling special types like Enums.

    Args:
        obj: Dataclass instance to serialize

    Returns:
        Dictionary with serialized values
    """
    result = {}
    for key, value in asdict(obj).items():
        if isinstance(value, Enum):
            # Serialize enum as its value
            result[key] = value.value
        elif isinstance(value, tuple) and hasattr(value, "_asdict"):
            # Handle named tuples
            result[key] = value._asdict()
        else:
            result[key] = value
    return result


def serialize_config(obj: Any) -> ConfigValue:
    """
    Convert configuration objects to dictionaries for display.

    Recursively processes configuration objects for JSON serialization,
    handling special types like Enums.

    Args:
        obj: Any configuration object or value to serialize

    Returns:
        A JSON-serializable representation of the configuration
    """
    if hasattr(obj, "__dict__"):
        d: Dict[str, ConfigValue] = {}
        for key, value in obj.__dict__.items():
            if not key.startswith("_"):
                d[key] = serialize_config(value)
        return d
    elif isinstance(obj, (list, tuple)):
        return [serialize_config(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: serialize_config(v) for k, v in obj.items()}
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


# ==========================================
# Global Configuration Instance
# ==========================================

# Global configuration instance
config = Config()


# ==========================================
# CLI Interface
# ==========================================


def main() -> None:
    """
    Display current configuration settings.

    This function demonstrates how to access configuration values and
    print the current configuration state for diagnostic purposes.
    """
    print("Word Forge Configuration")
    print("=======================")

    # Get JSON representation of config
    config_dict = serialize_config(config)
    config_json = json.dumps(config_dict, indent=2)

    # Print config sections
    print(config_json)

    # Print examples of accessing specific settings
    print("\nAccessing specific settings:")
    print(f"Database path: {config.database.db_path}")
    print(f"Vector model: {config.vectorizer.model_name}")
    print(f"Vector dimension: {config.vectorizer.dimension or 'Default from model'}")
    print(f"Data directory: {config.parser.data_dir}")

    # Print sample path resolutions
    print("\nPath resolution examples:")
    print(f"Full database path: {config.database.get_db_path.absolute()}")
    print(
        f"OpenThesaurus path: {config.parser.get_full_resource_path('openthesaurus').absolute()}"
    )
    print(f"Vector index path: {config.vectorizer.get_index_path.absolute()}")


if __name__ == "__main__":
    main()
