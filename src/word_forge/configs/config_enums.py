from enum import Enum


class StorageType(Enum):
    """Storage strategy for vector embeddings."""

    MEMORY = "memory"
    DISK = "disk"

    def __repr__(self) -> str:
        """Provide a clean representation for debugging."""
        return f"StorageType.{self.name}"


# Performance profile presets for queue processing
class QueuePerformanceProfile(Enum):
    """Performance profiles for queue processing operations."""

    LOW_LATENCY = "low_latency"  # Optimize for immediate response
    HIGH_THROUGHPUT = "high_throughput"  # Optimize for maximum processing volume
    BALANCED = "balanced"  # Balance between latency and throughput
    MEMORY_EFFICIENT = "memory_efficient"  # Minimize memory usage

    def __repr__(self) -> str:
        """Provide a clean representation for debugging."""
        return f"QueuePerformanceProfile.{self.name}"


# Retention policy for conversations
class ConversationRetentionPolicy(Enum):
    """Retention policy options for conversation history."""

    KEEP_FOREVER = "keep_forever"
    DELETE_AFTER_30_DAYS = "delete_after_30_days"
    DELETE_AFTER_90_DAYS = "delete_after_90_days"
    DELETE_AFTER_1_YEAR = "delete_after_1_year"

    def __repr__(self) -> str:
        """Provide a clean representation for debugging."""
        return f"ConversationRetentionPolicy.{self.name}"


# Structure defining conversation export formats
class ConversationExportFormat(Enum):
    """Export format options for conversation data."""

    JSON = "json"
    MARKDOWN = "markdown"
    TEXT = "text"
    HTML = "html"

    def __repr__(self) -> str:
        """Provide a clean representation for debugging."""
        return f"ConversationExportFormat.{self.name}"


class VectorModelType(Enum):
    """Vector embedding model types supported by the system."""

    TRANSFORMER = "transformer"  # Transformer-based embedding models
    SENTENCE = "sentence"  # Sentence embedding models
    WORD = "word"  # Word embedding models
    CUSTOM = "custom"  # Custom embedding implementations

    def __repr__(self) -> str:
        """Provide a clean representation for debugging."""
        return f"VectorModelType.{self.name}"


class VectorIndexStatus(Enum):
    """Status of a vector index."""

    UNINITIALIZED = "uninitialized"  # Index has not been created
    READY = "ready"  # Index is ready for use
    BUILDING = "building"  # Index is currently being built
    ERROR = "error"  # Index encountered an error

    def __repr__(self) -> str:
        """Provide a clean representation for debugging."""
        return f"VectorIndexStatus.{self.name}"


class GraphLayoutAlgorithm(Enum):
    """Layout algorithms for knowledge graph visualization."""

    FORCE_DIRECTED = "force_directed"  # Force-directed graph drawing
    CIRCULAR = "circular"  # Circular layout
    HIERARCHICAL = "hierarchical"  # Tree-like hierarchical layout
    SPECTRAL = "spectral"  # Spectral layout using eigenvectors
    RADIAL = "radial"  # Radial layout around central node

    def __repr__(self) -> str:
        """Provide a clean representation for debugging."""
        return f"GraphLayoutAlgorithm.{self.name}"


class GraphColorScheme(Enum):
    """Color schemes for graph visualization."""

    SEMANTIC = "semantic"  # Colors based on semantic relationships
    CATEGORY = "category"  # Colors based on word categories
    SENTIMENT = "sentiment"  # Colors based on sentiment analysis
    GRADIENT = "gradient"  # Gradient colors based on relationship strength
    MONOCHROME = "monochrome"  # Single color with varying intensity

    def __repr__(self) -> str:
        """Provide a clean representation for debugging."""
        return f"GraphColorScheme.{self.name}"


# Log format templates with predefined options
class LogFormatTemplate(Enum):
    """Standard logging format templates."""

    SIMPLE = "%(message)s"
    STANDARD = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DETAILED = (
        "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
    )
    JSON = '{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'

    def __repr__(self) -> str:
        """Provide a clean representation for debugging."""
        return f"LogFormatTemplate.{self.name}"


# Log rotation strategies
class LogRotationStrategy(Enum):
    """Log file rotation strategies."""

    SIZE = "size"  # Rotate based on file size
    TIME = "time"  # Rotate based on time intervals
    NONE = "none"  # No rotation

    def __repr__(self) -> str:
        """Provide a clean representation for debugging."""
        return f"LogRotationStrategy.{self.name}"


# Log output destinations
class LogDestination(Enum):
    """Logging output destinations."""

    CONSOLE = "console"  # Log to console only
    FILE = "file"  # Log to file only
    BOTH = "both"  # Log to both console and file
    SYSLOG = "syslog"  # Log to system log

    def __repr__(self) -> str:
        """Provide a clean representation for debugging."""
        return f"LogDestination.{self.name}"


class DatabaseDialect(Enum):
    """Database dialects supported by the system."""

    SQLITE = "sqlite"  # SQLite file-based database
    POSTGRES = "postgres"  # PostgreSQL database
    MYSQL = "mysql"  # MySQL database
    MEMORY = "memory"  # In-memory database (for testing)

    def __repr__(self) -> str:
        """Provide a clean representation for debugging."""
        return f"DatabaseDialect.{self.name}"
