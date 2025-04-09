"""
Configuration Essentials Module

This module serves as the type system foundation for WordForge's configuration architecture,
establishing a comprehensive type system that enforces correctness through static analysis
rather than runtime checking.

Key Features:
    - Type-safe configuration definitions with strict interfaces
    - Result pattern for exception-free error propagation
    - Comprehensive serialization utilities for configuration persistence
    - Protocol-based interfaces for consistent component behaviors
    - Fine-grained error classification system with context preservation
    - Thread-safe execution measurement and distributed tracing support

Type Categories:
    - Json types: Primitives, dictionaries, and lists for serialization
    - Configuration types: Component registries and environment mappings
    - Result types: Monadic error handling patterns
    - Worker types: Task priority, worker state management, circuit breakers
    - Domain-specific types: Query types, vector operations, graph processing
    - Template types: Structured definitions for consistent formatting

Performance Support:
    - Execution metrics collection for critical operations
    - Circuit breaker pattern implementation for failure isolation
    - Worker pool configurations for parallel processing
    - Tracing context for distributed operation monitoring

Error Handling:
    - Fine-grained error categorization (validation, resource, business, etc.)
    - Severity classification for appropriate handling strategies
    - Context preservation for diagnostic traceability
    - Monadic Result type for functional error propagation

Usage Guidelines:
    - Import specific types rather than using wildcard imports
    - Leverage type checking tools (mypy, pyright) with this module
    - Follow the Result pattern for cross-component error handling
    - Use the provided serialization utilities for configuration persistence

This module embodies the principle that strong typing prevents more errors than
exception handling ever could, making it the foundation of WordForge's reliability.
This module defines all essential types and constants used throughout the
configuration system, providing a consistent way to handle various
configuration-related tasks, including serialization, path management,
and error handling.
"""

import json
import resource
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Final,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeAlias,
    TypeVar,
    Union,
    cast,
)

# ==========================================
# Generic Type Variables
# ==========================================

# Generic type parameter for configuration value types
T = TypeVar("T")
# Define a contravariant type variable for input types
T_contra = TypeVar("T_contra", contravariant=True)

# Generic type parameter for function return types
R = TypeVar("R")

# Type variable bound to ConfigComponent protocol for generic configuration handling
C = TypeVar("C", bound="ConfigComponent")

# Additional generic variables for functional patterns
K = TypeVar("K")  # Key type for mappings
V = TypeVar("V")  # Value type for mappings
E = TypeVar("E")  # Error type for Result pattern

# ==========================================
# Project Paths
# ==========================================

# Define project paths with explicit typing for better IDE support
PROJECT_ROOT: Final[Path] = Path("/home/lloyd/eidosian_forge/word_forge")
DATA_ROOT: Final[Path] = PROJECT_ROOT / "data"
LOGS_ROOT: Final[Path] = PROJECT_ROOT / "logs"

# ==========================================
# Basic Type Definitions
# ==========================================

# JSON-related type definitions for configuration serialization
JsonPrimitive = Union[str, int, float, bool, None]
JsonDict = Dict[str, "JsonValue"]  # Forward reference for recursion
JsonList = List["JsonValue"]  # Forward reference for recursion
JsonValue: TypeAlias = Union[JsonDict, JsonList, JsonPrimitive]  # Explicit TypeAlias

# Configuration-specific type aliases
ConfigValue: TypeAlias = JsonValue  # Use defined JsonValue

# Logging
LoggingConfigDict: TypeAlias = Dict[str, Any]  # Define LoggingConfig as Dict
ValidationError: TypeAlias = str
FormatStr: TypeAlias = str
LogFilePathStr: TypeAlias = Optional[str]

# Function type for validation handlers
ValidationFunction: TypeAlias = Callable[
    [LoggingConfigDict, List[ValidationError]], None
]  # Use LoggingConfigDict
EnvVarType: TypeAlias = Union[str, int, float, bool, None]  # Define EnvVarType

# Type alias for serialized configuration data
SerializedConfig: TypeAlias = JsonDict

# Type alias for path-like objects
PathLike: TypeAlias = Union[str, Path]

# Type alias for environment variable mapping in ConfigComponent
EnvMapping: TypeAlias = Dict[str, Tuple[str, Callable[[str], EnvVarType]]]

# Type alias for the name of a configuration component
ComponentName: TypeAlias = str

# Type alias for a registry mapping component names to instances
ComponentRegistry: TypeAlias = Dict[ComponentName, "ConfigComponent"]

# Type alias for a dictionary representing a configuration section
ConfigDict: TypeAlias = Dict[str, ConfigValue]

# ==========================================
# Configuration Component Protocol
# ==========================================


class ConfigComponent(Protocol):
    """Protocol defining the interface for configuration components."""

    ENV_VARS: ClassVar[EnvMapping]

    def load_from_env(self) -> None:
        """Load configuration values from environment variables."""
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the component's configuration to a dictionary."""
        ...

    @classmethod
    def from_dict(cls: type[C], data: Dict[str, Any]) -> C:
        """Deserialize a component configuration from a dictionary."""
        ...


# ==========================================
# Result and Error Handling Types
# ==========================================


class ErrorCategory(Enum):
    """Categories of errors for systematic handling and reporting."""

    VALIDATION = auto()  # Input validation failures
    RESOURCE = auto()  # Resource availability issues
    BUSINESS = auto()  # Business rule violations
    EXTERNAL = auto()  # External system failures
    UNEXPECTED = auto()  # Unexpected failures
    CONFIGURATION = auto()  # Configuration errors
    SECURITY = auto()  # Security-related issues


class ErrorSeverity(Enum):
    """Severity levels for errors to guide handling strategies."""

    FATAL = auto()  # System cannot continue operation
    ERROR = auto()  # Operation failed completely
    WARNING = auto()  # Operation completed with issues
    INFO = auto()  # Operation completed with non-critical adjustments


@dataclass(frozen=True)
class Error:
    """
    Immutable error object with comprehensive context for accurate diagnostics.

    Provides a unified structure for error handling across the system,
    with severity classification, error codes, and contextual information.

    Attributes:
        message: Human-readable error message
        code: Machine-readable error code
        category: Error category for classification
        severity: Error severity for handling strategy
        context: Dictionary of additional contextual information (can be any primitive type)
        trace: Optional stack trace for debugging
    """

    message: str
    code: str
    category: ErrorCategory
    severity: ErrorSeverity
    context: Dict[str, Any] = field(default_factory=dict)
    trace: Optional[str] = None

    @classmethod
    def create(
        cls,
        message: str,
        code: str,
        category: ErrorCategory,
        severity: ErrorSeverity,
        context: Optional[Dict[str, Any]] = None,
    ) -> "Error":
        """
        Factory method for creating errors with standardized formatting.

        Args:
            message: Human-readable error description
            code: Machine-readable error code
            category: Error category for classification
            severity: Error severity for handling strategy
            context: Optional dictionary of contextual information (Any value type)

        Returns:
            Fully initialized Error object
        """
        # Ensure context values are serializable (convert non-primitives to string)
        serializable_context: Dict[str, Any] = {}
        if context:
            for k, v in context.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    serializable_context[k] = v
                else:
                    try:
                        # Attempt JSON serialization for complex types
                        json.dumps({k: v})
                        serializable_context[k] = v
                    except TypeError:
                        serializable_context[k] = str(
                            v
                        )  # Fallback to string representation

        return cls(
            message=message,
            code=code,
            category=category,
            severity=severity,
            context=serializable_context,
            trace=traceback.format_exc(),
        )


@dataclass(frozen=True)
class Result(Generic[T]):
    """Monadic result type for functional error handling without exceptions.

    Encapsulates either a success value (`value`) or an error object (`error`).
    Provides methods for safe unwrapping, mapping, and checking status.

    Attributes:
        value: The success value (present if `is_success` is True).
        error: The error object (present if `is_success` is False).
    """

    value: Optional[T] = None
    error: Optional[Error] = None

    @property
    def is_success(self) -> bool:
        """Check if the result represents a successful operation."""
        return self.error is None

    @property
    def is_failure(self) -> bool:
        """Check if the result represents a failed operation."""
        return self.error is not None

    def unwrap(self) -> T:
        """Return the success value, raising ValueError if the result is a failure.

        Raises:
            ValueError: If the result contains an error.

        Returns:
            The success value of type T.
        """
        if not self.is_success:
            # Ensure error is not None before accessing attributes
            error_msg = "Cannot unwrap a failed result"
            if self.error:
                error_msg += f": {self.error.code} - {self.error.message}"
            raise ValueError(error_msg)
        # Cast is safe here due to the is_success check
        return cast(T, self.value)

    def map(self, func: Callable[[T], R]) -> "Result[R]":
        """Apply a function to the success value, passing through errors.

        If the result is successful, applies `func` to the value and returns
        a new Result containing the transformed value. If the result is a failure,
        returns a new Result containing the original error.

        Args:
            func: The function to apply to the success value.

        Returns:
            A new Result object containing the transformed value or the original error.
        """
        if self.is_failure:
            # Type checker needs explicit type argument for failure case
            return Result[R](error=self.error)
        try:
            # Cast is safe here
            new_value = func(cast(T, self.value))
            return Result[R].success(new_value)
        except Exception as e:
            # Capture exceptions during mapping as new errors
            error = Error.create(
                f"Error during Result.map operation: {e}",
                "MAP_ERROR",
                ErrorCategory.UNEXPECTED,
                ErrorSeverity.ERROR,
                context={
                    "original_value": str(self.value)[:100]
                },  # Log truncated value
            )
            return Result[R].failure(error)

    @classmethod
    def success(cls, value: T) -> "Result[T]":
        """Create a success Result."""
        return cls(value=value)

    @classmethod
    def failure(
        cls,
        code: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        category: ErrorCategory = ErrorCategory.UNEXPECTED,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
    ) -> "Result[T]":
        """Create a failure Result using the Error factory."""
        error = Error.create(
            message=message,
            code=code,
            category=category,
            severity=severity,
            context=context,
        )
        return cls(error=error)


# ==========================================
# Performance and Tracing Types
# ==========================================


@dataclass
class ExecutionMetrics:
    """Dataclass for collecting execution performance metrics."""

    operation_name: str
    thread_id: int
    started_at: float
    duration_ms: float = 0.0
    cpu_time_ms: float = 0.0
    memory_before_bytes: int = 0
    memory_after_bytes: int = 0
    memory_delta_kb: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)


@contextmanager
def measure_execution(
    operation_name: str, context: Optional[Dict[str, Any]] = None
) -> Iterator[ExecutionMetrics]:
    """
    Context manager to measure execution time and resource usage.

    Args:
        operation_name: Name of the operation being measured
        context: Optional dictionary of contextual information

    Yields:
        ExecutionMetrics object with collected metrics
    """
    metrics_obj = ExecutionMetrics(
        operation_name=operation_name,
        thread_id=threading.get_ident(),
        started_at=time.time(),
        context=context or {},
    )

    start_time = time.time()
    start_cpu_time = time.process_time()
    try:
        mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metrics_obj.memory_before_bytes = mem_before * 1024
    except ImportError:
        metrics_obj.memory_before_bytes = 0
    except AttributeError:  # Handle cases where resource module might lack getrusage
        metrics_obj.memory_before_bytes = 0

    try:
        yield metrics_obj
    finally:
        metrics_obj.duration_ms = (time.time() - start_time) * 1000
        metrics_obj.cpu_time_ms = (time.process_time() - start_cpu_time) * 1000
        try:
            mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            metrics_obj.memory_after_bytes = mem_after * 1024
            metrics_obj.memory_delta_kb = (
                metrics_obj.memory_after_bytes - metrics_obj.memory_before_bytes
            ) / 1024
        except ImportError:
            metrics_obj.memory_after_bytes = 0
            metrics_obj.memory_delta_kb = 0
        except AttributeError:
            metrics_obj.memory_after_bytes = 0
            metrics_obj.memory_delta_kb = 0


# ==========================================
# Domain Specific Types (Examples)
# ==========================================


class ConversationStatusValue(str, Enum):
    ACTIVE = "ACTIVE"
    PENDING = "PENDING REVIEW"
    COMPLETED = "COMPLETED"
    ARCHIVED = "ARCHIVED"
    DELETED = "DELETED"


ConversationStatusMap: TypeAlias = Dict[str, ConversationStatusValue]
ConversationMetadataSchema: TypeAlias = Dict[str, Optional[Any]]


class ConversationRetentionPolicy(Enum):
    KEEP_FOREVER = auto()
    DELETE_AFTER_DAYS = auto()
    ARCHIVE_AFTER_DAYS = auto()


class ConversationExportFormat(str, Enum):
    JSON = "json"
    MARKDOWN = "md"
    TEXT = "txt"


# Lexical data type
LexicalDataset: TypeAlias = Dict[str, Any]  # Placeholder, refine if structure is known

# ==========================================
# Module Exports
# ==========================================

__all__ = [
    # Basic Types
    "JsonPrimitive",
    "JsonDict",
    "JsonList",
    "JsonValue",
    "ConfigValue",
    "LoggingConfigDict",
    "ValidationError",
    "FormatStr",
    "LogFilePathStr",
    "ValidationFunction",
    "EnvVarType",
    "SerializedConfig",
    "PathLike",
    "EnvMapping",
    "ComponentName",
    "ComponentRegistry",
    "ConfigDict",
    # Paths
    "PROJECT_ROOT",
    "DATA_ROOT",
    "LOGS_ROOT",
    # Error Handling
    "ErrorCategory",
    "ErrorSeverity",
    "Error",
    "Result",
    # Performance/Tracing
    "ExecutionMetrics",
    "measure_execution",
    # Configuration Component Protocol
    "ConfigComponent",
    # Domain Specific Enums/Types
    "ConversationStatusValue",
    "ConversationStatusMap",
    "ConversationMetadataSchema",
    "ConversationRetentionPolicy",
    "ConversationExportFormat",
    "LexicalDataset",
    # Generics
    "T",
    "R",
    "K",
    "V",
    "E",
    "C",
    "T_contra",
]
