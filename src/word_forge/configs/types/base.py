"""
Base type definitions for Word Forge configuration system.

This module provides foundational type variables and type aliases used
throughout the configuration system.

Type Variables:
    T: Generic type parameter for values
    T_contra: Contravariant type for input types
    R: Return type parameter
    C: ConfigComponent-bound type
    K: Key type for mappings
    V: Value type for mappings
    E: Error type for Result pattern

Type Aliases:
    JsonValue, ConfigValue: Configuration serialization types
    PathLike: Path-like object types
    EnvMapping: Environment variable mappings
"""

from collections.abc import Callable
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    TypeAlias,
    TypeVar,
    Union,
)

from word_forge.paths import DATA_ROOT, LOGS_ROOT, PROJECT_ROOT

if TYPE_CHECKING:
    from word_forge.configs.types.protocols import ConfigComponent

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

# Runtime paths are imported above with the other module dependencies.

# ==========================================
# Basic Type Definitions
# ==========================================

# JSON-related type definitions for configuration serialization
JsonPrimitive = Union[str, int, float, bool, None]
JsonDict = dict[str, "JsonValue"]  # Forward reference for recursion
JsonList = list["JsonValue"]  # Forward reference for recursion
JsonValue: TypeAlias = JsonDict | JsonList | JsonPrimitive

# Configuration-specific type aliases
ConfigValue: TypeAlias = JsonValue

# Logging
LoggingConfigDict: TypeAlias = dict[str, Any]
ValidationError: TypeAlias = str
FormatStr: TypeAlias = str
LogFilePathStr: TypeAlias = str | None

# Function type for validation handlers
ValidationFunction: TypeAlias = Callable[[LoggingConfigDict, list[ValidationError]], None]
EnvVarType: TypeAlias = type[Any]

# Type alias for serialized configuration data
SerializedConfig: TypeAlias = JsonDict

# Type alias for path-like objects
PathLike: TypeAlias = str | Path

# Type alias for environment variable mapping in ConfigComponent
EnvMapping: TypeAlias = dict[str, tuple[str, EnvVarType]]

# Type alias for the name of a configuration component
ComponentName: TypeAlias = str

# Type alias for a registry mapping component names to instances
ComponentRegistry: TypeAlias = dict[ComponentName, "ConfigComponent"]

# Type alias for a dictionary representing a configuration section
ConfigDict: TypeAlias = dict[str, ConfigValue]

__all__ = [
    # Type variables
    "T",
    "T_contra",
    "R",
    "C",
    "K",
    "V",
    "E",
    # Paths
    "PROJECT_ROOT",
    "DATA_ROOT",
    "LOGS_ROOT",
    # JSON types
    "JsonPrimitive",
    "JsonDict",
    "JsonList",
    "JsonValue",
    # Config types
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
]
