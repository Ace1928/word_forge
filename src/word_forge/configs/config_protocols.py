"""
Unified configuration system for Word Forge.

This module centralizes all configuration settings used throughout
the Word Forge system, ensuring consistency across components.

The configuration architecture follows a modular approach with specialized
dataclasses for each subsystem, unified through a central Config class
that manages environment variable overrides and directory creation.
"""

from pathlib import Path
from typing import ClassVar, Dict, Final, Protocol, Tuple, TypeVar

from word_forge.configs.config_types import EnvVarType

# Define project paths with explicit typing for better IDE support
PROJECT_ROOT: Final[Path] = Path("/home/lloyd/eidosian_forge/word_forge")
DATA_ROOT: Final[Path] = PROJECT_ROOT / "data"
LOGS_ROOT: Final[Path] = PROJECT_ROOT / "logs"

C = TypeVar("C", bound="ConfigComponent")  # Bound to ConfigComponent protocol


# ConfigComponent protocol defines interface for configuration components
class ConfigComponent(Protocol):
    """Protocol defining interface for all configuration components."""

    # Each component must have ENV_VARS class variable for env overrides
    ENV_VARS: ClassVar[Dict[str, Tuple[str, EnvVarType]]]


class JSONSerializable(Protocol):
    """Protocol for objects that can be serialized to JSON."""

    def __str__(self) -> str:
        """Convert object to string representation for serialization."""
        ...
