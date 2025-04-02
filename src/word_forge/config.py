"""
Unified configuration system for Word Forge.

This module centralizes all configuration settings used throughout
the Word Forge system, ensuring consistency across components.

The configuration architecture follows a modular approach with specialized
dataclasses for each subsystem, unified through a central Config class
that manages environment variable overrides and directory creation.

Architecture:
    ┌───────────────┐
    │     Config    │ ← Central configuration manager
    └───────┬───────┘
            │ orchestrates
    ┌───────┴───────┐
    │  Components   │ ← Individual subsystem configs
    └───────────────┘
    ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
    │ DB  │Vec  │Parse│Emo  │Graph│Queue│Conv │Log  │
    └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

Design Principles:
    1. Single responsibility per component
    2. Type safety throughout the system
    3. Environment-based configuration overrides
    4. Self-documenting interfaces
    5. Automatic resource management
"""

import json
import os
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Set, Tuple, Type

from word_forge.configs.config_essentials import serialize_config, serialize_dataclass
from word_forge.configs.config_exceptions import (
    ConfigError,
    DatabaseConfigError,
    DatabaseConnectionError,
    EnvVarError,
    GraphConfigError,
    LoggingConfigError,
    PathError,
    VectorConfigError,
    VectorIndexError,
)
from word_forge.configs.config_protocols import C, ConfigComponent

# Import all essential configuration types
from word_forge.configs.config_types import (
    DATA_ROOT,
    LOGS_ROOT,
    PROJECT_ROOT,
    ComponentName,
    ComponentRegistry,
    ConfigComponentInfo,
    ConfigDict,
    ConfigValue,
    ConnectionPoolMode,
    ConversationMetadataSchema,
    ConversationStatusMap,
    ConversationStatusValue,
    EmotionRange,
    EnvMapping,
    EnvVarType,
    GraphEdgeWeightStrategy,
    GraphExportFormat,
    GraphNodeSizeStrategy,
    InstructionTemplate,
    LockType,
    LogLevel,
    PathLike,
    QueryType,
    QueueMetricsFormat,
    SQLitePragmas,
    SQLQueryType,
    SQLTemplates,
    TemplateDict,
    TransactionIsolationLevel,
    VectorDistanceMetric,
    VectorOptimizationLevel,
    VectorSearchStrategy,
)

# Import all configuration components
from word_forge.configs.conversation_config import ConversationConfig
from word_forge.configs.database_config import DatabaseConfig
from word_forge.configs.emotion_config import EmotionConfig
from word_forge.configs.graph_config import GraphConfig
from word_forge.configs.logging_config import LoggingConfig
from word_forge.configs.parser_config import ParserConfig
from word_forge.configs.queue_config import QueueConfig
from word_forge.configs.vectorizer_config import VectorizerConfig

# ==========================================
# Configuration Manager
# ==========================================


class Config:
    """
    Unified configuration for all Word Forge components.

    This class centralizes configuration for database, vectorizer, parser,
    emotion analysis, graph management, queue processing, conversation management,
    and logging systems. It provides environment variable overrides and ensures
    required directories exist.

    Attributes:
        database: Database connection and query configuration
        vectorizer: Vector embedding and indexing configuration
        parser: Text parsing and processing configuration
        emotion: Emotion analysis model configuration
        graph: Knowledge graph visualization configuration
        queue: Task queue processing configuration
        conversation: Conversation management configuration
        logging: Logging levels and output configuration
        _component_registry: Internal registry of component metadata for reflection

    Usage:
        from word_forge.config import config

        # Access settings directly
        db_path = config.database.db_path

        # Get Path objects
        db_path_obj = config.database.get_db_path

        # Get typed components
        db_config = config.get_typed_component("database", DatabaseConfig)
    """

    # Registry of configuration components with metadata
    # This enables reflection, dependency tracking, and runtime validation
    _component_registry: Final[ComponentRegistry] = {
        "database": ConfigComponentInfo(name="database", class_type=DatabaseConfig),
        "vectorizer": ConfigComponentInfo(
            name="vectorizer", class_type=VectorizerConfig
        ),
        "parser": ConfigComponentInfo(name="parser", class_type=ParserConfig),
        "emotion": ConfigComponentInfo(name="emotion", class_type=EmotionConfig),
        "graph": ConfigComponentInfo(name="graph", class_type=GraphConfig),
        "queue": ConfigComponentInfo(name="queue", class_type=QueueConfig),
        "conversation": ConfigComponentInfo(
            name="conversation", class_type=ConversationConfig
        ),
        "logging": ConfigComponentInfo(name="logging", class_type=LoggingConfig),
    }

    def __init__(self) -> None:
        """
        Initialize configuration with defaults and environment overrides.

        Creates configuration components, applies environment variable overrides,
        and ensures all required directories exist.
        """
        # Initialize all configuration components
        self.database: DatabaseConfig = DatabaseConfig()
        self.vectorizer: VectorizerConfig = VectorizerConfig()
        self.parser: ParserConfig = ParserConfig()
        self.emotion: EmotionConfig = EmotionConfig()
        self.graph: GraphConfig = GraphConfig()
        self.queue: QueueConfig = QueueConfig()
        self.conversation: ConversationConfig = ConversationConfig()
        self.logging: LoggingConfig = LoggingConfig()

        # Apply environment variable overrides
        self._load_from_env()

        # Ensure data directories exist
        self._ensure_directories()

    def _load_from_env(self) -> None:
        """
        Load configuration from environment variables.

        Processes environment variables based on ENV_VARS mapping
        defined in each configuration class. Each variable is converted
        to the appropriate type and assigned to the corresponding attribute.

        Raises:
            EnvVarError: If an environment variable has an invalid format,
                cannot be converted to the target type, or the attribute doesn't exist
        """
        for _, config_obj in self._get_config_objects():
            env_vars = getattr(config_obj.__class__, "ENV_VARS", None)
            if not env_vars:
                continue

            for env_var, (attr_name, value_type) in env_vars.items():
                self._set_from_env(env_var, config_obj, attr_name, value_type)

    def _get_config_objects(self) -> List[Tuple[ComponentName, ConfigComponent]]:
        """
        Get all configuration objects with their names.

        Returns:
            List of (name, object) tuples for all registered configuration components
        """
        component_items: List[Tuple[ComponentName, ConfigComponent]] = []
        for name in self._component_registry.keys():
            component = getattr(self, name)
            if component:
                component_items.append((name, component))
        return component_items

    def _set_from_env(
        self,
        env_var: str,
        config_obj: object,
        attr_name: str,
        value_type: EnvVarType,
    ) -> None:
        """
        Set configuration attribute from environment variable if it exists.

        Args:
            env_var: Environment variable name
            config_obj: Configuration object to modify
            attr_name: Attribute name to set
            value_type: Type to convert value to (str, bool, Enum, etc.)

        Raises:
            EnvVarError: If the environment variable has an invalid format,
                can't be converted to the target type, or attribute doesn't exist
        """
        if env_var not in os.environ:
            return

        value = os.environ[env_var]

        try:
            if value_type is bool:
                # Convert string to boolean with explicit true values
                typed_value: Any = value.lower() in ("true", "yes", "1", "y")
            elif value_type and issubclass(value_type, Enum):
                # Convert string to Enum value
                typed_value = value_type(value)
            else:
                # General type conversion
                typed_value = value_type(value)

            # Ensure attribute exists before setting
            if not hasattr(config_obj, attr_name):
                raise EnvVarError(
                    f"Configuration attribute '{attr_name}' not found in {config_obj.__class__.__name__}"
                )

            setattr(config_obj, attr_name, typed_value)
        except (ValueError, TypeError) as e:
            raise EnvVarError(f"Invalid value '{value}' for {env_var}: {str(e)}") from e
        except AttributeError as e:
            raise EnvVarError(
                f"Failed to set '{attr_name}' from {env_var}: {str(e)}"
            ) from e

    def _ensure_directories(self) -> None:
        """
        Ensure all required directories exist.

        Creates paths for data storage, logs, and any other required
        directories defined in configuration. This prevents errors when
        files are later written to these locations.

        Raises:
            PathError: If directory creation fails due to permissions or disk issues
        """
        try:
            # Ensure base directories exist
            DATA_ROOT.mkdir(parents=True, exist_ok=True)
            LOGS_ROOT.mkdir(parents=True, exist_ok=True)

            # Ensure subdirectories exist for specific data paths
            self._ensure_directory_for(self.vectorizer.index_path)
            self._ensure_directory_for(self.graph.default_export_path)
            self._ensure_directory_for(self.graph.visualization_path)
            self._ensure_directory_for(self.database.db_path)

            # Ensure log directory exists
            if self.logging.file_path:
                self._ensure_directory_for(self.logging.file_path)
        except (OSError, PermissionError) as e:
            raise PathError(f"Failed to create directory: {str(e)}") from e

    @staticmethod
    def _ensure_directory_for(file_path: PathLike) -> None:
        """
        Ensure parent directory exists for a given file path.

        Args:
            file_path: Path to a file whose parent directory should exist
        """
        if not file_path:
            return

        path_str = str(file_path)
        parent_dir = os.path.dirname(path_str)
        if parent_dir:  # Only create if there's a directory to create
            os.makedirs(parent_dir, exist_ok=True)

    def get_full_path(self, path: str) -> Path:
        """
        Convert relative path to absolute using project data directory.

        Args:
            path: Relative path to convert

        Returns:
            Absolute path based on the configured data directory
        """
        return Path(self.parser.data_dir) / path

    def get_component(self, name: str) -> Optional[ConfigComponent]:
        """
        Get a configuration component by name.

        Args:
            name: Name of the component to retrieve

        Returns:
            The component if found, None otherwise
        """
        return getattr(self, name, None) if name in self._component_registry else None

    def get_typed_component(self, name: str, component_type: Type[C]) -> Optional[C]:
        """
        Get a configuration component with type checking.

        Args:
            name: Name of the component to retrieve
            component_type: Expected type of the component

        Returns:
            The component if found and of the correct type, None otherwise

        Example:
            db_config = config.get_typed_component("database", DatabaseConfig)
            if db_config:
                connection = create_connection(db_config.get_db_path)
        """
        component = self.get_component(name)
        if component is not None and isinstance(component, component_type):
            return component
        return None

    def get_available_components(self) -> Set[ComponentName]:
        """
        Get list of all available configuration component names.

        Returns:
            Set of component names that can be accessed
        """
        return set(self._component_registry.keys())

    def validate_all(self) -> Dict[ComponentName, List[str]]:
        """
        Validate all configuration components that support validation.

        Calls validate() method on each component that provides it.

        Returns:
            Dictionary mapping component names to validation errors, empty if all valid

        Example:
            validation_results = config.validate_all()
            if any(validation_results.values()):
                print("Configuration errors detected:")
                for component, errors in validation_results.items():
                    if errors:
                        print(f"  {component}: {', '.join(errors)}")
        """
        results: Dict[ComponentName, List[str]] = {}

        for name, component in self._get_config_objects():
            # Only validate components with validate method
            validate_method = getattr(component, "validate", None)
            if validate_method and callable(validate_method):
                try:
                    validate_method()
                    results[name] = []
                except ConfigError as e:
                    results[name] = [str(e)]
                except Exception as e:
                    results[name] = [f"Unexpected error during validation: {str(e)}"]
            else:
                results[name] = []

        return results

    def to_dict(self) -> ConfigDict:
        """
        Convert the entire configuration to a dictionary.

        Returns:
            Dictionary representation of all configuration components with
            serialized values that can be converted to JSON
        """
        return {
            "database": self.database,
            "vectorizer": serialize_dataclass(self.vectorizer),
            "parser": asdict(self.parser),
            "emotion": serialize_dataclass(self.emotion),
            "graph": serialize_dataclass(self.graph),
            "queue": asdict(self.queue),
            "conversation": asdict(self.conversation),
            "logging": asdict(self.logging),
        }

    def export_json(self, pretty: bool = True) -> str:
        """
        Export configuration as JSON string.

        Args:
            pretty: Whether to format the JSON with indentation for readability

        Returns:
            JSON string representation of the configuration
        """
        config_dict = serialize_config(self)
        indent = 2 if pretty else None
        return json.dumps(config_dict, indent=indent)

    def export_to_file(self, path: PathLike) -> None:
        """
        Export configuration to a JSON file.

        Args:
            path: Path where to save the configuration

        Raises:
            IOError: If file writing fails due to permissions or disk space
        """
        with open(path, "w") as f:
            f.write(self.export_json(pretty=True))


# ==========================================
# Global Configuration Instance
# ==========================================

# Global configuration instance for application-wide access
config: Final[Config] = Config()


# ==========================================
# CLI Interface
# ==========================================


def main() -> None:
    """
    Display current configuration settings.

    Command-line interface function that provides options to validate
    configuration, export to file, or display configuration components.

    Usage:
        python -m word_forge.config --validate
        python -m word_forge.config --export config.json
        python -m word_forge.config --component database
    """
    import argparse

    parser = argparse.ArgumentParser(description="Word Forge Configuration Utility")
    parser.add_argument(
        "--component",
        "-c",
        help="Display specific component configuration",
        choices=config.get_available_components(),
    )
    parser.add_argument(
        "--validate",
        "-v",
        action="store_true",
        help="Validate configuration",
    )
    parser.add_argument(
        "--export",
        "-e",
        help="Export configuration to JSON file",
    )
    args = parser.parse_args()

    if args.validate:
        validation_results = config.validate_all()
        invalid_components = {
            comp: errors for comp, errors in validation_results.items() if errors
        }

        if invalid_components:
            print("❌ Configuration validation failed:")
            for component, errors in invalid_components.items():
                print(f"  • {component}: {'; '.join(errors)}")
            return
        else:
            print("✅ Configuration validation passed for all components")

    if args.export:
        try:
            config.export_to_file(args.export)
            print(f"✅ Configuration exported to {args.export}")
        except Exception as e:
            print(f"❌ Export failed: {str(e)}")
        return

    if args.component:
        component = config.get_component(args.component)
        if component:
            print(f"{args.component.title()} Configuration")
            print("=" * (len(args.component) + 14))
            component_dict = serialize_config(component)
            print(json.dumps(component_dict, indent=2))
        else:
            print(f"Component {args.component} not found")
        return

    # Default: print all configuration
    print("Word Forge Configuration")
    print("=======================")

    # Print JSON representation of config
    print(config.export_json(pretty=True))

    # Print examples of accessing specific settings
    print("\nAccessing specific settings:")
    print(f"Database path: {config.database.db_path}")
    print(f"Vector model: {config.vectorizer.model_name}")
    print(f"Vector dimension: {config.vectorizer.dimension or 'Default from model'}")
    print(f"Data directory: {config.parser.data_dir}")

    # Print sample path resolutions
    print("\nPath resolution examples:")
    print(f"Full database path: {config.database.get_db_path}")
    print(
        f"OpenThesaurus path: {config.parser.get_full_resource_path('openthesaurus')}"
    )
    print(f"Vector index path: {config.vectorizer.get_index_path}")


# ==========================================
# Module Exports - Configuration Instance
# ==========================================

__all__ = [
    # Core configuration class and instance
    "Config",
    "config",
    "main",
]

# ==========================================
# Module Exports - Configuration Components
# ==========================================

__all__ += [
    # Component configurations
    "DatabaseConfig",
    "VectorizerConfig",
    "ParserConfig",
    "EmotionConfig",
    "GraphConfig",
    "QueueConfig",
    "ConversationConfig",
    "LoggingConfig",
]

# ==========================================
# Module Exports - Path Constants
# ==========================================

__all__ += [
    "PROJECT_ROOT",
    "DATA_ROOT",
    "LOGS_ROOT",
]

# ==========================================
# Module Exports - Error Types
# ==========================================

__all__ += [
    "ConfigError",
    "EnvVarError",
    "PathError",
    "VectorConfigError",
    "VectorIndexError",
    "GraphConfigError",
    "LoggingConfigError",
    "DatabaseConfigError",
    "DatabaseConnectionError",
]

# ==========================================
# Module Exports - Type Definitions
# ==========================================

__all__ += [
    # Component types
    "ConfigComponent",
    "ConfigComponentInfo",
    "ComponentName",
    "ComponentRegistry",
    "ConfigDict",
    "ConfigValue",
    "PathLike",
    # Template types
    "InstructionTemplate",
    "TemplateDict",
    "QueryType",
    "SQLQueryType",
    # Storage types
    "SQLitePragmas",
    "SQLTemplates",
    # Queue types
    "LockType",
    "QueueMetricsFormat",
    # Conversation types
    "ConversationStatusValue",
    "ConversationStatusMap",
    "ConversationMetadataSchema",
    # Vector types
    "VectorSearchStrategy",
    "VectorDistanceMetric",
    "VectorOptimizationLevel",
    # Graph types
    "GraphExportFormat",
    "GraphNodeSizeStrategy",
    "GraphEdgeWeightStrategy",
    # Logging types
    "LogLevel",
    # Database types
    "TransactionIsolationLevel",
    "ConnectionPoolMode",
    # Other types
    "EmotionRange",
    "EnvMapping",
    "EnvVarType",
]

# ==========================================
# Module Exports - Utility Functions
# ==========================================

__all__ += [
    "serialize_config",
    "serialize_dataclass",
]


if __name__ == "__main__":
    main()
