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
    6. Dynamic adaptability to execution environment
    7. Hot reloading capability for runtime updates
    8. Self-healing with intelligent defaults
    9. Performance optimization with caching strategies
"""

import json
import os
import tempfile
import threading
import time
from dataclasses import fields, replace
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Final,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    cast,
    get_type_hints,
)

# Import all essential configuration types
from word_forge.configs.config_essentials import (
    DATA_ROOT,
    LOGS_ROOT,
    PROJECT_ROOT,
    C,
    ComponentName,
    ComponentRegistry,
    ConfigComponent,
    ConfigComponentInfo,
    ConfigDict,
    ConfigError,
    ConfigValue,
    ConnectionPoolMode,
    ConversationMetadataSchema,
    ConversationStatusMap,
    ConversationStatusValue,
    DatabaseConfigError,
    DatabaseConnectionError,
    EmotionRange,
    EnvMapping,
    EnvVarError,
    EnvVarType,
    GraphConfigError,
    GraphEdgeWeightStrategy,
    GraphExportFormat,
    GraphNodeSizeStrategy,
    InstructionTemplate,
    LockType,
    LoggingConfigError,
    LogLevel,
    PathError,
    PathLike,
    QueryType,
    QueueMetricsFormat,
    SQLitePragmas,
    SQLQueryType,
    SQLTemplates,
    TemplateDict,
    TransactionIsolationLevel,
    VectorConfigError,
    VectorDistanceMetric,
    VectorIndexError,
    VectorOptimizationLevel,
    VectorSearchStrategy,
    serialize_config,
    serialize_dataclass,
)
from word_forge.configs.config_io import (
    ConfigurationFileError,
    coerce_configuration_value,
    load_configuration_document,
    merge_configuration_value,
    parse_environment_value,
)

# ... standard imports from the original file ...
from word_forge.configs.logging_config import LoggingConfig
from word_forge.conversation.conversation_config import ConversationConfig
from word_forge.database.database_config import DatabaseConfig
from word_forge.emotion.emotion_config import EmotionConfig
from word_forge.graph.graph_config import GraphConfig
from word_forge.parser.parser_config import ParserConfig
from word_forge.queue.queue_config import QueueConfig
from word_forge.vectorizer.vectorizer_config import VectorizerConfig

# New type definitions for enhanced features
ConfigObserver = Callable[["Config", ComponentName, str], None]
ValidationStrategy = Callable[[ConfigComponent], List[str]]
T = TypeVar("T")
CacheKey = Tuple[ComponentName, str, type]
ConfigVersion = Tuple[int, int, int]  # major, minor, patch


class ConfigChangeEvent:
    """
    Represents a change in configuration for observers.

    This class encapsulates details about a configuration change,
    including which component changed, which attribute was modified,
    and the old and new values.

    Attributes:
        component_name: Name of the component that changed
        attribute_name: Name of the attribute that changed
        old_value: Previous value before the change
        new_value: Current value after the change
        timestamp: When the change occurred
    """

    def __init__(
        self,
        component_name: ComponentName,
        attribute_name: str,
        old_value: Any,
        new_value: Any,
    ) -> None:
        """Initialize with change details."""
        self.component_name = component_name
        self.attribute_name = attribute_name
        self.old_value = old_value
        self.new_value = new_value
        self.timestamp = time.time()

    def __str__(self) -> str:
        """Human-readable representation of the change."""
        return (
            f"Config change: {self.component_name}.{self.attribute_name} "
            f"changed from {self.old_value!r} to {self.new_value!r}"
        )


class ConfigSourceType(Enum):
    """Types of configuration sources with priority order."""

    DEFAULT = 0  # Hardcoded defaults
    FILE = 1  # From configuration file
    ENVIRONMENT = 2  # From environment variables
    RUNTIME = 3  # Set during runtime


class ConfigSource:
    """
    Tracks the source of a configuration value.

    This helps in understanding where a particular setting came from,
    which is useful for debugging and when resolving conflicts.

    Attributes:
        type: The type of configuration source
        location: Where the value came from (e.g., file path, env var name)
        timestamp: When the value was set
    """

    def __init__(
        self,
        source_type: ConfigSourceType,
        location: str = "",
        timestamp: Optional[float] = None,
    ) -> None:
        """Initialize with source details."""
        self.type = source_type
        self.location = location
        self.timestamp = timestamp or time.time()

    def __str__(self) -> str:
        """Human-readable representation of the source."""
        source_desc = f"{self.type.name}"
        if self.location:
            source_desc += f" ({self.location})"
        return source_desc


class RuntimeAdaptiveMode(Enum):
    """Modes for runtime adaptive configuration behavior."""

    DISABLED = "disabled"  # No automatic adaptation
    PASSIVE = "passive"  # Collect metrics but don't auto-adjust
    ACTIVE = "active"  # Automatically adjust based on metrics
    LEARNING = "learning"  # Use reinforcement learning to optimize


class Config:
    """
    Unified configuration for all Word Forge components.

    This class centralizes configuration for database, vectorizer, parser,
    emotion analysis, graph management, queue processing, conversation management,
    and logging systems. It provides environment variable overrides and ensures
    required directories exist.

    Enhanced with dynamic hot reloading, adaptive configuration, and self-healing
    capabilities while maintaining backward compatibility.

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
        version: Configuration schema version (major, minor, patch)

    Usage:
        from word_forge.config import config

        # Access settings directly
        db_path = config.database.db_path

        # Get Path objects
        db_path_obj = config.database.get_db_path

        # Get typed components
        db_config = config.get_typed_component("database", DatabaseConfig)

        # Register for configuration changes
        config.register_observer(my_callback_function)

        # Get value with source information
        value, source = config.get_value_with_source("database", "db_path")
        print(f"Value {value} came from {source}")
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

    # Configuration schema version
    version: ClassVar[ConfigVersion] = (1, 1, 0)  # major.minor.patch

    # Component interdependency graph (for validation)
    _component_dependencies: ClassVar[Dict[ComponentName, Set[ComponentName]]] = {
        "database": set(),  # No dependencies
        "vectorizer": {"database"},  # May need DB for persistent storage
        "parser": {"database"},  # Needs DB for lexical data
        "emotion": {"database", "vectorizer"},  # May use vectors and DB
        "graph": {"database"},  # Needs DB for relationship data
        "queue": {"database"},  # Uses DB for task storage
        "conversation": {"database", "emotion"},  # Uses DB and emotion analysis
        "logging": set(),  # No dependencies
    }

    def __init__(self) -> None:
        """
        Initialize configuration with defaults and environment overrides.

        Creates configuration components, applies environment variable overrides,
        ensures all required directories exist, and sets up the observers and
        caching infrastructure.
        """
        # Initialize main components
        self.database: DatabaseConfig = DatabaseConfig()
        self.vectorizer: VectorizerConfig = VectorizerConfig()
        self.parser: ParserConfig = ParserConfig()
        self.emotion: EmotionConfig = EmotionConfig()
        self.graph: GraphConfig = GraphConfig()
        self.queue: QueueConfig = QueueConfig()
        self.conversation: ConversationConfig = ConversationConfig()
        self.logging: LoggingConfig = LoggingConfig()

        # Enhanced features
        self._observers: List[ConfigObserver] = []
        self._value_sources: Dict[Tuple[ComponentName, str], ConfigSource] = {}
        self._config_lock = threading.RLock()
        self._value_cache: Dict[CacheKey, Any] = {}
        self._last_refresh_time = time.time()
        self._hot_reload_enabled = False
        self._hot_reload_interval = 30.0  # seconds
        self._hot_reload_thread: Optional[threading.Thread] = None
        self._runtime_adaptive_mode = RuntimeAdaptiveMode.PASSIVE
        self._error_counts: Dict[ComponentName, int] = {
            name: 0 for name in self._component_registry
        }

        # Track which components have been accessed
        self._accessed_components: Set[ComponentName] = set()

        # Record defaults before applying higher-priority sources.
        self._initialize_value_sources()

        # Apply environment variable overrides
        self._load_from_env()

        # Ensure data directories exist
        self._ensure_directories()

    def _initialize_value_sources(self) -> None:
        """Initialize source tracking for all configuration values."""
        for component_name, component in self._get_config_objects():
            for field_info in fields(cast(Any, component)):
                attr_name = field_info.name
                if attr_name.startswith("_"):
                    continue
                # Record default values
                self._value_sources[(component_name, attr_name)] = ConfigSource(
                    ConfigSourceType.DEFAULT,
                    f"{component.__class__.__name__}.{attr_name}",
                )

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
        for component_name, config_obj in self._get_config_objects():
            env_vars = getattr(config_obj.__class__, "ENV_VARS", None)
            if not env_vars:
                continue

            for env_var, (attr_name, value_type) in env_vars.items():
                if env_var in os.environ:
                    # Each frozen-component update replaces the dataclass. Fetch
                    # the latest instance so multiple variables for one component
                    # compose instead of the last one discarding earlier values.
                    current_config_obj = getattr(self, component_name)
                    self._set_from_env(
                        env_var,
                        component_name,
                        current_config_obj,
                        attr_name,
                        value_type,
                    )

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
        component_name: ComponentName,
        config_obj: object,
        attr_name: str,
        value_type: EnvVarType,
    ) -> None:
        """
        Set configuration attribute from environment variable if it exists.

        Args:
            env_var: Environment variable name
            component_name: Name of the component being configured
            config_obj: Configuration object to modify
            attr_name: Attribute name to set
            value_type: Type to convert value to (str, bool, Enum, etc.)

        Raises:
            EnvVarError: If the environment variable has an invalid format,
                can't be converted to the target type, or attribute doesn't exist
        """
        current_source = self._value_sources.get((component_name, attr_name))
        if (
            current_source
            and current_source.type.value > ConfigSourceType.ENVIRONMENT.value
        ):
            return

        raw_value = os.environ[env_var]
        try:
            parsed_value = parse_environment_value(raw_value, value_type, env_var)
            updated, typed_value = self._updated_component(
                component_name, config_obj, attr_name, parsed_value
            )
        except (ConfigurationFileError, TypeError, ValueError) as exc:
            raise EnvVarError(
                f"Invalid value {raw_value!r} for {env_var}: {exc}"
            ) from exc

        old_value = getattr(config_obj, attr_name)
        with self._config_lock:
            setattr(self, component_name, updated)
            self._value_sources[(component_name, attr_name)] = ConfigSource(
                ConfigSourceType.ENVIRONMENT,
                f"Environment variable: {env_var}",
            )
            self._invalidate_cache(component_name, attr_name)
        self._notify_observers(component_name, attr_name, old_value, typed_value)

    @staticmethod
    def _updated_component(
        component_name: ComponentName,
        component: object,
        attr_name: str,
        supplied_value: object,
    ) -> Tuple[object, object]:
        """Create a component copy containing one validated typed field update.

        Args:
            component_name: Registered component name.
            component: Current or staged dataclass instance.
            attr_name: Field to update.
            supplied_value: Parsed but not necessarily typed value.

        Returns:
            Updated component and its coerced field value.

        Raises:
            ConfigurationFileError: If the field is unknown or has an invalid type.
        """
        configurable_fields = {
            field_info.name
            for field_info in fields(cast(Any, component))
            if field_info.init and not field_info.name.startswith("_")
        }
        if attr_name not in configurable_fields:
            raise ConfigurationFileError(
                f"Unknown configuration field '{component_name}.{attr_name}'"
            )

        current_value = getattr(component, attr_name)
        merged_value = merge_configuration_value(current_value, supplied_value)
        expected_type = get_type_hints(type(component)).get(attr_name, object)
        typed_value = coerce_configuration_value(
            merged_value, expected_type, f"{component_name}.{attr_name}"
        )
        try:
            updated = replace(cast(Any, component), **{attr_name: typed_value})
        except (TypeError, ValueError) as exc:
            raise ConfigurationFileError(
                f"Invalid configuration field '{component_name}.{attr_name}': {exc}"
            ) from exc
        return updated, typed_value

    @staticmethod
    def _component_validation_error(component: object) -> Optional[str]:
        """Return a component validation error without leaking validator styles."""
        validate_method = getattr(component, "validate", None)
        if not callable(validate_method):
            return None
        try:
            result = validate_method()
        except ConfigError as exc:
            return str(exc)
        except Exception as exc:
            return f"Unexpected error during validation: {exc}"

        if getattr(result, "is_failure", False):
            error = getattr(result, "error", None)
            return str(getattr(error, "message", error or "validation failed"))
        return None

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
        # Track access to parser component
        self._accessed_components.add("parser")
        return Path(self.parser.data_dir) / path

    def get_component(self, name: str) -> Optional[ConfigComponent]:
        """
        Get a configuration component by name.

        Args:
            name: Name of the component to retrieve

        Returns:
            The component if found, None otherwise
        """
        if name in self._component_registry:
            # Track component access
            self._accessed_components.add(name)
            return getattr(self, name, None)
        return None

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
            # Track access with type information
            self._accessed_components.add(name)
            return component
        return None

    def get_available_components(self) -> Set[ComponentName]:
        """
        Get list of all available configuration component names.

        Returns:
            Set of component names that can be accessed
        """
        return set(self._component_registry.keys())

    def get_accessed_components(self) -> Set[ComponentName]:
        """
        Get set of components that have been accessed during runtime.

        This is useful for diagnostics and determining which components
        are actually used in a particular execution path.

        Returns:
            Set of component names that have been accessed
        """
        return self._accessed_components.copy()

    def validate_all(self) -> Dict[ComponentName, List[str]]:
        """
        Validate all configuration components that support validation.

        Calls validate() method on each component that provides it.
        Validates components in dependency order to ensure proper validation.

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

        # Sort components by dependency order
        sorted_components = self._sort_components_by_dependencies()

        for component_name in sorted_components:
            component = getattr(self, component_name)

            # Skip if component's dependencies have validation errors
            should_skip = False
            for dep_name in self._component_dependencies.get(component_name, set()):
                if dep_name in results and results[dep_name]:
                    results[component_name] = [
                        f"Validation skipped due to errors in dependency '{dep_name}'"
                    ]
                    should_skip = True
                    break

            if should_skip:
                continue

            validation_error = self._component_validation_error(component)
            results[component_name] = (
                [validation_error] if validation_error is not None else []
            )
            if validation_error is not None:
                self._error_counts[component_name] += 1

        return results

    def _sort_components_by_dependencies(self) -> List[ComponentName]:
        """
        Sort components in dependency order.

        Returns a list of component names sorted so that dependencies come before
        components that depend on them.

        Returns:
            List of component names in dependency order
        """
        # Implementation uses topological sort
        result: List[ComponentName] = []
        visited: Set[ComponentName] = set()
        temp_visit: Set[ComponentName] = set()

        def visit(name: ComponentName) -> None:
            if name in temp_visit:
                # Circular dependency detected
                return
            if name in visited:
                return

            temp_visit.add(name)

            # Visit dependencies first
            for dep in self._component_dependencies.get(name, set()):
                visit(dep)

            temp_visit.remove(name)
            visited.add(name)
            result.append(name)

        # Visit all components
        for name in self._component_registry:
            if name not in visited:
                visit(name)

        return result

    def to_dict(self) -> ConfigDict:
        """
        Convert the entire configuration to a dictionary.

        Returns:
            Dictionary representation of all configuration components with
            serialized values that can be converted to JSON
        """
        return {
            "database": serialize_config(self.database),
            "vectorizer": serialize_config(self.vectorizer),
            "parser": serialize_config(self.parser),
            "emotion": serialize_config(self.emotion),
            "graph": serialize_config(self.graph),
            "queue": serialize_config(self.queue),
            "conversation": serialize_config(self.conversation),
            "logging": serialize_config(self.logging),
            "meta": {
                "version": ".".join(str(v) for v in self.version),
                "generated_at": time.time(),
                "accessed_components": cast(
                    ConfigValue, sorted(self._accessed_components)
                ),
            },
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
        return json.dumps(
            config_dict, ensure_ascii=False, indent=indent, sort_keys=True
        )

    def export_to_file(self, path: PathLike) -> None:
        """
        Export configuration to a JSON file.

        Args:
            path: Path where to save the configuration

        Raises:
            IOError: If file writing fails due to permissions or disk space
        """
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(self.export_json(pretty=True))
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, target)
        except OSError:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _read_config_mapping(path: Path) -> Mapping[str, Any]:
        """Read a bounded JSON or YAML configuration document."""
        return load_configuration_document(path)

    def _has_higher_priority_source(
        self, component_name: ComponentName, attr_name: str
    ) -> bool:
        """Return whether an environment/runtime value must beat a file value."""
        source = self._value_sources.get((component_name, attr_name))
        return source is not None and source.type.value > ConfigSourceType.FILE.value

    def load_from_file(self, path: PathLike) -> None:
        """Load JSON or YAML settings atomically.

        Unknown components and fields are rejected to surface misspellings.
        Runtime and environment overrides retain their documented precedence.
        Dataclass instances are prepared before any live state is changed, so a
        malformed document cannot leave a partially updated configuration.

        Args:
            path: JSON, YAML, or YML configuration file.

        Raises:
            ConfigError: If the file is invalid or contains unsupported settings.
        """
        config_path = Path(path).expanduser()
        document = self._read_config_mapping(config_path)
        allowed_top_level = set(self._component_registry) | {"meta"}
        unknown_components = sorted(set(document) - allowed_top_level)
        if unknown_components:
            raise ConfigError(
                "Unknown configuration component(s): " + ", ".join(unknown_components)
            )

        replacements: Dict[ComponentName, object] = {}
        changes: List[Tuple[ComponentName, str, object, object]] = []
        for raw_name, raw_settings in document.items():
            if raw_name == "meta":
                continue
            component_name = raw_name
            if not isinstance(raw_settings, Mapping):
                raise ConfigError(
                    f"Configuration component '{component_name}' must be an object"
                )

            component = getattr(self, component_name)
            known_fields = {
                field_info.name
                for field_info in fields(component)
                if field_info.init and not field_info.name.startswith("_")
            }
            unknown_fields = sorted(set(raw_settings) - known_fields)
            if unknown_fields:
                raise ConfigError(
                    f"Unknown setting(s) for {component_name}: "
                    + ", ".join(unknown_fields)
                )

            candidate = component
            for attr_name, raw_value in raw_settings.items():
                if self._has_higher_priority_source(component_name, attr_name):
                    continue
                old_value = getattr(candidate, attr_name)
                candidate, new_value = self._updated_component(
                    component_name, candidate, attr_name, raw_value
                )
                changes.append(
                    (
                        component_name,
                        attr_name,
                        old_value,
                        new_value,
                    )
                )
            validation_error = self._component_validation_error(candidate)
            if validation_error is not None:
                raise ConfigurationFileError(
                    f"Invalid '{component_name}' configuration: {validation_error}"
                )
            replacements[component_name] = candidate

        original_components = {
            component_name: getattr(self, component_name)
            for component_name in replacements
        }
        original_sources = self._value_sources.copy()
        with self._config_lock:
            try:
                for component_name, component in replacements.items():
                    setattr(self, component_name, component)
                for component_name, attr_name, _old, _new in changes:
                    self._value_sources[(component_name, attr_name)] = ConfigSource(
                        ConfigSourceType.FILE, str(config_path.resolve())
                    )
                    self._invalidate_cache(component_name, attr_name)
                self._ensure_directories()
            except Exception:
                for component_name, component in original_components.items():
                    setattr(self, component_name, component)
                self._value_sources = original_sources
                self.clear_caches()
                raise

        for component_name, attr_name, old_value, new_value in changes:
            self._notify_observers(component_name, attr_name, old_value, new_value)

    def register_observer(self, observer: ConfigObserver) -> None:
        """
        Register a function to be called when configuration changes.

        Args:
            observer: Callback function that takes component name and attribute name

        Example:
            def my_callback(config, component_name, attr_name):
                print(f"Configuration changed: {component_name}.{attr_name}")

            config.register_observer(my_callback)
        """
        if observer not in self._observers:
            # Use weak references to avoid memory leaks
            self._observers.append(observer)

    def unregister_observer(self, observer: ConfigObserver) -> None:
        """
        Remove a previously registered observer.

        Args:
            observer: Observer function to remove
        """
        if observer in self._observers:
            self._observers.remove(observer)

    def _notify_observers(
        self,
        component_name: ComponentName,
        attr_name: str,
        old_value: Any,
        new_value: Any,
    ) -> None:
        """
        Notify all observers of a configuration change.

        Args:
            component_name: Name of the component that changed
            attr_name: Name of the attribute that changed
            old_value: Previous value
            new_value: New value
        """
        # Copy the list to avoid issues if observers modify the list
        observers = self._observers.copy()
        for observer in observers:
            try:
                observer(self, component_name, attr_name)
            except Exception as e:
                # Log but don't propagate observer errors
                import logging

                logging.getLogger(__name__).error(
                    f"Error in configuration observer: {e}"
                )

    def _invalidate_cache(self, component_name: ComponentName, attr_name: str) -> None:
        """
        Invalidate cache entries related to a specific attribute.

        Args:
            component_name: Name of the component with changed attribute
            attr_name: Name of the attribute that changed
        """
        with self._config_lock:
            # Find all cache keys that match the component and attribute
            keys_to_remove: List[CacheKey] = []
            for key in self._value_cache:
                key_component, key_attr, _ = key
                if key_component == component_name and key_attr == attr_name:
                    keys_to_remove.append(key)

            # Remove matching cache entries
            for key in keys_to_remove:
                self._value_cache.pop(key, None)

            # functools.lru_cache does not expose per-key invalidation. Clearing
            # this small cache guarantees runtime and hot-reload reads are fresh.
            self.get_cached_value.cache_clear()

    def get_value_with_source(
        self, component_name: ComponentName, attr_name: str
    ) -> Tuple[Any, ConfigSource]:
        """
        Get a configuration value with information about its source.

        Args:
            component_name: Name of the component
            attr_name: Name of the attribute

        Returns:
            Tuple of (value, source)

        Raises:
            AttributeError: If the attribute doesn't exist
        """
        # Get the component
        component = self.get_component(component_name)
        if component is None:
            raise AttributeError(f"Component '{component_name}' not found")

        # Get the attribute
        if not hasattr(component, attr_name):
            raise AttributeError(
                f"Attribute '{attr_name}' not found in component '{component_name}'"
            )

        value = getattr(component, attr_name)

        # Get the source (default if not recorded)
        source = self._value_sources.get(
            (component_name, attr_name), ConfigSource(ConfigSourceType.DEFAULT)
        )

        return value, source

    @lru_cache(maxsize=128)
    def get_cached_value(
        self, component_name: ComponentName, attr_name: str, value_type: Type[T]
    ) -> T:
        """
        Get a configuration value with caching for performance.

        This method is useful for frequently accessed values, as it
        uses LRU caching to avoid repeated attribute lookups.

        Args:
            component_name: Name of the component
            attr_name: Name of the attribute
            value_type: Expected type of the value (for type checking)

        Returns:
            The configuration value

        Raises:
            AttributeError: If the attribute doesn't exist
            TypeError: If the value is not of the expected type
        """
        # Get the component
        component = self.get_component(component_name)
        if component is None:
            raise AttributeError(f"Component '{component_name}' not found")

        # Get the attribute
        if not hasattr(component, attr_name):
            raise AttributeError(
                f"Attribute '{attr_name}' not found in component '{component_name}'"
            )

        value = getattr(component, attr_name)

        # Type check the value
        if not isinstance(value, value_type):
            raise TypeError(
                f"Value for {component_name}.{attr_name} is {type(value)}, "
                f"expected {value_type}"
            )

        return value  # Value is already known to be of type T

    def set_runtime_value(
        self, component_name: ComponentName, attr_name: str, value: Any
    ) -> None:
        """
        Set a configuration value at runtime.

        This allows for dynamic reconfiguration during program execution.
        Sets are tracked as RUNTIME source type for debugging.

        Args:
            component_name: Name of the component
            attr_name: Name of the attribute
            value: New value to set

        Raises:
            AttributeError: If the component or attribute doesn't exist
            TypeError: If the value is not compatible with the attribute
        """
        # Get the component
        component = self.get_component(component_name)
        if component is None:
            raise AttributeError(f"Component '{component_name}' not found")

        # Check attribute exists and get old value
        if not hasattr(component, attr_name):
            raise AttributeError(
                f"Attribute '{attr_name}' not found in component '{component_name}'"
            )

        old_value = getattr(component, attr_name)

        try:
            updated, normalized_value = self._updated_component(
                component_name, component, attr_name, value
            )
        except ConfigurationFileError as exc:
            raise TypeError(
                f"Cannot set {component_name}.{attr_name} to {value!r}: {exc}"
            ) from exc

        validation_error = self._component_validation_error(updated)
        if validation_error is not None:
            raise TypeError(
                f"Cannot set {component_name}.{attr_name} to {value!r}: "
                f"{validation_error}"
            )

        with self._config_lock:
            setattr(self, component_name, updated)

            # Record the source
            self._value_sources[(component_name, attr_name)] = ConfigSource(
                ConfigSourceType.RUNTIME,
                "Runtime override",
            )

            # Invalidate cache
            self._invalidate_cache(component_name, attr_name)

        # Notify observers
        self._notify_observers(component_name, attr_name, old_value, normalized_value)

    def enable_hot_reload(self, interval: float = 30.0) -> None:
        """
        Enable configuration hot reloading.

        When enabled, the configuration will periodically check for changes
        in environment variables and files, applying updates without restart.

        Args:
            interval: How often to check for changes (in seconds)
        """
        self._hot_reload_enabled = True
        self._hot_reload_interval = interval

        # Start the monitoring thread if not already running
        if self._hot_reload_thread is None or not self._hot_reload_thread.is_alive():
            self._hot_reload_thread = threading.Thread(
                target=self._hot_reload_monitor,
                daemon=True,
                name="ConfigHotReloadMonitor",
            )
            self._hot_reload_thread.start()

    def disable_hot_reload(self) -> None:
        """Disable configuration hot reloading."""
        self._hot_reload_enabled = False

    def _hot_reload_monitor(self) -> None:
        """Background thread that checks for configuration changes."""
        while True:
            # Sleep first to avoid immediate refresh after initialization
            time.sleep(self._hot_reload_interval)

            # Skip if disabled
            if not self._hot_reload_enabled:
                continue

            try:
                # Check for environment variable changes
                self._refresh_from_environment()

                # Note: File monitoring would be added here
            except Exception as e:
                # Log but continue monitoring
                import logging

                logging.getLogger(__name__).error(
                    f"Error during configuration hot reload: {e}"
                )

    def _refresh_from_environment(self) -> None:
        """Check for changes in environment variables and apply them."""
        # This is a simplified version - a real implementation would track
        # which env vars have changed since last refresh
        self._load_from_env()
        self._last_refresh_time = time.time()

    def set_adaptive_mode(self, mode: RuntimeAdaptiveMode) -> None:
        """
        Set the adaptive configuration mode.

        Different modes affect how the configuration responds to
        runtime metrics and system conditions.

        Args:
            mode: The adaptive mode to use
        """
        old_mode = self._runtime_adaptive_mode
        self._runtime_adaptive_mode = mode

        # Log the change
        import logging

        logging.getLogger(__name__).info(
            f"Configuration adaptive mode changed from {old_mode} to {mode}"
        )

    def report_performance_metric(
        self, component_name: ComponentName, metric_name: str, value: float
    ) -> None:
        """
        Report a performance metric that may trigger adaptive configuration.

        In ACTIVE or LEARNING modes, this may cause configuration parameters
        to be automatically adjusted based on performance feedback.

        Args:
            component_name: Name of the component reporting the metric
            metric_name: Name of the metric (e.g. "query_time_ms")
            value: Numeric value of the metric
        """
        if self._runtime_adaptive_mode == RuntimeAdaptiveMode.DISABLED:
            return

        # In a real implementation, this would use the metrics to adjust
        # configuration parameters based on performance data

        # For demonstration, we'll just log the metric
        if self._runtime_adaptive_mode != RuntimeAdaptiveMode.PASSIVE:
            import logging

            logging.getLogger(__name__).debug(
                f"Performance metric: {component_name}.{metric_name} = {value}"
            )

    def apply_profile(self, profile_name: str) -> None:
        """Atomically apply a predefined configuration profile.

        Profiles define sets of configuration values optimized for
        specific use cases or environments.

        Args:
            profile_name: Name of the profile to apply

        Raises:
            ValueError: If the profile doesn't exist
            TypeError: If a profile violates a component contract.
        """
        profiles: Dict[str, Dict[str, Dict[str, object]]] = {
            "development": {
                "database": {"db_path": ":memory:"},
                "logging": {"level": "DEBUG"},
                "vectorizer": {"storage_type": "memory"},
            },
            "production": {
                "database": {"db_path": str(DATA_ROOT / "production.sqlite")},
                "logging": {"level": "WARNING"},
                "vectorizer": {"storage_type": "disk"},
            },
            "testing": {
                "database": {"db_path": ":memory:"},
                "logging": {"level": "ERROR"},
                "parser": {"enable_model": False},
                "vectorizer": {"batch_size": 4, "storage_type": "memory"},
            },
            "high_performance": {
                "database": {
                    "pragmas": {
                        "cache_size": "-64000",
                        "journal_mode": "WAL",
                        "synchronous": "NORMAL",
                    }
                },
                "queue": {
                    "batch_size": 200,
                    "lru_cache_size": 1024,
                    "throttle_seconds": 0.0,
                },
                "vectorizer": {
                    "batch_size": 64,
                    "optimization_level": "speed",
                    "reserved_memory_mb": 1024,
                },
            },
            "low_memory": {
                "graph": {
                    "high_quality_rendering": False,
                    "limit_edge_count": 600,
                    "limit_node_count": 300,
                },
                "parser": {"enable_model": False},
                "queue": {
                    "batch_size": 10,
                    "lru_cache_size": 32,
                    "max_queue_size": 1000,
                },
                "vectorizer": {
                    "batch_size": 4,
                    "enable_compression": True,
                    "reserved_memory_mb": 128,
                },
            },
        }

        if profile_name not in profiles:
            raise ValueError(
                f"Unknown profile '{profile_name}'. "
                f"Available profiles: {', '.join(profiles.keys())}"
            )

        replacements: Dict[ComponentName, object] = {}
        changes: List[Tuple[ComponentName, str, object, object]] = []
        for component_name, settings in profiles[profile_name].items():
            component = getattr(self, component_name)
            candidate = component
            for attr_name, value in settings.items():
                old_value = getattr(candidate, attr_name, None)
                candidate, typed_value = self._updated_component(
                    component_name, candidate, attr_name, value
                )
                changes.append((component_name, attr_name, old_value, typed_value))
            validation_error = self._component_validation_error(candidate)
            if validation_error is not None:
                raise TypeError(
                    f"Invalid '{profile_name}' profile for {component_name}: "
                    f"{validation_error}"
                )
            replacements[component_name] = candidate

        original_components = {
            component_name: getattr(self, component_name)
            for component_name in replacements
        }
        original_sources = self._value_sources.copy()
        with self._config_lock:
            try:
                for component_name, component in replacements.items():
                    setattr(self, component_name, component)
                for component_name, attr_name, _, _ in changes:
                    self._value_sources[(component_name, attr_name)] = ConfigSource(
                        ConfigSourceType.RUNTIME, f"Profile: {profile_name}"
                    )
                    self._invalidate_cache(component_name, attr_name)
                self._ensure_directories()
            except Exception:
                for component_name, component in original_components.items():
                    setattr(self, component_name, component)
                self._value_sources = original_sources
                self.clear_caches()
                raise

        for component_name, attr_name, old_value, new_value in changes:
            self._notify_observers(component_name, attr_name, old_value, new_value)

        import logging

        logging.getLogger(__name__).info(
            f"Applied configuration profile: {profile_name}"
        )

    def get_component_status(self, component_name: ComponentName) -> Dict[str, Any]:
        """
        Get the status of a configuration component.

        Returns metadata about the component including error counts,
        access frequency, and validation status.

        Args:
            component_name: Name of the component

        Returns:
            Dictionary with component status information

        Raises:
            ValueError: If the component doesn't exist
        """
        if component_name not in self._component_registry:
            raise ValueError(f"Unknown component '{component_name}'")

        component = self.get_component(component_name)

        # Get basic component info
        info: Dict[str, Any] = {
            "name": component_name,
            "type": component.__class__.__name__,
            "error_count": self._error_counts.get(component_name, 0),
            "accessed": component_name in self._accessed_components,
            "dependencies": list(
                self._component_dependencies.get(component_name, set())
            ),
        }

        validation_error = self._component_validation_error(component)
        info["validation"] = (
            "valid" if validation_error is None else f"invalid: {validation_error}"
        )

        return info

    def clear_caches(self) -> None:
        """
        Clear all internal caches.

        This can be useful when significant configuration changes
        have occurred and cached values might be stale.
        """
        with self._config_lock:
            self._value_cache.clear()
            self.get_cached_value.cache_clear()

    def __str__(self) -> str:
        """Human-readable representation of the configuration."""
        return f"Word Forge Configuration (v{'.'.join(map(str, self.version))})"


# ==========================================
# Global Configuration Instance
# ==========================================

# Global configuration instance for application-wide access
config: Final[Config] = Config()


# ==========================================
# Module Exports - Configuration Instance
# ==========================================

__all__ = [
    # Core configuration class and instance
    "Config",
    "config",
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
    # Enhanced types
    "ConfigChangeEvent",
    "ConfigSource",
    "ConfigSourceType",
    "RuntimeAdaptiveMode",
    "ConfigObserver",
    "ConfigVersion",
]

# ==========================================
# Module Exports - Utility Functions
# ==========================================

__all__ += [
    "serialize_config",
    "serialize_dataclass",
]
