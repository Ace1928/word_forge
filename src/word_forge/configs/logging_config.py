"""
Logging configuration system for Word Forge.

This module defines the configuration schema for the Word Forge logging system,
including log levels, formats, rotation policies, and output destinations.

Architecture:
    ┌─────────────────────┐
    │   LoggingConfig     │
    └───────────┬─────────┘
                │
    ┌───────────┴─────────┐
    │     Components      │
    └─────────────────────┘
    ┌─────┬─────┬─────┬───────┬─────┐
    │Level│Form │Dest │Rotation│Path │
    └─────┴─────┴─────┴───────┴─────┘
"""

from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

from word_forge.configs.config_essentials import (
    LOGS_ROOT,
    LogDestination,
    LogFormatTemplate,
    LoggingConfigError,
    LogLevel,
    LogRotationStrategy,
)
from word_forge.configs.config_types import EnvMapping


@dataclass
class LoggingConfig:
    """
    Configuration for Word Forge logging system.

    Controls log levels, formats, file paths, and rotation strategies
    for the application's logging infrastructure.

    Attributes:
        level: Logging level threshold (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Log message format string
        file_path: Path to log file (None = console logging only)
        destination: Where logs should be sent (console, file, both, syslog)
        rotation_strategy: Log rotation method (size, time, none)
        max_file_size_mb: Maximum log file size before rotation in MB
        max_files: Maximum number of rotated log files to keep
        include_timestamp_in_filename: Whether to add timestamps to log filenames
        propagate_to_root: Whether to propagate logs to the root logger
        log_exceptions: Whether to automatically log uncaught exceptions
        log_initialization: Whether to log when configuration is initialized

    Usage:
        from word_forge.config import config

        # Get log level
        level = config.logging.level

        # Get formatted log path
        log_path = config.logging.get_log_path_with_timestamp()

        # Check if file logging is enabled
        uses_file = config.logging.uses_file_logging
    """

    # Log level and format
    level: LogLevel = "INFO"
    format: str = LogFormatTemplate.STANDARD.value

    # Log file settings
    file_path: Optional[str] = str(LOGS_ROOT / "word_forge.log")
    destination: LogDestination = LogDestination.BOTH

    # Rotation settings
    rotation_strategy: LogRotationStrategy = LogRotationStrategy.SIZE
    max_file_size_mb: int = 10
    max_files: int = 5

    # Advanced options
    include_timestamp_in_filename: bool = False
    propagate_to_root: bool = False
    log_exceptions: bool = True
    log_initialization: bool = True

    # Environment variable mapping for configuration overrides
    ENV_VARS: ClassVar[EnvMapping] = {
        "WORD_FORGE_LOG_LEVEL": ("level", str),
        "WORD_FORGE_LOG_FILE": ("file_path", str),
        "WORD_FORGE_LOG_FORMAT": ("format", str),
        "WORD_FORGE_LOG_DESTINATION": ("destination", LogDestination),
        "WORD_FORGE_LOG_ROTATION": ("rotation_strategy", LogRotationStrategy),
        "WORD_FORGE_LOG_MAX_SIZE": ("max_file_size_mb", int),
        "WORD_FORGE_LOG_EXCEPTIONS": ("log_exceptions", bool),
    }

    # ==========================================
    # Cached Properties
    # ==========================================

    @cached_property
    def get_log_path(self) -> Optional[Path]:
        """
        Get log file path as Path object if set.

        Returns:
            Path: Path object representing the log file location,
                  or None if file logging is disabled
        """
        if not self.file_path:
            return None
        return Path(self.file_path)

    @cached_property
    def uses_file_logging(self) -> bool:
        """
        Determine if file logging is enabled based on configuration.

        Returns:
            bool: True if logs are written to a file, False otherwise
        """
        return (
            self.destination in (LogDestination.FILE, LogDestination.BOTH)
            and self.file_path is not None
        )

    @cached_property
    def uses_console_logging(self) -> bool:
        """
        Determine if console logging is enabled based on configuration.

        Returns:
            bool: True if logs are written to console, False otherwise
        """
        return self.destination in (LogDestination.CONSOLE, LogDestination.BOTH)

    # ==========================================
    # Public Methods
    # ==========================================

    def get_log_path_with_timestamp(self) -> Optional[Path]:
        """
        Get log path with timestamp if that option is enabled.

        Generates a filename with timestamp inserted before extension if
        include_timestamp_in_filename is True.

        Returns:
            Path: Path with timestamp added, or regular path if not enabled,
                  or None if file_path is None
        """
        if not self.file_path:
            return None

        path = Path(self.file_path)

        if not self.include_timestamp_in_filename:
            return path

        # Insert timestamp before extension
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = path.stem
        suffix = path.suffix

        return path.with_name(f"{stem}_{timestamp}{suffix}")

    def with_level(self, level: LogLevel) -> "LoggingConfig":
        """
        Create a new configuration with modified log level.

        Args:
            level: New log level value

        Returns:
            LoggingConfig: New configuration instance with updated level
        """
        return LoggingConfig(
            level=level,
            format=self.format,
            file_path=self.file_path,
            destination=self.destination,
            rotation_strategy=self.rotation_strategy,
            max_file_size_mb=self.max_file_size_mb,
            max_files=self.max_files,
            include_timestamp_in_filename=self.include_timestamp_in_filename,
            propagate_to_root=self.propagate_to_root,
            log_exceptions=self.log_exceptions,
            log_initialization=self.log_initialization,
        )

    def with_format_template(self, template: LogFormatTemplate) -> "LoggingConfig":
        """
        Create a new configuration using a predefined format template.

        Args:
            template: Log format template to use

        Returns:
            LoggingConfig: New configuration instance with updated format
        """
        return LoggingConfig(
            level=self.level,
            format=template.value,
            file_path=self.file_path,
            destination=self.destination,
            rotation_strategy=self.rotation_strategy,
            max_file_size_mb=self.max_file_size_mb,
            max_files=self.max_files,
            include_timestamp_in_filename=self.include_timestamp_in_filename,
            propagate_to_root=self.propagate_to_root,
            log_exceptions=self.log_exceptions,
            log_initialization=self.log_initialization,
        )

    def with_destination(self, destination: LogDestination) -> "LoggingConfig":
        """
        Create a new configuration with modified log destination.

        Args:
            destination: New log destination value

        Returns:
            LoggingConfig: New configuration instance with updated destination
        """
        config = LoggingConfig(
            level=self.level,
            format=self.format,
            file_path=self.file_path,
            destination=destination,
            rotation_strategy=self.rotation_strategy,
            max_file_size_mb=self.max_file_size_mb,
            max_files=self.max_files,
            include_timestamp_in_filename=self.include_timestamp_in_filename,
            propagate_to_root=self.propagate_to_root,
            log_exceptions=self.log_exceptions,
            log_initialization=self.log_initialization,
        )

        # Handle special case for FILE destination with no file path
        if (
            destination in (LogDestination.FILE, LogDestination.BOTH)
            and not config.file_path
        ):
            config.file_path = str(LOGS_ROOT / "word_forge.log")

        return config

    def get_rotation_config(self) -> Dict[str, Any]:
        """
        Get rotation-specific configuration parameters.

        Returns:
            Dict[str, Any]: Dictionary with rotation settings
        """
        if not self.uses_file_logging:
            return {"enabled": False}

        return {
            "enabled": self.rotation_strategy != LogRotationStrategy.NONE,
            "strategy": self.rotation_strategy.value,
            "max_size_mb": self.max_file_size_mb,
            "max_files": self.max_files,
        }

    def validate(self) -> None:
        """
        Validate the configuration for consistency and correctness.

        Raises:
            LoggingConfigError: If any validation fails
        """
        errors = []

        # Validate destination vs file_path consistency
        if (
            self.destination in (LogDestination.FILE, LogDestination.BOTH)
            and not self.file_path
        ):
            errors.append("File logging enabled but no file path specified")

        # Validate max file size
        if self.max_file_size_mb <= 0:
            errors.append(
                f"Maximum file size must be positive, got {self.max_file_size_mb}"
            )

        # Validate max files
        if self.max_files <= 0:
            errors.append(
                f"Maximum number of files must be positive, got {self.max_files}"
            )

        # Validate rotation settings
        if (
            self.rotation_strategy == LogRotationStrategy.SIZE
            and self.uses_file_logging
            and self.max_file_size_mb <= 0
        ):
            errors.append("Size-based rotation requires positive max_file_size_mb")

        if errors:
            raise LoggingConfigError(
                f"Configuration validation failed: {'; '.join(errors)}"
            )

    def get_python_logging_config(self) -> Dict[str, Any]:
        """
        Convert configuration to Python's logging module configuration dict.

        Returns:
            Dict[str, Any]: Configuration dictionary compatible with
                            logging.config.dictConfig()
        """
        handlers = []

        if self.uses_console_logging:
            handlers.append("console")

        if self.uses_file_logging:
            handlers.append("file")

        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"standard": {"format": self.format}},
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": self.level,
                    "formatter": "standard",
                    "stream": "ext://sys.stdout",
                }
            },
            "loggers": {
                "word_forge": {
                    "level": self.level,
                    "handlers": handlers,
                    "propagate": self.propagate_to_root,
                }
            },
        }

        if self.uses_file_logging and self.file_path:
            if self.rotation_strategy == LogRotationStrategy.SIZE:
                config["handlers"]["file"] = {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": self.level,
                    "formatter": "standard",
                    "filename": self.file_path,
                    "maxBytes": self.max_file_size_mb * 1024 * 1024,
                    "backupCount": self.max_files,
                }
            elif self.rotation_strategy == LogRotationStrategy.TIME:
                config["handlers"]["file"] = {
                    "class": "logging.handlers.TimedRotatingFileHandler",
                    "level": self.level,
                    "formatter": "standard",
                    "filename": self.file_path,
                    "when": "midnight",
                    "backupCount": self.max_files,
                }
            else:
                config["handlers"]["file"] = {
                    "class": "logging.FileHandler",
                    "level": self.level,
                    "formatter": "standard",
                    "filename": self.file_path,
                }

        return config


# ==========================================
# Module Exports
# ==========================================

__all__ = [
    "LoggingConfig",
    "LogLevel",
    "LogFormatTemplate",
    "LogRotationStrategy",
    "LogDestination",
    "LoggingConfigError",
]
