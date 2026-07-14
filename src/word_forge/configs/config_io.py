"""Safe parsing and type coercion for Word Forge configuration files."""

from __future__ import annotations

import json
import logging
import types
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, Union, get_args, get_origin, get_type_hints, is_typeddict

import yaml  # type: ignore[import-untyped]

from word_forge.configs.config_essentials import ConfigError, PathLike

MAX_CONFIG_FILE_BYTES = 1_048_576


class ConfigurationFileError(ConfigError):
    """Raised when a configuration document cannot be parsed or validated."""


def load_configuration_document(path: PathLike) -> dict[str, object]:
    """Read a bounded JSON or YAML configuration document safely.

    Args:
        path: JSON, YAML, or YML file to read.

    Returns:
        Parsed top-level mapping.

    Raises:
        ConfigurationFileError: If the file is missing, too large, malformed,
            uses an unsupported extension, or does not contain a mapping.
    """
    config_path = Path(path).expanduser()
    try:
        file_size = config_path.stat().st_size
    except OSError as exc:
        raise ConfigurationFileError(
            f"Cannot access configuration file '{config_path}': {exc}"
        ) from exc

    if file_size > MAX_CONFIG_FILE_BYTES:
        raise ConfigurationFileError(
            f"Configuration file '{config_path}' is {file_size} bytes; "
            f"the maximum is {MAX_CONFIG_FILE_BYTES} bytes"
        )

    try:
        document_text = config_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ConfigurationFileError(
            f"Cannot read configuration file '{config_path}' as UTF-8: {exc}"
        ) from exc

    suffix = config_path.suffix.casefold()
    try:
        if suffix == ".json":
            parsed = json.loads(document_text, object_pairs_hook=_unique_json_object)
        elif suffix in {".yaml", ".yml"}:
            parsed = yaml.safe_load(document_text)
        else:
            raise ConfigurationFileError(
                f"Unsupported configuration format '{suffix or '<none>'}'; "
                "use .json, .yaml, or .yml"
            )
    except ConfigurationFileError:
        raise
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ConfigurationFileError(
            f"Malformed configuration file '{config_path}': {exc}"
        ) from exc

    if parsed is None:
        return {}
    if not isinstance(parsed, Mapping):
        raise ConfigurationFileError(
            f"Configuration file '{config_path}' must contain a top-level mapping "
            "(top-level object)"
        )
    if not all(isinstance(key, str) for key in parsed):
        raise ConfigurationFileError(
            f"Configuration file '{config_path}' contains a non-string section name"
        )
    return {str(key): value for key, value in parsed.items()}


def merge_configuration_value(current: object, supplied: object) -> object:
    """Recursively merge mapping overrides while replacing scalar values.

    Args:
        current: Existing configuration value.
        supplied: Value supplied by a configuration source.

    Returns:
        A merged value without mutating either input.
    """
    if not isinstance(current, Mapping) or not isinstance(supplied, Mapping):
        return supplied

    merged: dict[object, object] = dict(current)
    for key, value in supplied.items():
        merged[key] = merge_configuration_value(merged.get(key), value)
    return merged


def coerce_configuration_value(
    value: object, expected_type: object, location: str
) -> object:
    """Coerce parsed data to a declared configuration field type.

    Args:
        value: Parsed JSON or YAML value.
        expected_type: Runtime type annotation for the target field.
        location: Dotted field path used in diagnostics.

    Returns:
        Value compatible with ``expected_type``.

    Raises:
        ConfigurationFileError: If the value violates the declared type.
    """
    origin = get_origin(expected_type)
    arguments = get_args(expected_type)

    if expected_type is object:
        return value
    if origin in {Union, types.UnionType}:
        if value is None and type(None) in arguments:
            return None
        failures: list[str] = []
        for candidate in arguments:
            if candidate is type(None):
                continue
            try:
                return coerce_configuration_value(value, candidate, location)
            except ConfigurationFileError as exc:
                failures.append(str(exc))
        raise _type_error(location, expected_type, value, failures)
    if origin is Literal:
        if value in arguments and not (
            isinstance(value, bool)
            and all(not isinstance(candidate, bool) for candidate in arguments)
        ):
            return value
        raise _type_error(location, expected_type, value)
    if isinstance(expected_type, type) and issubclass(expected_type, Enum):
        return _coerce_enum(value, expected_type, location)
    if expected_type is bool:
        if type(value) is bool:
            return value
        raise _type_error(location, expected_type, value)
    if expected_type is int:
        if type(value) is int:
            return value
        if location == "logging.level" and isinstance(value, str):
            level = _logging_level(value)
            if level is not None:
                return level
        raise _type_error(location, expected_type, value)
    if expected_type is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        raise _type_error(location, expected_type, value)
    if expected_type is str:
        if isinstance(value, str):
            return value
        raise _type_error(location, expected_type, value)
    if expected_type is tuple:
        if isinstance(value, (list, tuple)):
            return tuple(value)
        raise _type_error(location, expected_type, value)
    if expected_type is list:
        if isinstance(value, list):
            return value
        raise _type_error(location, expected_type, value)
    if expected_type is set:
        if isinstance(value, (list, set, tuple)):
            return set(value)
        raise _type_error(location, expected_type, value)
    if expected_type is dict:
        if isinstance(value, Mapping):
            return dict(value)
        raise _type_error(location, expected_type, value)
    if is_typeddict(expected_type):
        return _coerce_typed_mapping(value, expected_type, location)
    if origin in {dict, Mapping}:
        return _coerce_mapping(value, arguments, location)
    if origin is list:
        if not isinstance(value, list):
            raise _type_error(location, expected_type, value)
        item_type = arguments[0] if arguments else object
        return [
            coerce_configuration_value(item, item_type, f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    if origin is set:
        if not isinstance(value, (list, set, tuple)):
            raise _type_error(location, expected_type, value)
        item_type = arguments[0] if arguments else object
        return {
            coerce_configuration_value(item, item_type, f"{location}[]")
            for item in value
        }
    if origin is tuple:
        return _coerce_tuple(value, arguments, expected_type, location)
    if isinstance(expected_type, type) and is_dataclass(expected_type):
        return _coerce_dataclass(value, expected_type, location)

    if isinstance(expected_type, type) and isinstance(value, expected_type):
        return value
    raise _type_error(location, expected_type, value)


def parse_environment_value(
    raw_value: str, declared_type: object, location: str
) -> object:
    """Parse one environment variable using its declared scalar type.

    Args:
        raw_value: Unparsed environment variable text.
        declared_type: Type declared in a component's ``ENV_VARS`` mapping.
        location: Environment variable name for diagnostics.

    Returns:
        Parsed scalar or the original string for subsequent field coercion.

    Raises:
        ConfigurationFileError: If a boolean or numeric value is malformed.
    """
    if isinstance(declared_type, str):
        scalar_types: dict[str, type[object]] = {
            "bool": bool,
            "float": float,
            "int": int,
            "str": str,
        }
        declared_type = scalar_types.get(declared_type.casefold(), str)

    if declared_type is bool:
        normalized = raw_value.strip().casefold()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ConfigurationFileError(
            f"{location} must be a boolean (true/false, yes/no, on/off, or 1/0)"
        )
    if declared_type is int:
        try:
            return int(raw_value)
        except ValueError as exc:
            raise ConfigurationFileError(f"{location} must be an integer") from exc
    if declared_type is float:
        try:
            return float(raw_value)
        except ValueError as exc:
            raise ConfigurationFileError(f"{location} must be a number") from exc
    if declared_type is str:
        return raw_value
    if callable(declared_type):
        try:
            return declared_type(raw_value)
        except (TypeError, ValueError) as exc:
            raise ConfigurationFileError(
                f"{location} has an invalid value {raw_value!r}: {exc}"
            ) from exc
    return raw_value


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object while rejecting ambiguous duplicate keys."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigurationFileError(f"Duplicate JSON key '{key}'")
        result[key] = value
    return result


def _coerce_enum(value: object, enum_type: type[Enum], location: str) -> Enum:
    """Coerce an enum by exact value or case-insensitive member name."""
    if isinstance(value, enum_type):
        return value
    for member in enum_type:
        if value == member.value:
            return member
        if isinstance(value, str) and value.casefold() == member.name.casefold():
            return member
    choices = ", ".join(str(member.value) for member in enum_type)
    raise ConfigurationFileError(
        f"{location} must be one of: {choices}; received {value!r}"
    )


def _coerce_mapping(
    value: object, arguments: tuple[object, ...], location: str
) -> dict[object, object]:
    """Coerce a generic mapping recursively."""
    if not isinstance(value, Mapping):
        raise _type_error(location, dict, value)
    key_type, value_type = arguments if len(arguments) == 2 else (object, object)
    result: dict[object, object] = {}
    for raw_key, raw_item in value.items():
        key = coerce_configuration_value(raw_key, key_type, f"{location}.<key>")
        try:
            hash(key)
        except TypeError as exc:
            raise ConfigurationFileError(
                f"{location} contains an unhashable key {key!r}"
            ) from exc
        result[key] = coerce_configuration_value(
            raw_item, value_type, f"{location}.{raw_key}"
        )
    return result


def _coerce_typed_mapping(
    value: object, expected_type: object, location: str
) -> dict[str, object]:
    """Coerce a ``TypedDict`` while rejecting unknown keys."""
    if not isinstance(value, Mapping):
        raise _type_error(location, expected_type, value)
    hints = get_type_hints(expected_type)
    unknown_keys = sorted(str(key) for key in value if key not in hints)
    is_total = bool(getattr(expected_type, "__total__", True))
    if unknown_keys and is_total:
        raise ConfigurationFileError(
            f"{location} contains unknown keys: {', '.join(unknown_keys)}"
        )
    known_value_types = set(hints.values())
    fallback_type = (
        next(iter(known_value_types)) if len(known_value_types) == 1 else object
    )
    result: dict[str, object] = {}
    for raw_key, raw_item in value.items():
        if not isinstance(raw_key, str):
            raise ConfigurationFileError(f"{location} keys must be strings")
        result[raw_key] = coerce_configuration_value(
            raw_item, hints.get(raw_key, fallback_type), f"{location}.{raw_key}"
        )
    return result


def _coerce_tuple(
    value: object,
    arguments: tuple[object, ...],
    expected_type: object,
    location: str,
) -> tuple[object, ...]:
    """Coerce fixed-length and variadic tuples."""
    if not isinstance(value, (list, tuple)):
        raise _type_error(location, expected_type, value)
    if not arguments:
        return tuple(value)
    if len(arguments) == 2 and arguments[1] is Ellipsis:
        return tuple(
            coerce_configuration_value(item, arguments[0], f"{location}[]")
            for item in value
        )
    if len(value) != len(arguments):
        raise ConfigurationFileError(
            f"{location} requires {len(arguments)} items; received {len(value)}"
        )
    return tuple(
        coerce_configuration_value(item, item_type, f"{location}[{index}]")
        for index, (item, item_type) in enumerate(zip(value, arguments))
    )


def _coerce_dataclass(
    value: object, expected_type: type[object], location: str
) -> object:
    """Construct a nested dataclass from a mapping."""
    if not isinstance(value, Mapping):
        raise _type_error(location, expected_type, value)
    available_fields = {
        item.name: item
        for item in fields(expected_type)  # type: ignore[arg-type]
        if item.init and not item.name.startswith("_")
    }
    unknown_keys = sorted(str(key) for key in value if key not in available_fields)
    if unknown_keys:
        raise ConfigurationFileError(
            f"{location} contains unknown keys: {', '.join(unknown_keys)}"
        )
    hints = get_type_hints(expected_type)
    keyword_arguments = {
        str(key): coerce_configuration_value(
            item, hints.get(str(key), object), f"{location}.{key}"
        )
        for key, item in value.items()
    }
    try:
        return expected_type(**keyword_arguments)
    except (TypeError, ValueError) as exc:
        raise ConfigurationFileError(f"Invalid {location}: {exc}") from exc


def _logging_level(value: str) -> int | None:
    """Resolve a standard logging level name without private logging APIs."""
    levels = {
        "CRITICAL": logging.CRITICAL,
        "DEBUG": logging.DEBUG,
        "ERROR": logging.ERROR,
        "FATAL": logging.FATAL,
        "INFO": logging.INFO,
        "NOTSET": logging.NOTSET,
        "WARN": logging.WARNING,
        "WARNING": logging.WARNING,
    }
    return levels.get(value.strip().upper())


def _type_error(
    location: str,
    expected_type: object,
    value: object,
    details: list[str] | None = None,
) -> ConfigurationFileError:
    """Create a consistent type mismatch diagnostic."""
    expected_name = getattr(expected_type, "__name__", str(expected_type))
    readable_types = {
        "bool": "a boolean (must have type bool)",
        "float": "a number",
        "int": "an integer",
        "str": "a string",
    }
    requirement = readable_types.get(expected_name, f"type {expected_name}")
    message = (
        f"{location} must be {requirement}; received {type(value).__name__} ({value!r})"
    )
    if details:
        message = f"{message}. Alternatives failed: {'; '.join(details)}"
    return ConfigurationFileError(message)


__all__ = [
    "ConfigurationFileError",
    "MAX_CONFIG_FILE_BYTES",
    "coerce_configuration_value",
    "load_configuration_document",
    "merge_configuration_value",
    "parse_environment_value",
]
