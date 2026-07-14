"""Tests for :mod:`word_forge.config`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from word_forge.config import Config, ConfigSourceType
from word_forge.configs.config_essentials import ConfigError


def test_get_full_path_joins_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Environment-based data paths are applied without module reloads."""
    monkeypatch.setenv("WORD_FORGE_DATA_DIR", str(tmp_path))
    config = Config()

    result = config.get_full_path("example.txt")

    assert result == tmp_path / "example.txt"


def test_load_json_updates_mutable_and_frozen_components(tmp_path: Path) -> None:
    """File loading supports every dataclass mutability policy."""
    config_path = tmp_path / "word-forge.json"
    config_path.write_text(
        json.dumps(
            {
                "parser": {"enable_model": False},
                "vectorizer": {"batch_size": 7},
            }
        ),
        encoding="utf-8",
    )
    config = Config()

    config.load_from_file(config_path)

    assert config.parser.enable_model is False
    assert config.vectorizer.batch_size == 7
    _, source = config.get_value_with_source("vectorizer", "batch_size")
    assert source.type is ConfigSourceType.FILE


def test_exported_json_can_be_loaded(tmp_path: Path) -> None:
    """An exported configuration is a supported input document."""
    source = Config()
    config_path = tmp_path / "export.json"
    source.export_to_file(config_path)
    loaded = Config()

    loaded.load_from_file(config_path)

    assert loaded.to_dict()["database"] == source.to_dict()["database"]
    assert loaded.to_dict()["vectorizer"] == source.to_dict()["vectorizer"]


def test_load_from_file_is_transactional(tmp_path: Path) -> None:
    """A later invalid field cannot apply earlier changes."""
    config_path = tmp_path / "invalid.json"
    config_path.write_text(
        json.dumps(
            {
                "parser": {"enable_model": False},
                "vectorizer": {"unknown_option": 7},
            }
        ),
        encoding="utf-8",
    )
    config = Config()
    original_parser = config.parser

    with pytest.raises(ConfigError, match="unknown_option"):
        config.load_from_file(config_path)

    assert config.parser is original_parser


def test_environment_value_takes_precedence_over_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Configuration source priority is default < file < environment."""
    monkeypatch.setenv("WORD_FORGE_VECTOR_BATCH_SIZE", "5")
    config_path = tmp_path / "word-forge.json"
    config_path.write_text(
        json.dumps({"vectorizer": {"batch_size": 99}}), encoding="utf-8"
    )
    config = Config()

    config.load_from_file(config_path)

    assert config.vectorizer.batch_size == 5
    _, source = config.get_value_with_source("vectorizer", "batch_size")
    assert source.type is ConfigSourceType.ENVIRONMENT


def test_multiple_environment_values_compose_for_frozen_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One frozen-component environment override cannot erase another."""
    monkeypatch.setenv("WORD_FORGE_VECTOR_BATCH_SIZE", "5")
    monkeypatch.setenv("WORD_FORGE_VECTOR_DIMENSION", "384")

    config = Config()

    assert config.vectorizer.batch_size == 5
    assert config.vectorizer.dimension == 384


def test_runtime_updates_frozen_component_and_invalidates_cache() -> None:
    """Runtime changes work for frozen components and cannot return stale data."""
    config = Config()
    original = config.get_cached_value("vectorizer", "batch_size", int)

    config.set_runtime_value("vectorizer", "batch_size", original + 1)

    assert config.vectorizer.batch_size == original + 1
    assert config.get_cached_value("vectorizer", "batch_size", int) == original + 1


def test_load_yaml(tmp_path: Path) -> None:
    """YAML is a safe, supported core configuration format."""
    config_path = tmp_path / "word-forge.yaml"
    config_path.write_text("parser:\n  enable_model: false\n", encoding="utf-8")
    config = Config()

    config.load_from_file(config_path)

    assert config.parser.enable_model is False


def test_load_coerces_enums_sets_tuples_and_merges_mappings(tmp_path: Path) -> None:
    """Structured values retain their declared runtime types after loading."""
    config_path = tmp_path / "typed.yaml"
    config_path.write_text(
        """
database:
  dialect: sqlite
  pragmas:
    cache_size: "-8000"
emotion:
  valence_range: [-0.5, 0.75]
graph:
  active_dimensions: [lexical, contextual]
vectorizer:
  storage_type: memory
""".strip(),
        encoding="utf-8",
    )
    config = Config()

    config.load_from_file(config_path)

    assert config.database.dialect.value == "sqlite"
    assert config.database.pragmas["cache_size"] == "-8000"
    assert config.database.pragmas["foreign_keys"] == "ON"
    assert config.emotion.valence_range == (-0.5, 0.75)
    assert config.graph.active_dimensions == {"lexical", "contextual"}
    assert config.vectorizer.storage_type.value == "memory"


def test_semantically_invalid_file_is_transactional(tmp_path: Path) -> None:
    """Component validators run before any file values are committed."""
    config_path = tmp_path / "invalid-value.json"
    config_path.write_text(
        json.dumps(
            {
                "parser": {"enable_model": False},
                "vectorizer": {"batch_size": 0},
            }
        ),
        encoding="utf-8",
    )
    config = Config()
    original_parser = config.parser

    with pytest.raises(ConfigError, match="(?i)batch size"):
        config.load_from_file(config_path)

    assert config.parser is original_parser


def test_runtime_value_takes_precedence_over_file(tmp_path: Path) -> None:
    """Explicit runtime settings remain authoritative during later file loads."""
    config_path = tmp_path / "word-forge.json"
    config_path.write_text(
        json.dumps({"vectorizer": {"batch_size": 99}}), encoding="utf-8"
    )
    config = Config()
    config.set_runtime_value("vectorizer", "batch_size", 6)

    config.load_from_file(config_path)

    assert config.vectorizer.batch_size == 6
    _, source = config.get_value_with_source("vectorizer", "batch_size")
    assert source.type is ConfigSourceType.RUNTIME


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    """Ambiguous duplicate JSON keys fail instead of silently taking the last."""
    config_path = tmp_path / "duplicate.json"
    config_path.write_text(
        '{"parser":{"enable_model":true,"enable_model":false}}',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="Duplicate JSON key"):
        Config().load_from_file(config_path)


@pytest.mark.parametrize(
    ("profile_name", "attribute_name"),
    [
        ("development", "storage_type"),
        ("production", "storage_type"),
        ("testing", "batch_size"),
        ("high_performance", "batch_size"),
        ("low_memory", "batch_size"),
    ],
)
def test_profiles_are_valid_and_track_runtime_source(
    profile_name: str, attribute_name: str
) -> None:
    """Every bundled profile uses real fields and passes semantic validation."""
    config = Config()

    config.apply_profile(profile_name)

    assert not any(config.validate_all().values())
    _, source = config.get_value_with_source("vectorizer", attribute_name)
    assert source.type is ConfigSourceType.RUNTIME
    assert source.location == f"Profile: {profile_name}"


def test_validate_all_detects_result_style_failures() -> None:
    """Validators returning ``Result.failure`` are not mistaken for success."""
    config = Config()
    config.database.pool_size = 0

    validation = config.validate_all()

    assert "Pool size must be positive" in validation["database"][0]


@pytest.mark.parametrize(
    "payload, message",
    [
        ([], "top-level mapping"),
        ({"unknown": {}}, "Unknown configuration component"),
        ({"parser": {"enable_model": "yes"}}, "must have type bool"),
    ],
)
def test_load_rejects_invalid_shapes(
    tmp_path: Path, payload: object, message: str
) -> None:
    """Invalid structure and types produce actionable failures."""
    config_path = tmp_path / "invalid-shape.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        Config().load_from_file(config_path)
