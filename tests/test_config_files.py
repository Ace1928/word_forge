"""Tests for strict, atomic JSON and TOML configuration loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from word_forge.config import Config


def test_load_json_configuration(tmp_path: Path) -> None:
    path = tmp_path / "word-forge.json"
    path.write_text(
        json.dumps({"queue": {"batch_size": 12, "track_metrics": True}}),
        encoding="utf-8",
    )

    config = Config()
    config.load_from_file(path)

    assert config.queue.batch_size == 12
    assert config.queue.track_metrics is True


def test_load_toml_configuration(tmp_path: Path) -> None:
    path = tmp_path / "word-forge.toml"
    path.write_text("[queue]\nbatch_size = 24\n", encoding="utf-8")

    config = Config()
    config.load_from_file(path)

    assert config.queue.batch_size == 24


def test_unknown_field_does_not_partially_apply(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps({"queue": {"batch_size": 99, "batch_szie": 1}}),
        encoding="utf-8",
    )
    config = Config()
    original = config.queue.batch_size

    with pytest.raises(ValueError, match="batch_szie"):
        config.load_from_file(path)

    assert config.queue.batch_size == original
