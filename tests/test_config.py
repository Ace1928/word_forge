"""Tests for :mod:`word_forge.config`."""

import importlib
from pathlib import Path

import pytest


def test_get_full_path_joins_data_dir(monkeypatch, tmp_path):
    """Verify :func:`Config.get_full_path` joins ``parser.data_dir`` with a relative path."""

    # Ensure repository source is on ``sys.path`` for import reliability
    repo_src = Path(__file__).resolve().parents[1] / "src"
    monkeypatch.syspath_prepend(str(repo_src))

    # Override parser data directory before reloading the config module
    monkeypatch.setenv("WORD_FORGE_DATA_DIR", str(tmp_path))
    import word_forge.config as cfg

    importlib.reload(cfg)
    result = cfg.config.get_full_path("example.txt")
    assert result == Path(tmp_path) / "example.txt"

