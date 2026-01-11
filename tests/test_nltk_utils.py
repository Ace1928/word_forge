"""Tests for the centralized NLTK setup helpers."""

from __future__ import annotations

import importlib
from typing import List
from unittest.mock import MagicMock

import word_forge.utils.nltk_utils as nltk_utils


def test_ensure_nltk_data_downloads_missing(monkeypatch):
    """Test that ensure_nltk_data downloads missing NLTK resources."""
    module = importlib.reload(nltk_utils)
    downloads: List[str] = []

    # Reset initialization state
    monkeypatch.setattr(module, "_initialized", False)

    # Create a mock that simulates missing NLTK data
    mock_data = MagicMock()
    mock_data.find.side_effect = LookupError("Resource not found")

    mock_nltk = MagicMock()
    mock_nltk.data = mock_data
    mock_nltk.download = lambda pkg, quiet=True: downloads.append(pkg)

    monkeypatch.setattr(module, "nltk", mock_nltk, raising=False)

    downloaded = module.ensure_nltk_data()
    assert downloads == downloaded

    expected_paths = [resource.path for resource in module._NLTK_RESOURCES]
    actual_calls = [call[0][0] for call in mock_data.find.call_args_list]
    assert actual_calls == expected_paths

    # A second call should be a no-op (already initialized)
    downloads.clear()
    mock_data.reset_mock()
    assert module.ensure_nltk_data() == []
    assert downloads == []
    assert mock_data.find.call_count == 0


def test_ensure_nltk_data_already_present(monkeypatch):
    """Test that ensure_nltk_data doesn't download when data exists."""
    module = importlib.reload(nltk_utils)
    downloads: List[str] = []

    monkeypatch.setattr(module, "_initialized", False)

    # Create a mock that simulates data being present (no LookupError)
    mock_data = MagicMock()
    mock_data.find.return_value = "/some/path"

    mock_nltk = MagicMock()
    mock_nltk.data = mock_data
    mock_nltk.download = lambda pkg, quiet=True: downloads.append(pkg)

    monkeypatch.setattr(module, "nltk", mock_nltk, raising=False)

    downloaded = module.ensure_nltk_data()

    # Should not download anything since data is present
    assert downloads == []
    assert downloaded == []
