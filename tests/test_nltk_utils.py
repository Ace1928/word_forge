"""Tests for the centralized NLTK setup helpers."""

from __future__ import annotations

import importlib
from typing import List

import word_forge.utils.nltk_utils as nltk_utils


def test_ensure_nltk_data_downloads_missing(monkeypatch):
    module = importlib.reload(nltk_utils)
    downloads: List[str] = []

    class DummyData:
        def __init__(self) -> None:
            self.calls: List[str] = []

        def find(self, path: str) -> None:
            self.calls.append(path)
            raise LookupError

    dummy_data = DummyData()
    monkeypatch.setattr(module, "_initialized", False)
    stub_nltk = type("StubNltk", (), {})()
    stub_nltk.data = dummy_data
    stub_nltk.download = lambda pkg, quiet=True: downloads.append(pkg)
    monkeypatch.setattr(module, "nltk", stub_nltk, raising=False)

    downloaded = module.ensure_nltk_data()
    assert downloads == downloaded
    expected_paths = [resource.path for resource in module._NLTK_RESOURCES]
    assert dummy_data.calls == expected_paths

    # A second call should be a no-op
    downloads.clear()
    dummy_data.calls.clear()
    assert module.ensure_nltk_data() == []
    assert downloads == []
    assert dummy_data.calls == []
