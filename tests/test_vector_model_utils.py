"""Tests for embedding-model compatibility helpers."""

from word_forge.vectorizer.model_utils import get_embedding_dimension


class _ModernModel:
    def get_embedding_dimension(self) -> int:
        return 384

    def get_sentence_embedding_dimension(self) -> int:
        raise AssertionError("legacy getter must not run when the modern API exists")


class _LegacyModel:
    def get_sentence_embedding_dimension(self) -> int:
        return 768


class _InvalidModel:
    def get_embedding_dimension(self) -> int:
        return 0


def test_embedding_dimension_prefers_modern_api() -> None:
    assert get_embedding_dimension(_ModernModel()) == 384


def test_embedding_dimension_supports_legacy_api() -> None:
    assert get_embedding_dimension(_LegacyModel()) == 768


def test_embedding_dimension_rejects_invalid_values() -> None:
    assert get_embedding_dimension(_InvalidModel()) is None
    assert get_embedding_dimension(object()) is None
