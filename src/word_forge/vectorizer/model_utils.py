"""Compatibility helpers for embedding model metadata."""

from typing import Optional


def get_embedding_dimension(model: object) -> Optional[int]:
    """Return a validated embedding dimension across model API versions.

    Sentence Transformers 5 exposes ``get_embedding_dimension`` while older
    supported releases expose ``get_sentence_embedding_dimension``. The modern
    API is preferred so current releases do not emit deprecation warnings.

    Args:
        model: Embedding model that provides a dimension getter.

    Returns:
        A positive embedding dimension, or ``None`` when the model does not
        expose a supported getter or returns an invalid value.
    """
    getter = getattr(model, "get_embedding_dimension", None)
    if not callable(getter):
        getter = getattr(model, "get_sentence_embedding_dimension", None)
    if not callable(getter):
        return None

    dimension = getter()
    if isinstance(dimension, bool) or not isinstance(dimension, int):
        return None
    return dimension if dimension > 0 else None
