"""NLTK resource management utilities."""

from __future__ import annotations

import nltk

# Resources required across the codebase
_NLTK_RESOURCES = (
    "wordnet",
    "omw-1.4",
    "punkt",
    "averaged_perceptron_tagger",
    "stopwords",
    "maxent_ne_chunker",
    "words",
)

_initialized = False


def ensure_nltk_data() -> None:
    """Ensure that required NLTK data packages are available."""
    global _initialized
    if _initialized:
        return

    for resource in _NLTK_RESOURCES:
        try:
            nltk.data.find(resource)  # type: ignore[arg-type]
        except LookupError:
            nltk.download(resource, quiet=True)  # type: ignore
    _initialized = True


__all__ = ["ensure_nltk_data"]
