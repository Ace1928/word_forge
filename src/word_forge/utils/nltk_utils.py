"""NLTK resource management utilities."""

from __future__ import annotations

import logging
from typing import NamedTuple

import nltk

from word_forge.observability import eidosian


class _NLTKResource(NamedTuple):
    """Description of an NLTK package and the path used to locate it."""

    package: str
    path: str
    description: str


# Resources required across the codebase
_NLTK_RESOURCES: tuple[_NLTKResource, ...] = (
    _NLTKResource("wordnet", "corpora/wordnet", "WordNet lexical database"),
    _NLTKResource("omw-1.4", "corpora/omw-1.4", "Open Multilingual WordNet"),
    _NLTKResource("punkt", "tokenizers/punkt", "Punkt sentence tokenizer"),
    _NLTKResource(
        "averaged_perceptron_tagger",
        "taggers/averaged_perceptron_tagger",
        "Averaged perceptron POS tagger",
    ),
    _NLTKResource("stopwords", "corpora/stopwords", "Common stop words"),
    _NLTKResource(
        "maxent_ne_chunker",
        "chunkers/maxent_ne_chunker",
        "Named entity chunker",
    ),
    _NLTKResource("words", "corpora/words", "Word frequency lists"),
    _NLTKResource("vader_lexicon", "sentiment/vader_lexicon", "VADER sentiment lexicon"),
)

_initialized = False


@eidosian()
def ensure_nltk_data(logger: logging.Logger | None = None) -> list[str]:
    """Ensure that required NLTK data packages are available."""

    global _initialized
    if _initialized:
        if logger:
            logger.info("NLTK resources already initialized; nothing to download.")
        return []

    downloaded: list[str] = []
    for resource in _NLTK_RESOURCES:
        try:
            nltk.data.find(resource.path)
        except LookupError:
            nltk.download(resource.package, quiet=True)
            downloaded.append(resource.package)
            if logger:
                logger.info(
                    "Downloaded NLTK resource %s (%s)",
                    resource.package,
                    resource.description,
                )

    if logger:
        if downloaded:
            logger.info(
                "Fetched %d NLTK resource(s): %s",
                len(downloaded),
                ", ".join(downloaded),
            )
        else:
            logger.info("All required NLTK corpora already present.")

    _initialized = True
    return downloaded


__all__ = ["ensure_nltk_data"]
