"""NLTK resource management utilities."""

from __future__ import annotations

import logging
from typing import List, NamedTuple, Optional

import nltk  # type: ignore[import-untyped]

from word_forge.parser.wordnet_languages import MULTILINGUAL_WORDNET_PACKAGE


class _NLTKResource(NamedTuple):
    """Description of an NLTK package and the path used to locate it."""

    package: str
    path: str
    description: str


# Resources required across the codebase
_CORE_NLTK_RESOURCES: tuple[_NLTKResource, ...] = (
    _NLTKResource("wordnet", "corpora/wordnet", "WordNet lexical database"),
    _NLTKResource(
        "cmudict",
        "corpora/cmudict",
        "CMU Pronouncing Dictionary for American English",
    ),
    _NLTKResource("punkt", "tokenizers/punkt", "Punkt sentence tokenizer"),
    _NLTKResource(
        "punkt_tab",
        "tokenizers/punkt_tab",
        "Language-specific Punkt tokenizer tables",
    ),
    _NLTKResource(
        "averaged_perceptron_tagger",
        "taggers/averaged_perceptron_tagger",
        "Averaged perceptron POS tagger",
    ),
    _NLTKResource(
        "averaged_perceptron_tagger_eng",
        "taggers/averaged_perceptron_tagger_eng",
        "English averaged perceptron POS tagger",
    ),
    _NLTKResource("stopwords", "corpora/stopwords", "Common stop words"),
    _NLTKResource(
        "maxent_ne_chunker",
        "chunkers/maxent_ne_chunker",
        "Named entity chunker",
    ),
    _NLTKResource(
        "maxent_ne_chunker_tab",
        "chunkers/maxent_ne_chunker_tab",
        "Named entity chunker tables",
    ),
    _NLTKResource("words", "corpora/words", "Word frequency lists"),
    _NLTKResource(
        "vader_lexicon", "sentiment/vader_lexicon", "VADER sentiment lexicon"
    ),
)

_MULTILINGUAL_NLTK_RESOURCES: tuple[_NLTKResource, ...] = (
    _NLTKResource(
        MULTILINGUAL_WORDNET_PACKAGE,
        f"corpora/{MULTILINGUAL_WORDNET_PACKAGE}",
        "Open Multilingual Wordnet component datasets",
    ),
)

_initialized = False


class LexicalDataLicenseError(RuntimeError):
    """Raised when optional lexical data is requested without acknowledgement."""


def _resource_available(resource: _NLTKResource) -> bool:
    """Return whether a resource exists in extracted or archive form."""
    for candidate in (resource.path, f"{resource.path}.zip"):
        try:
            nltk.data.find(candidate)  # type: ignore[arg-type]
            return True
        except LookupError:
            continue
    return False


def ensure_nltk_data(
    logger: Optional[logging.Logger] = None,
    *,
    include_multilingual: bool = False,
    accept_source_licenses: bool = False,
) -> List[str]:
    """Ensure selected NLTK data packages are available.

    Open Multilingual Wordnet aggregates independently licensed wordnets and
    is therefore never downloaded by the unattended core path. Callers must
    explicitly request it and acknowledge responsibility for the component
    terms shipped in the selected snapshot.
    """

    if include_multilingual and not accept_source_licenses:
        raise LexicalDataLicenseError(
            "Open Multilingual Wordnet has per-component licenses. Review "
            "`word_forge sources list`, then pass accept_source_licenses=True "
            "or use --accept-source-licenses."
        )

    global _initialized
    if _initialized and not get_missing_nltk_resources(
        include_multilingual=include_multilingual
    ):
        if logger:
            logger.info("NLTK resources already initialized; nothing to download.")
        return []

    downloaded: List[str] = []
    for resource in _selected_resources(include_multilingual):
        if not _resource_available(resource):
            nltk.download(resource.package, quiet=True)  # type: ignore
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


def get_missing_nltk_resources(*, include_multilingual: bool = False) -> List[str]:
    """Return package names for selected NLTK resources not installed locally."""

    missing: List[str] = []
    for resource in _selected_resources(include_multilingual):
        if not _resource_available(resource):
            missing.append(resource.package)
    return missing


def _selected_resources(include_multilingual: bool) -> tuple[_NLTKResource, ...]:
    """Return core resources plus explicitly selected optional datasets."""

    if include_multilingual:
        return _CORE_NLTK_RESOURCES + _MULTILINGUAL_NLTK_RESOURCES
    return _CORE_NLTK_RESOURCES


__all__ = [
    "LexicalDataLicenseError",
    "ensure_nltk_data",
    "get_missing_nltk_resources",
]
