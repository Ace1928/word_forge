"""BCP 47 to NLTK WordNet language routing and resource readiness."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Tuple

import nltk  # type: ignore[import-untyped]

from word_forge.parser.linguistics import canonicalize_language_tag


def select_multilingual_wordnet_package(nltk_version: str) -> str:
    """Return the OMW package expected by an NLTK release line.

    NLTK 3.10 changed its built-in WordNet loader from ``omw-1.4`` to
    ``omw-2.0``. Selecting from the installed library version keeps setup and
    readiness checks aligned with the loader instead of merely finding an
    incompatible corpus left behind by an earlier environment.
    """

    match = re.match(r"^(\d+)\.(\d+)", nltk_version)
    if match is None:
        # The long-established package is the safest compatibility fallback
        # for downstream builds that do not expose a PEP 440 release string.
        return "omw-1.4"
    release = (int(match.group(1)), int(match.group(2)))
    return "omw-2.0" if release >= (3, 10) else "omw-1.4"


MULTILINGUAL_WORDNET_PACKAGE = select_multilingual_wordnet_package(
    str(getattr(nltk, "__version__", ""))
)
MULTILINGUAL_SETUP_COMMAND = (
    "word_forge setup-nltk --multilingual --accept-source-licenses"
)

# NLTK's OMW reader uses ISO 639-3 identifiers. BCP 47 commonly uses the
# corresponding ISO 639-1 primary subtag, so the boundary is explicit rather
# than relying on locale heuristics or a mutable host database.
_PRIMARY_TO_WORDNET: Dict[str, str] = {
    "ar": "arb",
    "bg": "bul",
    "ca": "cat",
    "da": "dan",
    "el": "ell",
    "en": "eng",
    "es": "spa",
    "eu": "eus",
    "fi": "fin",
    "fr": "fra",
    "gl": "glg",
    "he": "heb",
    "hr": "hrv",
    "id": "ind",
    "is": "isl",
    "it": "ita",
    "ja": "jpn",
    "lt": "lit",
    "ms": "zsm",
    "nb": "nob",
    "nl": "nld",
    "nn": "nno",
    "no": "nob",
    "pl": "pol",
    "pt": "por",
    "ro": "ron",
    "sk": "slk",
    "sl": "slv",
    "sq": "als",
    "sv": "swe",
    "th": "tha",
    "zh": "cmn",
}


class WordNetLanguageError(ValueError):
    """Base error for unsupported or unavailable WordNet language data."""


class UnsupportedWordNetLanguageError(WordNetLanguageError):
    """Raised when bundled WordNet sources do not map a language tag."""


class MultilingualWordNetUnavailableError(WordNetLanguageError):
    """Raised when a mapped non-English language has no installed OMW data."""


@dataclass(frozen=True, slots=True)
class WordNetLanguage:
    """Resolved language identity for an NLTK WordNet lookup."""

    bcp47: str
    nltk_code: str
    source_id: str
    requires_multilingual_data: bool


def resolve_wordnet_language(
    language: str, *, require_available: bool = True
) -> WordNetLanguage:
    """Resolve a BCP 47 tag to NLTK's ISO 639-3 WordNet identifier."""

    canonical = canonicalize_language_tag(language)
    primary = canonical.split("-", 1)[0]
    nltk_code = _PRIMARY_TO_WORDNET.get(primary)
    if nltk_code is None:
        supported = ", ".join(supported_primary_languages())
        raise UnsupportedWordNetLanguageError(
            f"WordNet data is not bundled for language {canonical!r}. "
            f"Supported primary tags: {supported}"
        )

    requires_multilingual = nltk_code != "eng"
    if (
        require_available
        and requires_multilingual
        and not multilingual_wordnet_available()
    ):
        raise MultilingualWordNetUnavailableError(
            f"Language {canonical!r} requires optional Open Multilingual "
            f"Wordnet data. Review its component licenses, then run: "
            f"{MULTILINGUAL_SETUP_COMMAND}"
        )

    return WordNetLanguage(
        bcp47=canonical,
        nltk_code=nltk_code,
        source_id=(
            "open-multilingual-wordnet"
            if requires_multilingual
            else "princeton-wordnet"
        ),
        requires_multilingual_data=requires_multilingual,
    )


def multilingual_wordnet_available() -> bool:
    """Return whether NLTK can locate its compatible OMW corpus locally."""

    for candidate in (
        f"corpora/{MULTILINGUAL_WORDNET_PACKAGE}",
        f"corpora/{MULTILINGUAL_WORDNET_PACKAGE}.zip",
    ):
        try:
            nltk.data.find(candidate)
            return True
        except LookupError:
            continue
    return False


def supported_primary_languages() -> Tuple[str, ...]:
    """Return stable BCP 47 primary tags routable through WordNet/OMW."""

    return tuple(sorted(_PRIMARY_TO_WORDNET))


__all__ = [
    "MULTILINGUAL_SETUP_COMMAND",
    "MULTILINGUAL_WORDNET_PACKAGE",
    "MultilingualWordNetUnavailableError",
    "UnsupportedWordNetLanguageError",
    "WordNetLanguage",
    "WordNetLanguageError",
    "multilingual_wordnet_available",
    "resolve_wordnet_language",
    "select_multilingual_wordnet_package",
    "supported_primary_languages",
]
