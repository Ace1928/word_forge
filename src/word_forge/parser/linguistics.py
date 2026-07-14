"""Unicode orthography and pronunciation primitives for lexical records.

Graphemes and phonemes are deliberately represented separately. Extended
grapheme clusters describe written user-perceived characters, while phoneme
records describe source-backed or explicitly derived pronunciations.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Literal, Mapping, Optional, Tuple

import regex  # type: ignore[import-untyped]

PhoneticNotation = Literal["arpabet", "ipa"]


class LanguageTagError(ValueError):
    """Raised when a language tag is not structurally well formed."""


_PRIVATE_USE_TAG = re.compile(r"^[xX](?:-[A-Za-z0-9]{1,8})+$")
_GRANDFATHERED_TAGS = frozenset(
    {
        "art-lojban",
        "cel-gaulish",
        "en-gb-oed",
        "i-ami",
        "i-bnn",
        "i-default",
        "i-enochian",
        "i-hak",
        "i-klingon",
        "i-lux",
        "i-mingo",
        "i-navajo",
        "i-pwn",
        "i-tao",
        "i-tay",
        "i-tsu",
        "no-bok",
        "no-nyn",
        "sgn-be-fr",
        "sgn-be-nl",
        "sgn-ch-de",
        "zh-guoyu",
        "zh-hakka",
        "zh-min",
        "zh-min-nan",
        "zh-xiang",
    }
)

_SCRIPT_PATTERNS: Tuple[Tuple[regex.Pattern[str], str], ...] = tuple(
    (regex.compile(rf"\A\p{{Script={unicode_name}}}\Z"), iso_code)
    for unicode_name, iso_code in (
        ("Latin", "Latn"),
        ("Cyrillic", "Cyrl"),
        ("Greek", "Grek"),
        ("Arabic", "Arab"),
        ("Hebrew", "Hebr"),
        ("Devanagari", "Deva"),
        ("Bengali", "Beng"),
        ("Gurmukhi", "Guru"),
        ("Gujarati", "Gujr"),
        ("Oriya", "Orya"),
        ("Tamil", "Taml"),
        ("Telugu", "Telu"),
        ("Kannada", "Knda"),
        ("Malayalam", "Mlym"),
        ("Sinhala", "Sinh"),
        ("Thai", "Thai"),
        ("Lao", "Laoo"),
        ("Myanmar", "Mymr"),
        ("Georgian", "Geor"),
        ("Armenian", "Armn"),
        ("Ethiopic", "Ethi"),
        ("Han", "Hani"),
        ("Hiragana", "Hira"),
        ("Katakana", "Kana"),
        ("Hangul", "Hang"),
    )
)


@dataclass(frozen=True, slots=True)
class Grapheme:
    """One Unicode extended grapheme cluster and its scalar metadata."""

    position: int
    text: str
    normalized: str
    codepoints: Tuple[str, ...]
    unicode_names: Tuple[str, ...]
    categories: Tuple[str, ...]
    combining_classes: Tuple[int, ...]
    script: str

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable grapheme record."""

        return {
            "position": self.position,
            "text": self.text,
            "normalized": self.normalized,
            "codepoints": list(self.codepoints),
            "unicode_names": list(self.unicode_names),
            "categories": list(self.categories),
            "combining_classes": list(self.combining_classes),
            "script": self.script,
        }


@dataclass(frozen=True, slots=True)
class Phoneme:
    """One phonetic segment with optional lexical stress information."""

    position: int
    symbol: str
    base_symbol: str
    stress: Optional[int]
    syllabic: bool

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable phoneme record."""

        return {
            "position": self.position,
            "symbol": self.symbol,
            "base_symbol": self.base_symbol,
            "stress": self.stress,
            "syllabic": self.syllabic,
        }


@dataclass(frozen=True, slots=True)
class Pronunciation:
    """A pronunciation from an identified source and notation system."""

    notation: PhoneticNotation
    phonemes: Tuple[Phoneme, ...]
    language: str
    dialect: Optional[str]
    source: str
    confidence: float = 1.0
    generated: bool = False

    @property
    def text(self) -> str:
        """Return the conventional space-delimited transcription."""

        return " ".join(phoneme.symbol for phoneme in self.phonemes)

    @property
    def syllable_count(self) -> int:
        """Return the number of explicitly syllabic segments."""

        return sum(phoneme.syllabic for phoneme in self.phonemes)

    @property
    def stress_pattern(self) -> Tuple[int, ...]:
        """Return stress values in syllable order."""

        return tuple(
            phoneme.stress
            for phoneme in self.phonemes
            if phoneme.syllabic and phoneme.stress is not None
        )

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable pronunciation record."""

        return {
            "notation": self.notation,
            "text": self.text,
            "phonemes": [phoneme.to_dict() for phoneme in self.phonemes],
            "language": self.language,
            "dialect": self.dialect,
            "source": self.source,
            "confidence": self.confidence,
            "generated": self.generated,
            "syllable_count": self.syllable_count,
            "stress_pattern": list(self.stress_pattern),
        }


def canonicalize_language_tag(language: str) -> str:
    """Validate and apply conventional BCP 47 subtag casing.

    This validates structural well-formedness without consulting the mutable
    IANA language-subtag registry. Registry-level validation can therefore be
    layered on later without changing stored tag casing.
    """

    if not isinstance(language, str) or not language.strip():
        raise LanguageTagError("language must be a non-empty BCP 47 tag")
    value = language.strip().replace("_", "-")
    if _PRIVATE_USE_TAG.fullmatch(value):
        return value.lower()
    if value.casefold() in _GRANDFATHERED_TAGS:
        return value.lower()

    subtags = value.split("-")
    primary = subtags[0]
    if not primary.isalpha() or not 2 <= len(primary) <= 8:
        raise LanguageTagError(f"Invalid primary language subtag: {primary!r}")

    canonical = [primary.lower()]
    index = 1
    if 2 <= len(primary) <= 3:
        extlang_count = 0
        while (
            index < len(subtags)
            and len(subtags[index]) == 3
            and subtags[index].isalpha()
            and extlang_count < 3
        ):
            canonical.append(subtags[index].lower())
            index += 1
            extlang_count += 1

    if index < len(subtags) and len(subtags[index]) == 4 and subtags[index].isalpha():
        canonical.append(subtags[index].title())
        index += 1

    if index < len(subtags) and (
        (len(subtags[index]) == 2 and subtags[index].isalpha())
        or (len(subtags[index]) == 3 and subtags[index].isdigit())
    ):
        canonical.append(subtags[index].upper())
        index += 1

    while index < len(subtags) and _is_variant_subtag(subtags[index]):
        canonical.append(subtags[index].lower())
        index += 1

    extension_singletons: set[str] = set()
    while index < len(subtags) and _is_extension_singleton(subtags[index]):
        singleton = subtags[index].lower()
        if singleton in extension_singletons:
            raise LanguageTagError(
                f"Repeated extension singleton {singleton!r} in {language!r}"
            )
        extension_singletons.add(singleton)
        canonical.append(singleton)
        index += 1
        extension_start = index
        while index < len(subtags) and _is_extension_value(subtags[index]):
            canonical.append(subtags[index].lower())
            index += 1
        if index == extension_start:
            raise LanguageTagError(
                f"Extension {singleton!r} has no value in {language!r}"
            )

    if index < len(subtags) and subtags[index].casefold() == "x":
        canonical.append("x")
        index += 1
        private_start = index
        while index < len(subtags) and _is_private_value(subtags[index]):
            canonical.append(subtags[index].lower())
            index += 1
        if index == private_start:
            raise LanguageTagError(f"Private-use sequence is empty in {language!r}")

    if index != len(subtags):
        raise LanguageTagError(f"Invalid language tag: {language!r}")
    return "-".join(canonical)


def normalize_term(term: str) -> str:
    """Return the Unicode NFKC/case-folded key used for lexical identity."""

    if not isinstance(term, str) or not term.strip():
        raise ValueError("term must be a non-empty string")
    return unicodedata.normalize("NFKC", term.strip()).casefold()


def segment_graphemes(text: str) -> Tuple[Grapheme, ...]:
    """Segment text into Unicode extended grapheme clusters (UAX #29)."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    records: List[Grapheme] = []
    for position, cluster in enumerate(regex.findall(r"\X", text)):
        records.append(
            Grapheme(
                position=position,
                text=cluster,
                normalized=unicodedata.normalize("NFC", cluster),
                codepoints=tuple(f"U+{ord(char):04X}" for char in cluster),
                unicode_names=tuple(
                    unicodedata.name(char, "UNNAMED") for char in cluster
                ),
                categories=tuple(unicodedata.category(char) for char in cluster),
                combining_classes=tuple(
                    unicodedata.combining(char) for char in cluster
                ),
                script=_cluster_script(cluster),
            )
        )
    return tuple(records)


def infer_script(text: str) -> str:
    """Return the predominant ISO 15924 script code in ``text``."""

    counts: Dict[str, int] = {}
    for grapheme in segment_graphemes(text):
        if grapheme.script not in {"Zyyy", "Zinh", "Zzzz"}:
            counts[grapheme.script] = counts.get(grapheme.script, 0) + 1
    if not counts:
        return "Zyyy" if text else "Zzzz"
    return min(counts, key=lambda script: (-counts[script], script))


_ARPABET_VOWELS = frozenset(
    {
        "AA",
        "AE",
        "AH",
        "AO",
        "AW",
        "AY",
        "EH",
        "ER",
        "EY",
        "IH",
        "IY",
        "OW",
        "OY",
        "UH",
        "UW",
    }
)

_ARPABET_TO_IPA: Mapping[str, str] = {
    "AA": "ɑ",
    "AE": "æ",
    "AH": "ʌ",
    "AO": "ɔ",
    "AW": "aʊ",
    "AY": "aɪ",
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "EH": "ɛ",
    "ER": "ɝ",
    "EY": "eɪ",
    "F": "f",
    "G": "ɡ",
    "HH": "h",
    "IH": "ɪ",
    "IY": "i",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "P": "p",
    "R": "ɹ",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "UH": "ʊ",
    "UW": "u",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
}


def arpabet_to_ipa(symbols: Iterable[str]) -> Tuple[str, ...]:
    """Convert CMUdict ARPABET symbols to an approximate US-English IPA form."""

    converted: List[str] = []
    for symbol in symbols:
        base, stress = _split_arpabet_symbol(symbol)
        if base not in _ARPABET_TO_IPA:
            raise ValueError(f"Unsupported ARPABET symbol: {symbol!r}")
        ipa = _ARPABET_TO_IPA[base]
        if base == "AH" and stress == 0:
            ipa = "ə"
        elif base == "ER" and stress == 0:
            ipa = "ɚ"
        if stress == 1:
            ipa = f"ˈ{ipa}"
        elif stress == 2:
            ipa = f"ˌ{ipa}"
        converted.append(ipa)
    return tuple(converted)


def lookup_pronunciations(term: str, language: str = "en") -> Tuple[Pronunciation, ...]:
    """Return source-backed pronunciations available for a lexical term.

    CMUdict is an American-English resource, so non-English requests return an
    empty tuple instead of fabricating language coverage.
    """

    canonical_language = canonicalize_language_tag(language)
    if canonical_language.split("-", 1)[0] != "en":
        return ()
    key = normalize_term(term)
    try:
        variants = _cmudict_entries().get(key, ())
    except LookupError:
        return ()

    pronunciations: List[Pronunciation] = []
    seen: set[Tuple[PhoneticNotation, str]] = set()
    for variant in variants:
        arpabet_phonemes = _arpabet_phonemes(variant)
        _append_unique_pronunciation(
            pronunciations,
            seen,
            Pronunciation(
                notation="arpabet",
                phonemes=arpabet_phonemes,
                language=canonical_language,
                dialect="en-US",
                source="cmudict",
            ),
        )
        ipa_symbols = arpabet_to_ipa(variant)
        ipa_phonemes = tuple(
            Phoneme(
                position=index,
                symbol=ipa_symbol,
                base_symbol=ipa_symbol.lstrip("ˈˌ"),
                stress=arpabet_phonemes[index].stress,
                syllabic=arpabet_phonemes[index].syllabic,
            )
            for index, ipa_symbol in enumerate(ipa_symbols)
        )
        _append_unique_pronunciation(
            pronunciations,
            seen,
            Pronunciation(
                notation="ipa",
                phonemes=ipa_phonemes,
                language=canonical_language,
                dialect="en-US",
                source="cmudict-derived",
                confidence=0.9,
                generated=True,
            ),
        )
    return tuple(pronunciations)


def _cluster_script(cluster: str) -> str:
    """Resolve the script of a grapheme cluster using its base scalar."""

    inherited_only = True
    for character in cluster:
        category = unicodedata.category(character)
        if category.startswith("M"):
            continue
        inherited_only = False
        for pattern, iso_code in _SCRIPT_PATTERNS:
            if pattern.fullmatch(character):
                return iso_code
    if inherited_only and cluster:
        return "Zinh"
    return "Zyyy" if cluster else "Zzzz"


def _is_variant_subtag(subtag: str) -> bool:
    """Return whether a subtag matches the BCP 47 variant production."""

    return subtag.isalnum() and (
        5 <= len(subtag) <= 8 or (len(subtag) == 4 and subtag[0].isdigit())
    )


def _is_extension_singleton(subtag: str) -> bool:
    """Return whether a subtag can introduce a BCP 47 extension."""

    return len(subtag) == 1 and subtag.isalnum() and subtag.casefold() != "x"


def _is_extension_value(subtag: str) -> bool:
    """Return whether a subtag can occur within an extension."""

    return 2 <= len(subtag) <= 8 and subtag.isalnum()


def _is_private_value(subtag: str) -> bool:
    """Return whether a subtag can occur within a private-use sequence."""

    return 1 <= len(subtag) <= 8 and subtag.isalnum()


def _split_arpabet_symbol(symbol: str) -> Tuple[str, Optional[int]]:
    """Split an ARPABET token into its base symbol and optional stress."""

    normalized = symbol.strip().upper()
    if normalized and normalized[-1] in "012":
        return normalized[:-1], int(normalized[-1])
    return normalized, None


def _arpabet_phonemes(symbols: Iterable[str]) -> Tuple[Phoneme, ...]:
    """Build structured ARPABET phonemes."""

    result = []
    for index, symbol in enumerate(symbols):
        base, stress = _split_arpabet_symbol(symbol)
        result.append(
            Phoneme(
                position=index,
                symbol=symbol,
                base_symbol=base,
                stress=stress,
                syllabic=base in _ARPABET_VOWELS,
            )
        )
    return tuple(result)


def _append_unique_pronunciation(
    output: List[Pronunciation],
    seen: set[Tuple[PhoneticNotation, str]],
    pronunciation: Pronunciation,
) -> None:
    """Append a pronunciation when its notation/text pair is new."""

    key = (pronunciation.notation, pronunciation.text)
    if key not in seen:
        seen.add(key)
        output.append(pronunciation)


@lru_cache(maxsize=1)
def _cmudict_entries() -> Mapping[str, List[List[str]]]:
    """Load CMUdict once and keep model-independent imports lazy."""

    from nltk.corpus import cmudict  # type: ignore

    return cmudict.dict()  # type: ignore[no-any-return]


__all__ = [
    "Grapheme",
    "LanguageTagError",
    "Phoneme",
    "PhoneticNotation",
    "Pronunciation",
    "arpabet_to_ipa",
    "canonicalize_language_tag",
    "infer_script",
    "lookup_pronunciations",
    "normalize_term",
    "segment_graphemes",
]
