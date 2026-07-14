"""Tests for Unicode grapheme and source-backed pronunciation primitives."""

from __future__ import annotations

import pytest

from word_forge.parser.linguistics import (
    LanguageTagError,
    arpabet_to_ipa,
    canonicalize_language_tag,
    infer_script,
    lookup_pronunciations,
    normalize_term,
    segment_graphemes,
)
from word_forge.utils.nltk_utils import ensure_nltk_data


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("EN_us", "en-US"),
        ("sr-latn-rs", "sr-Latn-RS"),
        ("zh-Hant-TW", "zh-Hant-TW"),
        ("en-u-CA-gregory", "en-u-ca-gregory"),
        ("x-word-forge", "x-word-forge"),
        ("und", "und"),
    ],
)
def test_canonicalize_language_tag(raw: str, expected: str) -> None:
    assert canonicalize_language_tag(raw) == expected


@pytest.mark.parametrize(
    "value",
    ["", "e", "en--US", "en-@", "en-abcde-xy", "en-u", "en-u-ca-u-nu"],
)
def test_invalid_language_tag_is_rejected(value: str) -> None:
    with pytest.raises(LanguageTagError):
        canonicalize_language_tag(value)


def test_normalize_term_uses_nfkc_and_casefold() -> None:
    assert normalize_term("  ＷＯＲＤ  ") == "word"
    assert normalize_term("Straße") == "strasse"


def test_extended_grapheme_segmentation_preserves_user_characters() -> None:
    graphemes = segment_graphemes("e\u0301👨‍👩‍👧‍👦🇦🇺")

    assert [record.text for record in graphemes] == [
        "e\u0301",
        "👨‍👩‍👧‍👦",
        "🇦🇺",
    ]
    assert graphemes[0].normalized == "é"
    assert graphemes[0].codepoints == ("U+0065", "U+0301")
    assert graphemes[0].script == "Latn"


@pytest.mark.parametrize(
    ("text", "script"),
    [
        ("Word", "Latn"),
        ("Москва", "Cyrl"),
        ("Ελλάδα", "Grek"),
        ("日本", "Hani"),
        ("かな", "Hira"),
        ("한글", "Hang"),
        ("123!", "Zyyy"),
        ("", "Zzzz"),
    ],
)
def test_infer_script(text: str, script: str) -> None:
    assert infer_script(text) == script


def test_arpabet_conversion_preserves_stress() -> None:
    assert arpabet_to_ipa(("K", "AE1", "T")) == ("k", "ˈæ", "t")
    assert arpabet_to_ipa(("AH0", "B", "AW1", "T")) == (
        "ə",
        "b",
        "ˈaʊ",
        "t",
    )


def test_lookup_pronunciations_returns_arpabet_and_derived_ipa() -> None:
    ensure_nltk_data()

    pronunciations = lookup_pronunciations("cat", "en-US")

    by_notation = {item.notation: item for item in pronunciations}
    assert {"arpabet", "ipa"} <= by_notation.keys()
    assert by_notation["arpabet"].text == "K AE1 T"
    assert by_notation["arpabet"].stress_pattern == (1,)
    assert by_notation["arpabet"].syllable_count == 1
    assert by_notation["arpabet"].generated is False
    assert by_notation["ipa"].text == "k ˈæ t"
    assert by_notation["ipa"].generated is True


def test_lookup_pronunciations_does_not_claim_non_english_coverage() -> None:
    assert lookup_pronunciations("chat", "fr") == ()
