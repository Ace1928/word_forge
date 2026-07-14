"""Tests for explicit multilingual WordNet routing and provenance."""

from __future__ import annotations

from pathlib import Path

import pytest

from word_forge.parser.lexical_functions import (
    create_lexical_dataset,
    get_wordnet_data,
)
from word_forge.parser.wordnet_languages import (
    MultilingualWordNetUnavailableError,
    UnsupportedWordNetLanguageError,
    multilingual_wordnet_available,
    resolve_wordnet_language,
)
from word_forge.utils.nltk_utils import ensure_nltk_data


def test_bcp47_language_is_mapped_to_wordnet_iso_639_3() -> None:
    french = resolve_wordnet_language("fr_FR", require_available=False)
    mandarin = resolve_wordnet_language("zh-Hant-TW", require_available=False)

    assert french.bcp47 == "fr-FR"
    assert french.nltk_code == "fra"
    assert french.source_id == "open-multilingual-wordnet"
    assert mandarin.nltk_code == "cmn"


def test_english_wordnet_is_core_and_provenanced() -> None:
    ensure_nltk_data()

    entries = get_wordnet_data("dog", "en-AU")

    assert entries
    assert entries[0]["language"] == "en-AU"
    assert entries[0]["source"] == "princeton-wordnet"
    assert entries[0]["synset_id"]
    assert entries[0]["definition_language"] == "en"


def test_unsupported_wordnet_language_is_explicit() -> None:
    with pytest.raises(UnsupportedWordNetLanguageError, match="not bundled"):
        resolve_wordnet_language("vi", require_available=False)


def test_missing_multilingual_data_has_actionable_setup_command() -> None:
    if multilingual_wordnet_available():
        pytest.skip("Open Multilingual Wordnet is installed in this environment")

    with pytest.raises(
        MultilingualWordNetUnavailableError,
        match="--accept-source-licenses",
    ):
        resolve_wordnet_language("fr")


def test_dataset_degrades_to_structured_warning_for_unavailable_language(
    tmp_path: Path,
) -> None:
    missing = str(tmp_path / "missing")

    dataset = create_lexical_dataset(
        "xin_chào",
        language="vi",
        openthesaurus_path=missing,
        odict_path=missing,
        dbnary_path=missing,
        opendict_path=missing,
        thesaurus_path=missing,
    )

    assert dataset["language"] == "vi"
    assert dataset["wordnet_data"] == []
    assert dataset["source_warnings"]
    assert "not bundled" in dataset["source_warnings"][0]


@pytest.mark.skipif(
    not multilingual_wordnet_available(),
    reason="Optional license-gated Open Multilingual Wordnet is not installed",
)
def test_installed_multilingual_wordnet_returns_real_french_lemmas() -> None:
    entries = get_wordnet_data("chat", "fr")

    assert entries
    assert all(entry["source"] == "open-multilingual-wordnet" for entry in entries)
    assert any("chat" in entry["synonyms"] for entry in entries)
