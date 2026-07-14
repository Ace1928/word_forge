"""Tests for NLTK data management using real downloads."""

from __future__ import annotations

import nltk
import pytest

from word_forge.utils.nltk_utils import LexicalDataLicenseError, ensure_nltk_data


def test_ensure_nltk_data_runs() -> None:
    downloaded = ensure_nltk_data()
    assert isinstance(downloaded, list)


def test_ensure_nltk_data_supports_parser_operations() -> None:
    ensure_nltk_data()

    sentences = nltk.sent_tokenize("Word Forge works. Bootstrap is reliable.")
    tagged = nltk.pos_tag(nltk.word_tokenize(sentences[0]))
    chunked = nltk.ne_chunk(tagged)

    assert len(sentences) == 2
    assert tagged
    assert len(chunked) > 0


def test_multilingual_download_requires_explicit_license_acknowledgement() -> None:
    with pytest.raises(LexicalDataLicenseError, match="per-component licenses"):
        ensure_nltk_data(include_multilingual=True)
