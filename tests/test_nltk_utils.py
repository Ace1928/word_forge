"""Tests for NLTK data management using real downloads."""

from __future__ import annotations

import nltk

from word_forge.utils.nltk_utils import ensure_nltk_data


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
