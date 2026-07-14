"""Tests for deterministic embedding-model contracts."""

from __future__ import annotations

import pytest

from word_forge.vectorizer.embedding_models import (
    DEFAULT_EMBEDDING_MODEL,
    MAX_COLLECTION_NAME_LENGTH,
    collection_name_for_model,
    format_embedding_text,
    get_embedding_model_spec,
)


def test_portable_model_metadata_is_multilingual_and_small() -> None:
    spec = get_embedding_model_spec(DEFAULT_EMBEDDING_MODEL)

    assert spec is not None
    assert spec.dimension == 384
    assert spec.prompt_style == "e5"
    assert spec.license_name == "MIT"


@pytest.mark.parametrize("text", ["hello", "南瓜", "مرحبا", "გამარჯობა"])
def test_standard_e5_uses_query_and_passage_prefixes(text: str) -> None:
    assert (
        format_embedding_text(DEFAULT_EMBEDDING_MODEL, text, is_query=True)
        == f"query: {text}"
    )
    assert (
        format_embedding_text(DEFAULT_EMBEDDING_MODEL, text, is_query=False)
        == f"passage: {text}"
    )


def test_standard_e5_does_not_duplicate_existing_prefix() -> None:
    assert (
        format_embedding_text(
            DEFAULT_EMBEDDING_MODEL, "query: recursion", is_query=True
        )
        == "query: recursion"
    )


def test_e5_instruct_formats_only_queries() -> None:
    model_name = "intfloat/multilingual-e5-large-instruct"

    assert (
        format_embedding_text(
            model_name,
            "recursion",
            is_query=True,
            task="Retrieve matching lexical definitions",
        )
        == "Instruct: Retrieve matching lexical definitions\nQuery: recursion"
    )
    assert (
        format_embedding_text(model_name, "A recursive definition", is_query=False)
        == "A recursive definition"
    )


def test_unknown_model_receives_no_unverified_prompt() -> None:
    assert (
        format_embedding_text("acme/custom-embedder", "  raw text  ", is_query=True)
        == "raw text"
    )


def test_collection_names_are_stable_bounded_and_model_specific() -> None:
    first = collection_name_for_model(DEFAULT_EMBEDDING_MODEL)
    repeated = collection_name_for_model(DEFAULT_EMBEDDING_MODEL)
    other = collection_name_for_model("sentence-transformers/all-MiniLM-L6-v2")

    assert first == repeated
    assert first != other
    assert first.startswith("wf_")
    assert len(first) <= MAX_COLLECTION_NAME_LENGTH


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_model_names_are_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="model_name"):
        collection_name_for_model(value)
