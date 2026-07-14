"""Validation tests for provenance-preserving graph assertions."""

from __future__ import annotations

from typing import cast

import pytest

from word_forge.graph.graph_assertions import create_graph_assertion
from word_forge.graph.graph_config import RelationshipDimension


def test_create_graph_assertion_rejects_unsupported_dimension() -> None:
    with pytest.raises(ValueError, match="unsupported relationship dimension"):
        create_graph_assertion(
            1,
            2,
            "synonym",
            dimension=cast(RelationshipDimension, "invented"),
            source="test",
            confidence=1.0,
            related_language="en",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("relationship", 7, "relationship must be a string"),
        ("source", 7, "source must be a string"),
        ("related_language", 7, "related_language must be a string"),
    ],
)
def test_create_graph_assertion_rejects_non_text_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    values: dict[str, object] = {
        "relationship": "synonym",
        "source": "test",
        "related_language": "en",
    }
    values[field] = value

    with pytest.raises(TypeError, match=message):
        create_graph_assertion(
            1,
            2,
            cast(str, values["relationship"]),
            dimension="lexical",
            source=cast(str, values["source"]),
            confidence=1.0,
            related_language=cast(str, values["related_language"]),
        )
