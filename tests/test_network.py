"""Tests for bounded graph snapshots used by the live GUI."""

from pathlib import Path

import pytest

from word_forge import Lexicon


def test_network_contains_semantic_and_translation_edges(tmp_path: Path) -> None:
    with Lexicon(tmp_path / "network.sqlite") as lexicon:
        bird = lexicon.add("bird", definition="A feathered animal.")
        sparrow = lexicon.add("sparrow", definition="A small bird.")
        lexicon.relate("sparrow", "bird", kind="is_a")
        lexicon.translate(
            "bird",
            source_language="en",
            target_language="es",
            target_term="ave",
        )

        snapshot = lexicon.network(limit=20)
        nodes = {(node.term, node.language): node for node in snapshot.nodes}
        edge_kinds = {edge.kind for edge in snapshot.edges}

        assert {("bird", "en"), ("sparrow", "en"), ("ave", "es")} <= set(nodes)
        assert {"is_a", "translation"} <= edge_kinds
        assert nodes[("bird", "en")].id == bird.id
        assert nodes[("sparrow", "en")].id == sparrow.id
        assert nodes[("bird", "en")].weight >= 2


def test_network_query_and_language_filter_are_bounded(tmp_path: Path) -> None:
    with Lexicon(tmp_path / "network.sqlite") as lexicon:
        for index in range(8):
            lexicon.add(f"term-{index}", definition="shared definition")
        lexicon.add("casa", language="es", definition="hogar")

        snapshot = lexicon.network(query="shared", language="en", limit=3)
        assert len(snapshot.nodes) == 3
        assert snapshot.truncated is True
        assert {node.language for node in snapshot.nodes} == {"en"}


def test_network_limit_validation(tmp_path: Path) -> None:
    with Lexicon(tmp_path / "network.sqlite") as lexicon:
        with pytest.raises(ValueError, match="between 1 and 2000"):
            lexicon.network(limit=0)
