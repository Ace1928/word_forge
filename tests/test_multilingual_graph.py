"""Integration tests for language-aware, source-preserving graph identity."""

from __future__ import annotations

from pathlib import Path

import pytest

from word_forge.database.database_manager import DBManager
from word_forge.emotion.emotion_manager import EmotionManager
from word_forge.exceptions import AmbiguousTermError
from word_forge.graph.graph_manager import GraphManager


def test_homographs_resolve_by_language_and_edges_target_correct_nodes(
    tmp_path: Path,
) -> None:
    database = DBManager(db_path=tmp_path / "homographs.sqlite")
    english_chat = database.insert_or_update_word("chat", language="en")
    french_chat = database.insert_or_update_word("chat", language="fr")
    conversation = database.insert_or_update_word("conversation", language="en")
    discussion = database.insert_or_update_word("discussion", language="fr")
    database.insert_relationship(
        "chat",
        "conversation",
        "synonym",
        base_language="en",
        related_language="en",
        source="english-test",
    )
    database.insert_relationship(
        "chat",
        "discussion",
        "synonym",
        base_language="fr",
        related_language="fr",
        source="french-test",
    )
    database.insert_relationship(
        "chat",
        "chat",
        "translation",
        base_language="en",
        related_language="fr",
        source="translation-test",
    )

    manager = GraphManager(db_manager=database)
    manager.build_graph()

    assert manager.get_node_id("CHAT", "en") == english_chat
    assert manager.get_node_id("chat", "fr") == french_chat
    with pytest.raises(AmbiguousTermError, match="provide language"):
        manager.get_node_id("chat")

    assert manager.g.nodes[english_chat]["label"] == "chat [en]"
    assert manager.g.nodes[french_chat]["label"] == "chat [fr]"
    assert manager.g.has_edge(english_chat, conversation)
    assert manager.g.has_edge(french_chat, discussion)
    assert manager.g.has_edge(english_chat, french_chat)
    assert not manager.g.has_edge(english_chat, discussion)
    assert not manager.g.has_edge(french_chat, conversation)


def test_parallel_source_assertions_are_preserved_on_one_connection(
    tmp_path: Path,
) -> None:
    database = DBManager(db_path=tmp_path / "assertions.sqlite")
    alpha = database.insert_or_update_word("alpha")
    beta = database.insert_or_update_word("beta")
    database.insert_relationship("alpha", "beta", "synonym", source="source-a")
    database.insert_relationship("alpha", "beta", "synonym", source="source-b")
    database.insert_relationship("alpha", "beta", "related", source="source-a")

    manager = GraphManager(db_manager=database)
    manager.build_graph()

    edge = manager.g.get_edge_data(alpha, beta)
    assert edge is not None
    assert manager.get_edge_count() == 1
    assert edge["assertion_count"] == 3
    assert set(edge["relationship_types"].split("|")) == {"related", "synonym"}
    assert set(edge["sources"].split("|")) == {"source-a", "source-b"}
    synonym_assertions = manager.get_relationships_by_dimension(
        "lexical", rel_type="synonym"
    )
    assert len(synonym_assertions) == 2
    assert {item[3]["source"] for item in synonym_assertions} == {
        "source-a",
        "source-b",
    }


def test_relationship_only_updates_advance_graph_incrementally(tmp_path: Path) -> None:
    database = DBManager(db_path=tmp_path / "incremental.sqlite")
    alpha = database.insert_or_update_word("alpha")
    beta = database.insert_or_update_word("beta")
    manager = GraphManager(db_manager=database)
    manager.build_graph()

    assert database.insert_relationship("alpha", "beta", "synonym", source="source-a")
    assert manager.update_graph() == 0
    first_metrics = manager.get_last_update_metrics()
    assert first_metrics.new_edges == 1
    assert first_metrics.new_relationships == 1
    assert manager.g.has_edge(alpha, beta)

    assert database.insert_relationship("alpha", "beta", "synonym", source="source-b")
    assert manager.update_graph() == 0
    second_metrics = manager.get_last_update_metrics()
    assert second_metrics.new_edges == 0
    assert second_metrics.new_relationships == 1
    assert manager.g.edges[alpha, beta]["assertion_count"] == 2


def test_language_identity_and_assertions_survive_gexf_round_trip(
    tmp_path: Path,
) -> None:
    database = DBManager(db_path=tmp_path / "roundtrip.sqlite")
    english = database.insert_or_update_word("chat", language="en")
    french = database.insert_or_update_word("chat", language="fr")
    database.insert_relationship(
        "chat",
        "chat",
        "translation",
        base_language="en",
        related_language="fr",
        source="source-a",
    )
    database.insert_relationship(
        "chat",
        "chat",
        "translation",
        base_language="en",
        related_language="fr",
        source="source-b",
    )
    manager = GraphManager(db_manager=database)
    manager.build_graph()
    output_path = tmp_path / "multilingual.gexf"
    manager.save_to_gexf(str(output_path))

    loaded = GraphManager(db_manager=database)
    loaded.load_from_gexf(str(output_path))

    assert loaded.get_node_id("chat", "en") == english
    assert loaded.get_node_id("chat", "fr") == french
    assert loaded.g.edges[english, french]["assertion_count"] == 2
    with pytest.raises(AmbiguousTermError):
        loaded.get_node_id("chat")


def test_opposite_assertion_directions_survive_one_undirected_edge(
    tmp_path: Path,
) -> None:
    database = DBManager(db_path=tmp_path / "direction.sqlite")
    alpha = database.insert_or_update_word("alpha")
    beta = database.insert_or_update_word("beta")
    database.insert_relationship("alpha", "beta", "broader", source="source-a")
    database.insert_relationship("beta", "alpha", "broader", source="source-a")

    manager = GraphManager(db_manager=database)
    manager.build_graph()

    assert manager.get_edge_count() == 1
    assert manager.g.edges[alpha, beta]["assertion_count"] == 2
    relationships = manager.get_relationships_by_dimension(
        "lexical", rel_type="broader"
    )
    assert {(source, target) for source, target, _, _ in relationships} == {
        ("alpha", "beta"),
        ("beta", "alpha"),
    }

    output_path = tmp_path / "direction.gexf"
    manager.save_to_gexf(str(output_path))
    loaded = GraphManager(db_manager=database)
    loaded.load_from_gexf(str(output_path))
    loaded_relationships = loaded.get_relationships_by_dimension(
        "lexical", rel_type="broader"
    )
    assert {(source, target) for source, target, _, _ in loaded_relationships} == {
        ("alpha", "beta"),
        ("beta", "alpha"),
    }
    assert loaded._relationship_counts["broader"] == 2


def test_graph_api_preserves_assertions_and_protects_identity(tmp_path: Path) -> None:
    manager = GraphManager(DBManager(db_path=tmp_path / "graph-api.sqlite"))
    alpha = manager.add_word_node("alpha", language="en")
    beta = manager.add_word_node("beta", language="en")

    assert manager.add_relationship(alpha, beta, "synonym", source="source-a")
    assert manager.add_relationship(alpha, beta, "related", source="source-b")
    edge = manager.g.edges[alpha, beta]
    assert edge["assertion_count"] == 2
    assert set(edge["sources"].split("|")) == {"source-a", "source-b"}

    with pytest.raises(ValueError, match="lexical identity"):
        manager.add_word_node("alpha", {"language": "fr"}, language="en")
    with pytest.raises(ValueError, match="confidence must be finite"):
        manager.add_relationship(
            alpha,
            beta,
            "antonym",
            source="source-c",
            confidence=float("nan"),
        )
    assert manager.g.edges[alpha, beta]["assertion_count"] == 2


def test_visualizer_filters_aggregated_dimensions(tmp_path: Path) -> None:
    manager = GraphManager(DBManager(db_path=tmp_path / "dimensions.sqlite"))
    alpha = manager.add_word_node("alpha")
    beta = manager.add_word_node("beta")
    manager.add_relationship(alpha, beta, "synonym", source="lexical-source")
    manager.add_relationship(
        alpha,
        beta,
        "joy_associated",
        dimension="emotional",
        source="emotion-source",
        valence=0.8,
        arousal=0.7,
    )

    assert manager.visualizer._filter_graph_by_dimensions(["lexical"]).has_edge(
        alpha, beta
    )
    assert manager.visualizer._filter_graph_by_dimensions(["emotional"]).has_edge(
        alpha, beta
    )
    assert not manager.visualizer._filter_graph_by_dimensions(["contextual"]).has_edge(
        alpha, beta
    )
    assert manager.visualizer._filter_graph_by_dimensions([]).number_of_nodes() == 0


def test_assertion_serialization_is_independent_of_database_row_order(
    tmp_path: Path,
) -> None:
    serialized_edges = []
    for database_name, sources in (
        ("forward.sqlite", ("source-a", "source-b")),
        ("reverse.sqlite", ("source-b", "source-a")),
    ):
        database = DBManager(db_path=tmp_path / database_name)
        alpha = database.insert_or_update_word("alpha")
        beta = database.insert_or_update_word("beta")
        for source in sources:
            database.insert_relationship("alpha", "beta", "synonym", source=source)
        manager = GraphManager(database)
        manager.build_graph()
        serialized_edges.append(manager.g.edges[alpha, beta]["assertions_json"])

    assert serialized_edges[0] == serialized_edges[1]


def test_incremental_update_refreshes_node_metadata(tmp_path: Path) -> None:
    database = DBManager(db_path=tmp_path / "node-refresh.sqlite")
    word_id = database.insert_or_update_word("lexeme", source="seed", is_stub=True)
    manager = GraphManager(database)
    manager.build_graph()

    database.insert_or_update_word("lexeme", source="kaikki-wiktionary", is_stub=False)
    assert manager.update_graph() == 0

    metrics = manager.get_last_update_metrics()
    assert metrics.updated_nodes == 1
    assert manager.g.nodes[word_id]["source"] == "kaikki-wiktionary"
    assert manager.g.nodes[word_id]["is_stub"] is False


def test_incremental_emotional_assertion_replaces_stale_values(
    tmp_path: Path,
) -> None:
    database = DBManager(db_path=tmp_path / "emotion-refresh.sqlite")
    alpha = database.insert_or_update_word("alpha")
    emotions = EmotionManager(database)
    emotions.set_word_emotion(alpha, 0.8, 0.7)
    manager = GraphManager(database)
    manager.build_graph()
    joy = database.get_word_id("joy")
    assert joy is not None

    emotions.set_word_emotion(alpha, 0.6, 0.8)
    assert manager.update_graph() == 0

    edge = manager.g.edges[alpha, joy]
    metrics = manager.get_last_update_metrics()
    assert metrics.new_relationships == 0
    assert metrics.updated_relationships == 1
    assert edge["assertion_count"] == 1
    assert edge["valence"] == pytest.approx(0.6)
    assert edge["arousal"] == pytest.approx(0.8)

    assert manager.update_graph() == 0
    stable_metrics = manager.get_last_update_metrics()
    assert stable_metrics.new_relationships == 0
    assert stable_metrics.updated_relationships == 0


def test_visualization_escapes_untrusted_lexical_metadata(tmp_path: Path) -> None:
    manager = GraphManager(DBManager(db_path=tmp_path / "safe-visualization.sqlite"))
    manager.add_word_node(
        "<script>alert(1)</script>",
        {"source": '<img src=x onerror="alert(2)">'},
    )
    output_path = tmp_path / "safe.html"

    manager.visualize(output_path=str(output_path), open_in_browser=False)

    rendered = output_path.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in rendered
    assert '<img src=x onerror="alert(2)">' not in rendered
    assert r"\u0026lt;script\u0026gt;alert(1)" in rendered
    assert r"\u0026lt;img src=x onerror=\u0026quot;alert(2)" in rendered
    assert "cdnjs.cloudflare.com" not in rendered
    assert "lib/bindings" not in rendered
    assert 'data-word-forge-viewer="1"' in rendered
    assert 'id="wf-search"' in rendered
    assert 'id="wf-language"' in rendered
    assert 'id="wf-relationship"' in rendered
    assert "Content-Security-Policy" in rendered


def test_visualizer_selects_language_qualified_bounded_neighborhood(
    tmp_path: Path,
) -> None:
    manager = GraphManager(DBManager(db_path=tmp_path / "bounded-view.sqlite"))
    english_chat = manager.add_word_node("chat", language="en")
    french_chat = manager.add_word_node("chat", language="fr")
    hello = manager.add_word_node("hello", language="en")
    bonjour = manager.add_word_node("bonjour", language="fr")
    distant = manager.add_word_node("lointain", language="fr")
    manager.add_relationship(english_chat, hello, "related", source="english")
    manager.add_relationship(french_chat, bonjour, "related", source="french")
    manager.add_relationship(bonjour, distant, "related", source="french")

    view = manager.visualizer._select_graph(
        ["lexical"],
        focus_term="chat",
        focus_language="fr",
        depth=1,
        max_nodes=10,
        max_edges=10,
    )
    edge_limited = manager.visualizer._select_graph(
        ["lexical"],
        focus_term="chat",
        focus_language="fr",
        depth=2,
        max_nodes=10,
        max_edges=1,
    )

    assert set(view.nodes()) == {french_chat, bonjour}
    assert english_chat not in view
    assert hello not in view
    assert distant not in view
    assert edge_limited.number_of_edges() == 1
    assert edge_limited.has_edge(french_chat, bonjour)
