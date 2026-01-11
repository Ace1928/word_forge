"""Tests for word_forge.graph.graph_query module.

This module provides comprehensive tests for the GraphQuery class
including node/edge queries, relationship lookups, subgraph extraction,
and graph information retrieval.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from word_forge.graph.graph_query import GraphQuery
from word_forge.exceptions import NodeNotFoundError


class TestGraphQueryInit:
    """Tests for GraphQuery initialization."""

    def test_init_with_manager(self) -> None:
        """Test GraphQuery initializes with manager reference."""
        mock_manager = MagicMock()
        mock_manager.config = MagicMock()

        query = GraphQuery(mock_manager)

        assert query.manager is mock_manager
        assert query.logger is not None
        assert query._config is mock_manager.config


class TestGraphQueryGetNodeId:
    """Tests for GraphQuery get_node_id method."""

    def test_get_node_id_found(self) -> None:
        """Test getting node ID for existing term."""
        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager._term_to_id = {"happiness": 1, "joy": 2}

        query = GraphQuery(mock_manager)
        result = query.get_node_id("happiness")

        assert result == 1

    def test_get_node_id_case_insensitive(self) -> None:
        """Test that node ID lookup is case-insensitive."""
        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager._term_to_id = {"happiness": 1}

        query = GraphQuery(mock_manager)

        assert query.get_node_id("HAPPINESS") == 1
        assert query.get_node_id("Happiness") == 1
        assert query.get_node_id("happiness") == 1

    def test_get_node_id_not_found(self) -> None:
        """Test getting node ID for non-existent term."""
        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager._term_to_id = {"happiness": 1}

        query = GraphQuery(mock_manager)
        result = query.get_node_id("nonexistent")

        assert result is None


class TestGraphQueryGetRelatedTerms:
    """Tests for GraphQuery get_related_terms method."""

    def test_get_related_terms_success(self) -> None:
        """Test getting related terms for existing term."""
        g = nx.DiGraph()
        g.add_node(1, term="happiness")
        g.add_node(2, term="joy")
        g.add_node(3, term="sadness")
        g.add_edge(1, 2, relationship="synonym")
        g.add_edge(1, 3, relationship="antonym")

        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager._term_to_id = {"happiness": 1, "joy": 2, "sadness": 3}
        mock_manager.g = g

        query = GraphQuery(mock_manager)
        related = query.get_related_terms("happiness")

        assert len(related) == 2
        assert "joy" in related
        assert "sadness" in related

    def test_get_related_terms_with_filter(self) -> None:
        """Test getting related terms with relationship type filter."""
        g = nx.DiGraph()
        g.add_node(1, term="happiness")
        g.add_node(2, term="joy")
        g.add_node(3, term="sadness")
        g.add_edge(1, 2, relationship="synonym")
        g.add_edge(1, 3, relationship="antonym")

        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager._term_to_id = {"happiness": 1, "joy": 2, "sadness": 3}
        mock_manager.g = g

        query = GraphQuery(mock_manager)
        related = query.get_related_terms("happiness", rel_type="synonym")

        assert len(related) == 1
        assert "joy" in related

    def test_get_related_terms_not_found(self) -> None:
        """Test getting related terms for non-existent term raises error."""
        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager._term_to_id = {}

        query = GraphQuery(mock_manager)

        with pytest.raises(NodeNotFoundError):
            query.get_related_terms("nonexistent")


class TestGraphQueryCounts:
    """Tests for GraphQuery count methods."""

    def test_get_node_count(self) -> None:
        """Test getting node count."""
        g = nx.DiGraph()
        g.add_node(1, term="happiness")
        g.add_node(2, term="joy")

        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager.g = g

        query = GraphQuery(mock_manager)
        count = query.get_node_count()

        assert count == 2

    def test_get_node_count_empty(self) -> None:
        """Test getting node count for empty graph."""
        g = nx.DiGraph()

        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager.g = g

        query = GraphQuery(mock_manager)
        count = query.get_node_count()

        assert count == 0

    def test_get_edge_count(self) -> None:
        """Test getting edge count."""
        g = nx.DiGraph()
        g.add_node(1, term="happiness")
        g.add_node(2, term="joy")
        g.add_edge(1, 2, relationship="synonym")

        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager.g = g

        query = GraphQuery(mock_manager)
        count = query.get_edge_count()

        assert count == 1

    def test_get_edge_count_empty(self) -> None:
        """Test getting edge count for graph without edges."""
        g = nx.DiGraph()
        g.add_node(1)

        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager.g = g

        query = GraphQuery(mock_manager)
        count = query.get_edge_count()

        assert count == 0


class TestGraphQueryGetTermById:
    """Tests for GraphQuery get_term_by_id method."""

    def test_get_term_by_id_found(self) -> None:
        """Test getting term by ID for existing node."""
        g = nx.DiGraph()
        g.add_node(1, term="happiness")

        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager.g = g

        query = GraphQuery(mock_manager)
        term = query.get_term_by_id(1)

        assert term == "happiness"

    def test_get_term_by_id_not_found(self) -> None:
        """Test getting term by ID for non-existent node."""
        g = nx.DiGraph()

        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager.g = g

        query = GraphQuery(mock_manager)
        term = query.get_term_by_id(999)

        assert term is None

    def test_get_term_by_id_no_term_attr(self) -> None:
        """Test getting term by ID when node lacks term attribute."""
        g = nx.DiGraph()
        g.add_node(1)  # No term attribute

        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager.g = g

        query = GraphQuery(mock_manager)
        term = query.get_term_by_id(1)

        assert term is None


class TestGraphQueryGetGraphInfo:
    """Tests for GraphQuery get_graph_info method."""

    def test_get_graph_info_populated(self) -> None:
        """Test getting graph info for populated graph."""
        g = nx.DiGraph()
        g.add_node(1, term="happiness")
        g.add_node(2, term="joy")
        g.add_edge(1, 2, relationship="synonym")

        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager.g = g
        mock_manager.dimensions = 2
        mock_manager._relationship_counts = {"synonym": 1}

        query = GraphQuery(mock_manager)
        info = query.get_graph_info()

        assert info["nodes"] == 2
        assert info["edges"] == 1
        assert info["dimensions"] == 2
        assert "sample_nodes" in info
        assert "sample_relationships" in info
        assert "relationship_types" in info

    def test_get_graph_info_empty(self) -> None:
        """Test getting graph info for empty graph."""
        g = nx.DiGraph()

        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager.g = g
        mock_manager.dimensions = 2
        mock_manager._relationship_counts = {}

        query = GraphQuery(mock_manager)
        info = query.get_graph_info()

        assert info["nodes"] == 0
        assert info["edges"] == 0
        assert info["sample_nodes"] == []
        assert info["sample_relationships"] == []


class TestGraphQueryGetSubgraph:
    """Tests for GraphQuery get_subgraph method."""

    def test_get_subgraph_success(self) -> None:
        """Test extracting subgraph successfully."""
        g = nx.DiGraph()
        g.add_node(1, term="happiness")
        g.add_node(2, term="joy")
        g.add_node(3, term="sadness")
        g.add_node(4, term="unrelated")
        g.add_edge(1, 2, relationship="synonym")
        g.add_edge(1, 3, relationship="antonym")

        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager.g = g
        mock_manager._term_to_id = {
            "happiness": 1,
            "joy": 2,
            "sadness": 3,
            "unrelated": 4,
        }

        query = GraphQuery(mock_manager)
        subgraph = query.get_subgraph("happiness", depth=1)

        assert subgraph.number_of_nodes() == 3
        assert 1 in subgraph
        assert 2 in subgraph
        assert 3 in subgraph
        assert 4 not in subgraph  # Not connected

    def test_get_subgraph_not_found(self) -> None:
        """Test extracting subgraph for non-existent term raises error."""
        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager._term_to_id = {}

        query = GraphQuery(mock_manager)

        with pytest.raises(NodeNotFoundError):
            query.get_subgraph("nonexistent")

    def test_get_subgraph_negative_depth(self) -> None:
        """Test that negative depth raises ValueError."""
        mock_manager = MagicMock()
        mock_manager.config = MagicMock()

        query = GraphQuery(mock_manager)

        with pytest.raises(ValueError, match="negative"):
            query.get_subgraph("test", depth=-1)

    def test_get_subgraph_depth_zero(self) -> None:
        """Test extracting subgraph with depth 0."""
        g = nx.DiGraph()
        g.add_node(1, term="happiness")
        g.add_node(2, term="joy")
        g.add_edge(1, 2, relationship="synonym")

        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager.g = g
        mock_manager._term_to_id = {"happiness": 1, "joy": 2}

        query = GraphQuery(mock_manager)
        subgraph = query.get_subgraph("happiness", depth=0)

        # Depth 0 should only include the center node
        assert subgraph.number_of_nodes() == 1
        assert 1 in subgraph


class TestGraphQueryRelationshipsByDimension:
    """Tests for GraphQuery get_relationships_by_dimension method."""

    def test_get_relationships_lexical(self) -> None:
        """Test getting lexical relationships."""
        g = nx.DiGraph()
        g.add_node(1, term="happiness")
        g.add_node(2, term="joy")
        g.add_edge(1, 2, relationship="synonym", dimension="lexical")

        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager.g = g

        query = GraphQuery(mock_manager)
        rels = query.get_relationships_by_dimension(dimension="lexical")

        assert len(rels) == 1
        source, target, rel_type, data = rels[0]
        assert source == "happiness"
        assert target == "joy"
        assert rel_type == "synonym"

    def test_get_relationships_emotional(self) -> None:
        """Test getting emotional relationships."""
        g = nx.DiGraph()
        g.add_node(1, term="happiness", valence=0.8)
        g.add_node(2, term="joy", valence=0.9)
        g.add_edge(1, 2, relationship="evokes", dimension="emotional")

        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager.g = g

        query = GraphQuery(mock_manager)
        rels = query.get_relationships_by_dimension(dimension="emotional")

        assert len(rels) == 1

    def test_get_relationships_filtered_by_type(self) -> None:
        """Test getting relationships filtered by type."""
        g = nx.DiGraph()
        g.add_node(1, term="happiness")
        g.add_node(2, term="joy")
        g.add_node(3, term="sadness")
        g.add_edge(1, 2, relationship="synonym", dimension="lexical")
        g.add_edge(1, 3, relationship="antonym", dimension="lexical")

        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager.g = g

        query = GraphQuery(mock_manager)
        rels = query.get_relationships_by_dimension(
            dimension="lexical", rel_type="synonym"
        )

        assert len(rels) == 1
        assert rels[0][2] == "synonym"

    def test_get_relationships_no_matches(self) -> None:
        """Test getting relationships when no matches found."""
        g = nx.DiGraph()
        g.add_node(1, term="happiness")
        g.add_node(2, term="joy")
        g.add_edge(1, 2, relationship="synonym", dimension="lexical")

        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager.g = g

        query = GraphQuery(mock_manager)
        rels = query.get_relationships_by_dimension(dimension="emotional")

        assert len(rels) == 0


class TestGraphQueryDisplaySummary:
    """Tests for GraphQuery display_graph_summary method."""

    def test_display_summary_runs_without_error(self, capsys) -> None:
        """Test that display_graph_summary runs without error."""
        g = nx.DiGraph()
        g.add_node(1, term="happiness")
        g.add_node(2, term="joy")
        g.add_edge(1, 2, relationship="synonym")

        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager.g = g
        mock_manager.dimensions = 2
        mock_manager._relationship_counts = {"synonym": 1}

        query = GraphQuery(mock_manager)
        query.display_graph_summary()

        captured = capsys.readouterr()
        assert "Graph Summary" in captured.out
        assert "Nodes:" in captured.out
        assert "Edges:" in captured.out

    def test_display_summary_empty_graph(self, capsys) -> None:
        """Test that display_graph_summary handles empty graph."""
        g = nx.DiGraph()

        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager.g = g
        mock_manager.dimensions = 2
        mock_manager._relationship_counts = {}

        query = GraphQuery(mock_manager)
        query.display_graph_summary()

        captured = capsys.readouterr()
        assert "Nodes: 0" in captured.out
        assert "Edges: 0" in captured.out
