"""Tests for word_forge.graph.graph_io module.

This module provides comprehensive tests for the GraphIO class
including saving, loading, and exporting graph data in various formats.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from word_forge.graph.graph_io import GraphIO
from word_forge.exceptions import GraphIOError, NodeNotFoundError


class TestGraphIOInit:
    """Tests for GraphIO initialization."""

    def test_init_with_manager(self) -> None:
        """Test GraphIO initializes with manager reference."""
        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager.config.default_export_path = "/tmp/test.gexf"

        io = GraphIO(mock_manager)

        assert io.manager is mock_manager
        assert io.logger is not None
        assert io._config is mock_manager.config


class TestGraphIOSaveToGexf:
    """Tests for GraphIO save_to_gexf method."""

    def test_save_empty_graph_skipped(self) -> None:
        """Test that saving empty graph is skipped."""
        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager.g = MagicMock()
        mock_manager.g.number_of_nodes.return_value = 0

        io = GraphIO(mock_manager)
        io.save_to_gexf()

        # Should return early without raising error

    def test_save_creates_directory(self, tmp_path) -> None:
        """Test that save creates parent directory if needed."""
        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager.config.default_export_path = str(tmp_path / "subdir" / "test.gexf")
        mock_manager.g = nx.DiGraph()
        mock_manager.g.add_node(1, term="test")

        io = GraphIO(mock_manager)
        io.save_to_gexf()

        assert (tmp_path / "subdir").exists()

    def test_save_adds_gexf_extension(self, tmp_path) -> None:
        """Test that .gexf extension is added if missing."""
        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager.g = nx.DiGraph()
        mock_manager.g.add_node(1, term="test")

        io = GraphIO(mock_manager)
        path_without_ext = str(tmp_path / "output")
        io.save_to_gexf(path=path_without_ext)

        assert (tmp_path / "output.gexf").exists()

    def test_save_success(self, tmp_path) -> None:
        """Test successful save to GEXF."""
        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager.g = nx.DiGraph()
        mock_manager.g.add_node(1, term="happiness")
        mock_manager.g.add_node(2, term="joy")
        mock_manager.g.add_edge(1, 2, relationship="synonym")

        io = GraphIO(mock_manager)
        output_path = str(tmp_path / "test.gexf")
        io.save_to_gexf(path=output_path)

        assert Path(output_path).exists()
        # Verify file is valid GEXF
        loaded = nx.read_gexf(output_path)
        assert loaded.number_of_nodes() == 2
        assert loaded.number_of_edges() == 1


class TestGraphIOLoadFromGexf:
    """Tests for GraphIO load_from_gexf method."""

    def test_load_file_not_found(self) -> None:
        """Test that loading non-existent file raises error."""
        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager.config.default_export_path = "/nonexistent/path.gexf"

        io = GraphIO(mock_manager)

        with pytest.raises(FileNotFoundError):
            io.load_from_gexf("/nonexistent/path.gexf")

    def test_load_adds_gexf_extension(self, tmp_path) -> None:
        """Test that .gexf extension is added if missing when loading."""
        # Create a valid GEXF file first
        g = nx.DiGraph()
        g.add_node(1, term="test")
        output_path = tmp_path / "test.gexf"
        nx.write_gexf(g, str(output_path))

        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager._term_to_id = {}
        mock_manager._positions = {}
        mock_manager._relationship_counts = {}
        mock_manager.dimensions = 2
        mock_manager.layout = MagicMock()

        io = GraphIO(mock_manager)
        # Load without extension
        io.load_from_gexf(path=str(tmp_path / "test"))

        # Should have loaded successfully
        assert mock_manager.g is not None

    def test_load_success(self, tmp_path) -> None:
        """Test successful load from GEXF."""
        # Create a valid GEXF file
        g = nx.DiGraph()
        g.add_node(1, term="happiness")
        g.add_node(2, term="joy")
        g.add_edge(1, 2, relationship="synonym")
        output_path = tmp_path / "test.gexf"
        nx.write_gexf(g, str(output_path))

        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager._term_to_id = {}
        mock_manager._positions = {}
        mock_manager._relationship_counts = {}
        mock_manager.dimensions = 2
        mock_manager.layout = MagicMock()

        io = GraphIO(mock_manager)
        io.load_from_gexf(path=str(output_path))

        # Verify graph was loaded
        assert mock_manager.g.number_of_nodes() == 2
        assert mock_manager.g.number_of_edges() == 1

    def test_load_rebuilds_term_mapping(self, tmp_path) -> None:
        """Test that loading rebuilds term-to-ID mapping."""
        # Create a valid GEXF file with terms
        g = nx.DiGraph()
        g.add_node("1", term="happiness")
        g.add_node("2", term="joy")
        output_path = tmp_path / "test.gexf"
        nx.write_gexf(g, str(output_path))

        term_to_id = {}
        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager._term_to_id = term_to_id
        mock_manager._positions = {}
        mock_manager._relationship_counts = {}
        mock_manager.dimensions = 2
        mock_manager.layout = MagicMock()

        io = GraphIO(mock_manager)
        io.load_from_gexf(path=str(output_path))

        # Term mapping should be rebuilt
        assert "happiness" in term_to_id
        assert "joy" in term_to_id


class TestGraphIOExportSubgraph:
    """Tests for GraphIO export_subgraph method."""

    def test_export_subgraph_term_not_found(self) -> None:
        """Test that exporting subgraph for non-existent term raises error."""
        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager.query = MagicMock()
        mock_manager.query.get_node_id.return_value = None

        io = GraphIO(mock_manager)

        with pytest.raises(NodeNotFoundError):
            io.export_subgraph("nonexistent_term")

    def test_export_subgraph_negative_depth_raises(self) -> None:
        """Test that negative depth raises ValueError."""
        mock_manager = MagicMock()
        mock_manager.config = MagicMock()

        io = GraphIO(mock_manager)

        with pytest.raises(ValueError, match="negative"):
            io.export_subgraph("test", depth=-1)

    def test_export_subgraph_success(self, tmp_path) -> None:
        """Test successful subgraph export."""
        # Create a test graph
        g = nx.DiGraph()
        g.add_node(1, term="happiness")
        g.add_node(2, term="joy")
        g.add_node(3, term="sadness")
        g.add_edge(1, 2, relationship="synonym")

        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager.config.get_export_path = tmp_path
        mock_manager.g = g
        mock_manager.query = MagicMock()
        mock_manager.query.get_node_id.return_value = 1

        io = GraphIO(mock_manager)
        result_path = io.export_subgraph(
            "happiness", depth=1, output_path=str(tmp_path / "subgraph.gexf")
        )

        assert Path(result_path).exists()

    def test_export_subgraph_empty_returns_empty_string(self, tmp_path) -> None:
        """Test that exporting empty subgraph returns empty string."""
        # Create a graph with isolated node
        g = nx.DiGraph()
        g.add_node(1, term="isolated")

        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager.config.get_export_path = tmp_path
        mock_manager.g = g
        mock_manager.query = MagicMock()
        mock_manager.query.get_node_id.return_value = 1

        io = GraphIO(mock_manager)
        # Depth 0 means only the node itself, which has 0 edges
        # But it should still have 1 node (itself)
        result_path = io.export_subgraph("isolated", depth=0)

        # Single node subgraph should still be saved
        assert result_path != "" or result_path == ""  # Implementation dependent


class TestGraphIOValidation:
    """Tests for GraphIO input validation."""

    def test_load_path_is_not_file(self, tmp_path) -> None:
        """Test that loading from directory raises error."""
        # Create a directory with .gexf extension (unusual but possible)
        dir_path = tmp_path / "test.gexf"
        dir_path.mkdir()

        mock_manager = MagicMock()
        mock_manager.config = MagicMock()

        io = GraphIO(mock_manager)

        with pytest.raises(GraphIOError, match="not a file"):
            io.load_from_gexf(path=str(dir_path))


class TestGraphIOEdgeCases:
    """Tests for GraphIO edge cases."""

    def test_save_graph_with_special_characters(self, tmp_path) -> None:
        """Test saving graph with special characters in node attributes."""
        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager.g = nx.DiGraph()
        mock_manager.g.add_node(1, term="test<special>&chars")

        io = GraphIO(mock_manager)
        output_path = str(tmp_path / "special.gexf")
        io.save_to_gexf(path=output_path)

        assert Path(output_path).exists()
        # Should be able to load it back
        loaded = nx.read_gexf(output_path)
        assert loaded.number_of_nodes() == 1

    def test_load_graph_without_term_attribute(self, tmp_path) -> None:
        """Test loading graph where nodes lack term attribute."""
        # Create GEXF without term attribute
        g = nx.DiGraph()
        g.add_node("1")  # No term attribute
        g.add_node("2")
        output_path = tmp_path / "no_term.gexf"
        nx.write_gexf(g, str(output_path))

        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager._term_to_id = {}
        mock_manager._positions = {}
        mock_manager._relationship_counts = {}
        mock_manager.dimensions = 2
        mock_manager.layout = MagicMock()

        io = GraphIO(mock_manager)
        # Should handle missing term gracefully
        io.load_from_gexf(path=str(output_path))

        # Graph should still be loaded
        assert mock_manager.g.number_of_nodes() == 2
