"""Tests for word_forge.graph.graph_builder module.

This module provides comprehensive tests for the GraphBuilder class
including graph construction, incremental updates, sample data management,
and database verification.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from word_forge.graph.graph_builder import GraphBuilder, GraphUpdateMetrics


class TestGraphUpdateMetrics:
    """Tests for the GraphUpdateMetrics dataclass."""

    def test_default_values(self) -> None:
        """Test default values for GraphUpdateMetrics."""
        metrics = GraphUpdateMetrics()
        assert metrics.new_nodes == 0
        assert metrics.new_edges == 0
        assert metrics.processed_words == 0
        assert metrics.max_last_refreshed == 0.0
        assert metrics.full_rebuild is False

    def test_custom_values(self) -> None:
        """Test custom values for GraphUpdateMetrics."""
        metrics = GraphUpdateMetrics(
            new_nodes=10,
            new_edges=20,
            processed_words=30,
            max_last_refreshed=1234567890.0,
            full_rebuild=True,
        )
        assert metrics.new_nodes == 10
        assert metrics.new_edges == 20
        assert metrics.processed_words == 30
        assert metrics.max_last_refreshed == 1234567890.0
        assert metrics.full_rebuild is True

    def test_immutability(self) -> None:
        """Test that GraphUpdateMetrics is immutable (frozen)."""
        metrics = GraphUpdateMetrics()
        with pytest.raises(AttributeError):
            metrics.new_nodes = 5  # type: ignore


class TestGraphBuilderInit:
    """Tests for GraphBuilder initialization."""

    def test_init_with_manager(self) -> None:
        """Test GraphBuilder initializes with manager reference."""
        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager._db_connection.return_value.__enter__ = MagicMock(
            return_value=MagicMock()
        )
        mock_manager._db_connection.return_value.__exit__ = MagicMock()

        with patch.object(
            GraphBuilder, "_load_last_refresh_watermark", return_value=0.0
        ):
            builder = GraphBuilder(mock_manager)

        assert builder.manager is mock_manager
        assert builder.logger is not None
        assert builder._config is mock_manager.config

    def test_init_loads_watermark(self) -> None:
        """Test GraphBuilder loads watermark on initialization."""
        mock_manager = MagicMock()
        mock_manager.config = MagicMock()

        with patch.object(
            GraphBuilder, "_load_last_refresh_watermark", return_value=1234567890.0
        ) as mock_load:
            builder = GraphBuilder(mock_manager)
            mock_load.assert_called_once()
            assert builder._last_refresh_watermark == 1234567890.0


class TestGraphBuilderProperties:
    """Tests for GraphBuilder properties."""

    def test_last_update_metrics_property(self) -> None:
        """Test last_update_metrics property returns metrics."""
        mock_manager = MagicMock()
        mock_manager.config = MagicMock()

        with patch.object(
            GraphBuilder, "_load_last_refresh_watermark", return_value=0.0
        ):
            builder = GraphBuilder(mock_manager)

        metrics = builder.last_update_metrics
        assert isinstance(metrics, GraphUpdateMetrics)
        assert metrics.new_nodes == 0
        assert metrics.new_edges == 0


class TestGraphBuilderDataFetching:
    """Tests for GraphBuilder data fetching operations."""

    def test_fetch_data_empty_database(self) -> None:
        """Test fetching data from empty database."""
        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager.config.sql_templates = {
            "check_words_table": "SELECT 1 FROM sqlite_master WHERE type='table' AND name='words'",
            "fetch_all_words": "SELECT id, term, last_refreshed FROM words",
            "check_relationships_table": "SELECT 1 FROM sqlite_master WHERE type='table' AND name='relationships'",
            "fetch_all_relationships": "SELECT word_id, related_term, relationship_type FROM relationships",
            "get_all_emotional_relationships": "SELECT word_id, related_term, relationship_type, valence, arousal FROM emotional_relationships",
        }

        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [(1,), (1,)]  # Tables exist
        mock_cursor.fetchall.side_effect = [[], []]  # Empty data

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_manager._db_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_manager._db_connection.return_value.__exit__ = MagicMock()

        with patch.object(
            GraphBuilder, "_load_last_refresh_watermark", return_value=0.0
        ):
            builder = GraphBuilder(mock_manager)

        # Mock the table existence checks more specifically
        with patch.object(builder, "_fetch_data", return_value=([], [], 0.0)):
            words, rels, refresh = builder._fetch_data()
            assert words == []
            assert rels == []


class TestGraphBuilderBuildGraph:
    """Tests for GraphBuilder build_graph method."""

    def test_build_graph_clears_existing(self) -> None:
        """Test that build_graph clears existing graph state."""
        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        # Use MagicMock for graph to track clear() calls
        mock_graph = MagicMock()
        mock_graph.number_of_nodes.return_value = 1
        mock_manager.g = mock_graph
        mock_manager._term_to_id = MagicMock()
        mock_manager._positions = MagicMock()
        mock_manager._relationship_counts = MagicMock()
        mock_manager.layout = MagicMock()

        with patch.object(
            GraphBuilder, "_load_last_refresh_watermark", return_value=0.0
        ):
            builder = GraphBuilder(mock_manager)

        with patch.object(builder, "_fetch_data", return_value=([], [], 0.0)):
            with patch.object(builder, "_persist_watermark"):
                builder.build_graph()

        # Verify clear was called on graph state
        mock_manager.g.clear.assert_called_once()
        mock_manager._term_to_id.clear.assert_called_once()
        mock_manager._positions.clear.assert_called_once()


class TestGraphBuilderUpdateGraph:
    """Tests for GraphBuilder update_graph method."""

    def test_update_graph_empty_triggers_build(self) -> None:
        """Test that update on empty graph triggers full build."""
        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager.g = MagicMock()
        mock_manager.g.number_of_nodes.return_value = 0

        with patch.object(
            GraphBuilder, "_load_last_refresh_watermark", return_value=0.0
        ):
            builder = GraphBuilder(mock_manager)

        with patch.object(builder, "build_graph") as mock_build:
            builder.update_graph()
            mock_build.assert_called_once()


class TestGraphBuilderVerifyDatabase:
    """Tests for GraphBuilder verify_database_tables method."""

    def test_verify_tables_both_exist(self) -> None:
        """Test verification when both tables exist."""
        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager.config.sql_templates = {
            "check_words_table": "SELECT 1",
            "check_relationships_table": "SELECT 1",
        }

        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [(1,), (1,)]  # Both tables exist

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_manager._db_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_manager._db_connection.return_value.__exit__ = MagicMock()

        with patch.object(
            GraphBuilder, "_load_last_refresh_watermark", return_value=0.0
        ):
            builder = GraphBuilder(mock_manager)

        result = builder.verify_database_tables()
        assert result is True

    def test_verify_tables_words_missing(self) -> None:
        """Test verification when words table is missing."""
        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager.config.sql_templates = {
            "check_words_table": "SELECT 1",
            "check_relationships_table": "SELECT 1",
        }

        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [None, (1,)]  # words missing

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_manager._db_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_manager._db_connection.return_value.__exit__ = MagicMock()

        with patch.object(
            GraphBuilder, "_load_last_refresh_watermark", return_value=0.0
        ):
            builder = GraphBuilder(mock_manager)

        result = builder.verify_database_tables()
        assert result is False

    def test_verify_tables_relationships_missing(self) -> None:
        """Test verification when relationships table is missing."""
        mock_manager = MagicMock()
        mock_manager.config = MagicMock()
        mock_manager.config.sql_templates = {
            "check_words_table": "SELECT 1",
            "check_relationships_table": "SELECT 1",
        }

        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [(1,), None]  # relationships missing

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_manager._db_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_manager._db_connection.return_value.__exit__ = MagicMock()

        with patch.object(
            GraphBuilder, "_load_last_refresh_watermark", return_value=0.0
        ):
            builder = GraphBuilder(mock_manager)

        result = builder.verify_database_tables()
        assert result is False


class TestGraphBuilderSampleData:
    """Tests for GraphBuilder ensure_sample_data method."""

    def test_ensure_sample_data_skips_if_populated(self) -> None:
        """Test that sample data is skipped if database has data."""
        mock_manager = MagicMock()
        mock_manager.config = MagicMock()

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (10,)  # 10 words exist

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_manager._db_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_manager._db_connection.return_value.__exit__ = MagicMock()

        with patch.object(
            GraphBuilder, "_load_last_refresh_watermark", return_value=0.0
        ):
            builder = GraphBuilder(mock_manager)

        result = builder.ensure_sample_data()
        assert result is False


class TestGraphBuilderWatermark:
    """Tests for GraphBuilder watermark operations."""

    def test_persist_watermark_updates_value(self) -> None:
        """Test that persisting watermark updates the value."""
        mock_manager = MagicMock()
        mock_manager.config = MagicMock()

        mock_conn = MagicMock()
        mock_manager._db_connection.return_value.__enter__ = MagicMock(
            return_value=mock_conn
        )
        mock_manager._db_connection.return_value.__exit__ = MagicMock()

        with patch.object(
            GraphBuilder, "_load_last_refresh_watermark", return_value=0.0
        ):
            builder = GraphBuilder(mock_manager)

        builder._persist_watermark(1234567890.0)
        assert builder._last_refresh_watermark == 1234567890.0

    def test_persist_watermark_skips_older(self) -> None:
        """Test that persisting older watermark is skipped."""
        mock_manager = MagicMock()
        mock_manager.config = MagicMock()

        with patch.object(
            GraphBuilder, "_load_last_refresh_watermark", return_value=2000000000.0
        ):
            builder = GraphBuilder(mock_manager)

        # This should be skipped since it's older
        builder._persist_watermark(1000000000.0)
        # Watermark should remain unchanged
        assert builder._last_refresh_watermark == 2000000000.0
