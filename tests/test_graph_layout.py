"""Tests for graph layout functionality.

This module tests the GraphManager's layout system using the real networkx library.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import networkx as nx
import numpy as np

from word_forge.database.database_manager import DBManager
from word_forge.graph.graph_manager import GraphManager


def create_manager(tmp_path):
    db = DBManager(tmp_path / "layout.db")
    return GraphManager(db_manager=db)


def test_incremental_layout_updates_existing_positions(tmp_path):
    manager = create_manager(tmp_path)
    a = manager.add_word_node("a")
    b = manager.add_word_node("b")

    manager.layout.compute_layout()
    pos_before = {
        a: manager._positions[a],
        b: manager._positions[b],
    }

    c = manager.add_word_node("c")

    assert np.allclose(manager._positions[a], pos_before[a])
    assert np.allclose(manager._positions[b], pos_before[b])
    assert c in manager._positions


def test_view_layout_is_deterministic_and_does_not_mutate_manager(tmp_path):
    manager = create_manager(tmp_path)
    alpha = manager.add_word_node("alpha")
    beta = manager.add_word_node("beta")
    gamma = manager.add_word_node("gamma")
    manager.add_relationship(alpha, beta, "related")
    manager.add_relationship(beta, gamma, "related")
    manager._positions.clear()

    first = manager.layout.compute_positions(manager.g, dimensions=2)
    second = manager.layout.compute_positions(manager.g, dimensions=2)

    assert manager._positions == {}
    assert first.keys() == second.keys()
    assert all(np.allclose(first[node_id], second[node_id]) for node_id in first)


def test_layout_is_reproducible_across_insertion_orders(tmp_path):
    manager = create_manager(tmp_path)
    first_graph = nx.Graph()
    first_graph.add_nodes_from([3, 1, 2])
    first_graph.add_edges_from([(3, 2), (2, 1)])
    second_graph = nx.Graph()
    second_graph.add_nodes_from([1, 2, 3])
    second_graph.add_edges_from([(1, 2), (2, 3)])

    first = manager.layout.compute_positions(first_graph, dimensions=2)
    second = manager.layout.compute_positions(second_graph, dimensions=2)

    assert first.keys() == second.keys()
    assert all(np.allclose(first[node_id], second[node_id]) for node_id in first)
