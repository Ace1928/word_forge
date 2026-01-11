"""Tests for word_forge.graph.graph_worker module.

This module tests the GraphWorker class which handles background graph updates.
"""

import time


from word_forge.database.database_manager import DBManager
from word_forge.graph.graph_manager import GraphManager
from word_forge.graph.graph_worker import GraphWorker


def test_restart_returns_running_worker(tmp_path, monkeypatch):
    """Test that restart() returns a new running worker instance."""
    # Create real database and graph manager
    db_path = tmp_path / "test.db"
    db = DBManager(db_path=db_path)
    manager = GraphManager(db_manager=db)

    worker = GraphWorker(
        graph_manager=manager,
        poll_interval=0.01,
        output_path=str(tmp_path / "gexf.gexf"),
        visualization_path=str(tmp_path / "vis.html"),
    )

    def quick_cycle(self):
        """Quick cycle that stops immediately for testing."""
        time.sleep(0.02)
        self.stop()

    monkeypatch.setattr(GraphWorker, "_execute_update_cycle", quick_cycle)

    worker.start()
    worker.join(timeout=1)
    assert not worker.is_alive()

    new_worker = worker.restart()
    assert new_worker is not worker
    time.sleep(0.01)
    assert new_worker.is_alive()

    new_worker.stop()
    new_worker.join(timeout=1)


def test_worker_initialization(tmp_path):
    """Test that GraphWorker initializes correctly."""
    db_path = tmp_path / "init_test.db"
    db = DBManager(db_path=db_path)
    manager = GraphManager(db_manager=db)

    worker = GraphWorker(
        graph_manager=manager,
        poll_interval=1.0,
        output_path=str(tmp_path / "output.gexf"),
        visualization_path=str(tmp_path / "vis.html"),
    )

    assert worker.poll_interval == 1.0
    assert not worker.is_alive()


def test_worker_stop_when_not_running(tmp_path):
    """Test that stopping a non-running worker doesn't raise errors."""
    db_path = tmp_path / "stop_test.db"
    db = DBManager(db_path=db_path)
    manager = GraphManager(db_manager=db)

    worker = GraphWorker(
        graph_manager=manager,
        poll_interval=1.0,
    )

    # Should not raise
    worker.stop()
    assert not worker.is_alive()
