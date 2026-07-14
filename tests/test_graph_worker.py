"""Tests for word_forge.graph.graph_worker module using real components."""

from __future__ import annotations

import time
from pathlib import Path

from word_forge.database.database_manager import DBManager
from word_forge.graph.graph_manager import GraphManager
from word_forge.graph.graph_worker import GraphWorker


def _wait_until(predicate, timeout: float = 10.0, interval: float = 0.2) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_restart_returns_running_worker(tmp_path: Path) -> None:
    db = DBManager(db_path=tmp_path / "test.db")
    manager = GraphManager(db_manager=db)

    worker = GraphWorker(
        graph_manager=manager,
        poll_interval=0.2,
        output_path=str(tmp_path / "gexf.gexf"),
        visualization_path=str(tmp_path / "vis.html"),
    )

    worker.start()
    try:
        assert _wait_until(
            lambda: worker.get_status()["update_count"] > 0, timeout=10.0
        )
    finally:
        worker.stop()
        worker.join(timeout=5)

    new_worker = worker.restart()
    assert new_worker is not worker
    assert _wait_until(lambda: new_worker.is_alive(), timeout=5.0)

    new_worker.stop()
    new_worker.join(timeout=5)


def test_worker_initialization(tmp_path: Path) -> None:
    db = DBManager(db_path=tmp_path / "init_test.db")
    manager = GraphManager(db_manager=db)

    worker = GraphWorker(
        graph_manager=manager,
        poll_interval=1.0,
        output_path=str(tmp_path / "output.gexf"),
        visualization_path=str(tmp_path / "vis.html"),
    )

    assert worker.poll_interval == 1.0
    assert not worker.is_alive()


def test_nonexistent_visualization_path_is_treated_as_directory(
    tmp_path: Path,
) -> None:
    db = DBManager(db_path=tmp_path / "path_test.db")
    manager = GraphManager(db_manager=db)
    output_directory = tmp_path / "new_visualizations"

    worker = GraphWorker(
        graph_manager=manager,
        visualization_path=str(output_directory),
    )

    assert worker.visualization_path == str(output_directory / "lexical_graph.html")
    db.close()


def test_worker_stop_when_not_running(tmp_path: Path) -> None:
    db = DBManager(db_path=tmp_path / "stop_test.db")
    manager = GraphManager(db_manager=db)

    worker = GraphWorker(graph_manager=manager, poll_interval=1.0)

    worker.stop()
    assert not worker.is_alive()


def test_synchronous_refresh_captures_late_database_writes(tmp_path: Path) -> None:
    """A final refresh must include data written after the prior cycle."""
    db = DBManager(db_path=tmp_path / "late_write.db")
    manager = GraphManager(db_manager=db)
    worker = GraphWorker(
        graph_manager=manager,
        poll_interval=60.0,
        output_path=str(tmp_path / "late_write.gexf"),
        visualization_path=str(tmp_path / "late_write.html"),
    )

    first_metrics = worker.refresh()
    assert first_metrics.processed_words == 0

    db.insert_or_update_word("alpha")
    db.insert_or_update_word("beta")
    assert db.insert_relationship("alpha", "beta", "related") is True

    final_metrics = worker.refresh()

    assert final_metrics.full_rebuild is True
    assert final_metrics.new_nodes == 2
    assert final_metrics.new_edges == 1
    assert manager.get_node_count() == 2
    assert manager.get_edge_count() == 1
    assert (tmp_path / "late_write.gexf").is_file()
    db.close()
