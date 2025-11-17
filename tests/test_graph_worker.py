import sys
import types
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Provide a lightweight GraphManager stub so GraphWorker can be imported
gm_mod = types.ModuleType("word_forge.graph.graph_manager")

from word_forge.graph.graph_builder import GraphUpdateMetrics


class DummyGraphManager:
    def __init__(self):
        self._metrics = GraphUpdateMetrics()

    def build_graph(self) -> None:
        self._metrics = GraphUpdateMetrics(full_rebuild=True)

    def update_graph(self) -> int:
        self._metrics = GraphUpdateMetrics()
        return 0

    def get_node_count(self) -> int:
        return 0

    def get_last_update_metrics(self) -> GraphUpdateMetrics:
        return self._metrics

    def save_to_gexf(self, path: str) -> None:
        pass

    def visualize(self, output_path: str, open_in_browser: bool = False) -> None:
        pass


gm_mod.GraphManager = DummyGraphManager
sys.modules["word_forge.graph.graph_manager"] = gm_mod

from word_forge.graph.graph_worker import GraphWorker


def test_restart_returns_running_worker(tmp_path, monkeypatch):
    manager = DummyGraphManager()
    worker = GraphWorker(
        graph_manager=manager,
        poll_interval=0.01,
        output_path=str(tmp_path / "gexf.gexf"),
        visualization_path=str(tmp_path / "vis.html"),
    )

    def quick_cycle(self):
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
