import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from word_forge.queue.worker_manager import WorkerManager


class DummyWorker:
    def __init__(self):
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def is_alive(self) -> bool:
        return self.started and not self.stopped

    def join(self, timeout=None):
        pass


def test_start_and_stop_all():
    w1 = DummyWorker()
    w2 = DummyWorker()
    manager = WorkerManager()
    manager.register(w1)
    manager.register(w2)

    manager.start_all()
    assert w1.started and w2.started
    assert manager.any_alive()

    manager.stop_all()
    assert w1.stopped and w2.stopped
    assert not manager.any_alive()
