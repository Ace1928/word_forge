"""Integration coverage for the ingestion pipeline."""

from __future__ import annotations

import time
from pathlib import Path

LLM_MODEL = "sshleifer/tiny-gpt2"


class _FastModelState:
    """Deterministic LLM stand-in for CI-safe parser integration tests."""

    def get_model_name(self) -> str:
        return LLM_MODEL

    def generate_text(self, prompt: str, *args, **kwargs) -> str:
        if '"ipa"' in prompt and '"arpabet"' in prompt:
            return '{"ipa":"haepi","arpabet":"HH AE1 P IY0","stress_pattern":"1-0"}'
        if '"phrases"' in prompt and '"entities"' in prompt:
            return '{"phrases":[],"entities":[]}'
        if '"pitch_multiplier"' in prompt:
            return (
                '{"pitch_multiplier":1.0,"duration_multiplier":1.0,'
                '"intensity_multiplier":1.0,"pitch_contour":"flat","emphasis_indices":[]}'
            )
        return "Example Sentence: The happy child smiled."


def _wait_until(predicate, timeout: float = 30.0, interval: float = 0.25) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_end_to_end_crawl_and_ingest(tmp_path: Path) -> None:
    """Process a real word through the queue -> parser -> DB pipeline."""
    from word_forge.database.database_manager import DBManager
    from word_forge.parser.parser_refiner import ParserRefiner
    from word_forge.queue.queue_manager import QueueManager
    from word_forge.queue.queue_worker import (
        ParallelWordProcessor,
        WordProcessor,
        WorkerPoolConfig,
    )
    from word_forge.utils.nltk_utils import ensure_nltk_data

    ensure_nltk_data()

    db_path = tmp_path / "ingestion.db"
    db_manager = DBManager(db_path=db_path)
    queue_manager: QueueManager[str] = QueueManager()
    queue_manager.start()

    parser_refiner = ParserRefiner(
        db_manager=db_manager,
        queue_manager=queue_manager,
        model_name=LLM_MODEL,
        llm_state=_FastModelState(),
    )
    processor = WordProcessor(db_manager=db_manager, parser_refiner=parser_refiner)
    worker_pool = ParallelWordProcessor(processor, config=WorkerPoolConfig(worker_count=1))

    queue_manager.enqueue("happy")
    worker_pool.start()

    try:
        assert _wait_until(lambda: processor.stats.processed_count > 0, timeout=60.0)
        assert _wait_until(
            lambda: db_manager.word_exists("happy"),
            timeout=10.0,
        )

        entry = db_manager.get_word_entry("happy")
        assert entry["term"] == "happy"

        assert _wait_until(
            lambda: len(db_manager.get_relationships(str(entry["id_int"]))) > 0,
            timeout=10.0,
        )
    finally:
        worker_pool.stop(wait=True)
        queue_manager.stop()
        parser_refiner.shutdown()
        db_manager.close()


def test_queue_running_state_allows_processing(tmp_path: Path) -> None:
    """Queue must be running to allow worker loops to dequeue items."""
    from word_forge.database.database_manager import DBManager
    from word_forge.parser.parser_refiner import ParserRefiner
    from word_forge.queue.queue_manager import QueueManager, QueueState
    from word_forge.queue.queue_worker import (
        ParallelWordProcessor,
        WordProcessor,
        WorkerPoolConfig,
    )

    db_path = tmp_path / "queue_state.db"
    db_manager = DBManager(db_path=db_path)
    queue_manager: QueueManager[str] = QueueManager()
    queue_manager.start()

    parser_refiner = ParserRefiner(
        db_manager=db_manager,
        queue_manager=queue_manager,
        model_name=LLM_MODEL,
        llm_state=_FastModelState(),
    )
    processor = WordProcessor(db_manager=db_manager, parser_refiner=parser_refiner)
    worker_pool = ParallelWordProcessor(processor, config=WorkerPoolConfig(worker_count=1))

    queue_manager.enqueue("oxygen")
    worker_pool.start()

    try:
        assert queue_manager.state == QueueState.RUNNING
        assert _wait_until(lambda: processor.stats.processed_count > 0, timeout=60.0)
    finally:
        worker_pool.stop(wait=True)
        queue_manager.stop()
        parser_refiner.shutdown()
        db_manager.close()
