"""Top-level orchestration utilities for Word Forge.

This module exposes a minimal CLI allowing users to start the lexical
processing pipeline with a single command:

    word_forge start

Running the command launches the queue based parser/refiner which
recursively builds lexical entries for all discovered terms.  The
pipeline relies on :class:`ParserRefiner` which in turn uses the
:func:`create_lexical_dataset` function from :mod:`lexical_proto` to
pull in data from WordNet and other sources and to generate additional
lexical insight using a language model.

The CLI is intentionally lightweight so that it can be used as a quick
entry point.  More advanced control flows can still be achieved by
using the underlying classes directly.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Iterable, List, Optional


LOGGER = logging.getLogger("word_forge")


def _setup_logging(level: str = "INFO") -> None:
    """Configure basic console logging."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def start(
    seed_words: Optional[Iterable[str]] = None,
    run_minutes: Optional[float] = None,
    worker_count: int = 4,
) -> None:
    """Launch the Word Forge processing pipeline.

    Parameters
    ----------
    seed_words:
        Optional iterable of seed terms. When ``None`` a default
        selection of general words is used.
    run_minutes:
        Optional duration to run before shutting down. ``None`` means run
        until interrupted.
    """

    from word_forge.database.database_manager import DBManager
    from word_forge.parser.parser_refiner import ParserRefiner
    from word_forge.queue.queue_manager import QueueManager
    from word_forge.queue.queue_worker import (
        ParallelWordProcessor,
        WordProcessor,
        WorkerPoolConfig,
    )
    from word_forge.configs.config_essentials import measure_execution

    _setup_logging()
    LOGGER.info("Starting Word Forge")

    db_manager = DBManager()
    queue_manager: QueueManager[str] = QueueManager()
    parser_refiner = ParserRefiner(db_manager=db_manager, queue_manager=queue_manager)
    processor = WordProcessor(
        db_manager=db_manager, parser_refiner=parser_refiner, logger=LOGGER
    )
    pool_config = WorkerPoolConfig(worker_count=worker_count)
    worker_pool = ParallelWordProcessor(processor, config=pool_config, logger=LOGGER)

    seeds = (
        list(seed_words)
        if seed_words is not None
        else ["language", "knowledge", "system"]
    )
    for term in seeds:
        queue_manager.enqueue(term)

    with measure_execution("forge.start", {"workers": worker_count}) as metrics:
        worker_pool.start()
        LOGGER.info(
            "Worker pool started with %d workers in %.1fms",
            worker_count,
            metrics.duration_ms,
        )

    start_time = time.time()
    try:
        while True:
            time.sleep(0.5)
            if (
                run_minutes is not None
                and (time.time() - start_time) > run_minutes * 60
            ):
                break
            if queue_manager.is_empty and not worker_pool._active:
                # Nothing left to process and workers stopped
                break
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user")
    finally:
        worker_pool.stop()
        parser_refiner.shutdown()
        db_manager.close()
        LOGGER.info("Word Forge stopped")


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the ``word_forge`` command."""

    parser = argparse.ArgumentParser(description="Word Forge command line interface")
    subparsers = parser.add_subparsers(dest="command")

    start_parser = subparsers.add_parser("start", help="Start processing seed words")
    start_parser.add_argument("words", nargs="*", help="Optional seed words")
    start_parser.add_argument(
        "--minutes",
        type=float,
        default=None,
        help="Run for a limited number of minutes",
    )
    start_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads",
    )

    args = parser.parse_args(argv)

    if args.command == "start":
        start(args.words, run_minutes=args.minutes, worker_count=args.workers)
    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main(sys.argv[1:])
