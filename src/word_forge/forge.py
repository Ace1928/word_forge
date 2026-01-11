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
import os
import sys
import time
from typing import TYPE_CHECKING, Callable, Iterable, List, Optional

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from word_forge.database.database_manager import DBManager
    from word_forge.graph.graph_manager import GraphManager


LOGGER = logging.getLogger("word_forge")

# Package version - dynamically retrieved from package metadata
__version__ = "0.1.0"


def _get_version() -> str:
    """Get the package version string.

    Returns:
        Version string in format 'word_forge VERSION'
    """
    try:
        from importlib.metadata import version, PackageNotFoundError

        return f"word_forge {version('word_forge')}"
    except (ImportError, PackageNotFoundError):
        return f"word_forge {__version__}"


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
    from word_forge.queue.worker_manager import WorkerManager
    from word_forge.graph.graph_manager import GraphManager
    from word_forge.graph.graph_worker import GraphWorker
    from word_forge.vectorizer.vector_store import VectorStore
    from word_forge.vectorizer.vector_worker import VectorWorker
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

    graph_manager = GraphManager(db_manager=db_manager)
    graph_worker = GraphWorker(graph_manager=graph_manager)

    vector_store = VectorStore(db_manager=db_manager)
    vector_worker = VectorWorker(
        db=db_manager, vector_store=vector_store, embedder="MiniLM"
    )

    manager = WorkerManager(logger=LOGGER)
    manager.register(worker_pool)
    manager.register(graph_worker)
    manager.register(vector_worker)

    seeds = (
        list(seed_words)
        if seed_words is not None
        else ["language", "knowledge", "system"]
    )
    for term in seeds:
        queue_manager.enqueue(term)

    with measure_execution("forge.start", {"workers": worker_count}) as metrics:
        manager.start_all()
        LOGGER.info(
            "Workers started in %.1fms",
            metrics.duration_ms,
        )

    start_time = time.time()
    last_report = start_time
    try:
        while True:
            time.sleep(0.5)
            if time.time() - last_report >= 5:
                status = worker_pool.get_status()
                stats = status["stats"]
                LOGGER.info(
                    "Progress - processed:%d success:%d errors:%d queue:%d",
                    stats.get("processed_count", 0),
                    stats.get("success_count", 0),
                    stats.get("error_count", 0),
                    status.get("queue_size", 0),
                )
                graph_status = graph_worker.get_status()
                if graph_status.get("last_new_nodes") or graph_status.get(
                    "last_new_edges"
                ):
                    LOGGER.info(
                        "Graph updates - nodes:+%d edges:+%d state:%s",
                        graph_status.get("last_new_nodes", 0),
                        graph_status.get("last_new_edges", 0),
                        graph_status.get("state", "unknown"),
                    )
                last_report = time.time()
            if (
                run_minutes is not None
                and (time.time() - start_time) > run_minutes * 60
            ):
                break
            if queue_manager.is_empty and not manager.any_alive():
                break
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user")
    finally:
        manager.stop_all()
        parser_refiner.shutdown()
        db_manager.close()
        LOGGER.info("Word Forge stopped")


def run_setup_nltk() -> int:
    """Ensure all required NLTK corpora are installed locally."""

    _setup_logging()
    LOGGER.info("Checking NLTK dependencies")
    from word_forge.utils.nltk_utils import ensure_nltk_data

    downloaded = ensure_nltk_data(logger=LOGGER)
    if downloaded:
        LOGGER.info("Downloaded NLTK corpora: %s", ", ".join(downloaded))
    else:
        LOGGER.info("NLTK corpora already installed; no downloads required")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the ``word_forge`` command."""

    parser = argparse.ArgumentParser(description="Word Forge command line interface")
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=_get_version(),
        help="Show program version and exit",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-error output",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose/debug output",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        metavar="FILE",
        help="Path to configuration file (YAML or JSON)",
    )
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

    graph_parser = subparsers.add_parser("graph", help="Graph management commands")
    graph_sub = graph_parser.add_subparsers(dest="graph_command")

    graph_build = graph_sub.add_parser(
        "build", help="Run the graph worker until a build cycle completes"
    )
    graph_build.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Seconds to wait for the worker to finish",
    )
    graph_build.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Seconds between graph worker polling cycles",
    )

    graph_visualize = graph_sub.add_parser(
        "visualize", help="Generate a graph visualization"
    )
    graph_visualize.add_argument(
        "--3d",
        dest="use_3d",
        action="store_true",
        help="Render the visualization with 3D layouts",
    )
    graph_visualize.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the generated visualization in a browser",
    )
    graph_visualize.add_argument(
        "--output",
        dest="output_path",
        default=None,
        help="Override the default visualization output path",
    )

    vector_parser = subparsers.add_parser(
        "vector", help="Vector index management commands"
    )
    vector_sub = vector_parser.add_subparsers(dest="vector_command")
    vector_index = vector_sub.add_parser(
        "index", help="Run the vector worker until one cycle completes"
    )
    vector_index.add_argument(
        "--embedder",
        default="MiniLM",
        help="Sentence transformer model name",
    )
    vector_index.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Seconds to wait for the indexing cycle",
    )
    vector_index.add_argument(
        "--poll-interval",
        type=float,
        default=0.25,
        help="Seconds between database polling cycles",
    )

    emotion_parser = subparsers.add_parser(
        "emotion", help="Emotion annotation utilities"
    )
    emotion_sub = emotion_parser.add_subparsers(dest="emotion_command")
    emotion_annotate = emotion_sub.add_parser(
        "annotate", help="Run the emotion worker until all words are tagged"
    )
    emotion_annotate.add_argument(
        "--strategy",
        default="random",
        help="Emotion assignment strategy (random, recursive, hybrid)",
    )
    emotion_annotate.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for annotation completion",
    )
    emotion_annotate.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="Seconds between annotation cycles",
    )

    demo_parser = subparsers.add_parser("demo", help="Pre-baked demo flows")
    demo_sub = demo_parser.add_subparsers(dest="demo_command")
    demo_full = demo_sub.add_parser(
        "full",
        help="Generate sample data, run indexing, and emit a visualization",
    )
    demo_full.add_argument(
        "--3d",
        dest="use_3d",
        action="store_true",
        help="Render demo visualization in 3D",
    )
    demo_full.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the demo visualization in a browser",
    )
    demo_full.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for each worker-driven stage",
    )

    subparsers.add_parser(
        "setup-nltk",
        help="Download the NLTK corpora required by Word Forge",
    )

    args = parser.parse_args(argv)

    # Configure logging based on quiet/verbose flags
    # These flags are global arguments, so they're always present
    if args.quiet:
        _setup_logging("ERROR")
    elif args.verbose:
        _setup_logging("DEBUG")

    # Load configuration file if specified
    if args.config:
        config_path = args.config
        if not os.path.exists(config_path):
            LOGGER.error("Configuration file not found: %s", config_path)
            return 1
        try:
            from word_forge.config import config

            config.load_from_file(config_path)
            LOGGER.info("Loaded configuration from: %s", config_path)
        except Exception as exc:
            LOGGER.error("Failed to load configuration: %s", exc)
            return 1

    exit_code = 0

    if args.command == "start":
        start(args.words, run_minutes=args.minutes, worker_count=args.workers)
    elif args.command == "graph":
        if args.graph_command == "build":
            exit_code = (
                0
                if run_graph_build(
                    poll_interval=args.poll_interval, timeout=args.timeout
                )
                else 1
            )
        elif args.graph_command == "visualize":
            exit_code = (
                0
                if run_graph_visualization(
                    output_path=args.output_path,
                    use_3d=args.use_3d,
                    open_in_browser=args.open_browser,
                )
                else 1
            )
        else:
            graph_parser.print_help()
            exit_code = 1
    elif args.command == "vector":
        if args.vector_command == "index":
            exit_code = (
                0
                if run_vector_index(
                    embedder=args.embedder,
                    poll_interval=args.poll_interval,
                    timeout=args.timeout,
                )
                else 1
            )
        else:
            vector_parser.print_help()
            exit_code = 1
    elif args.command == "emotion":
        if args.emotion_command == "annotate":
            exit_code = (
                0
                if run_emotion_annotation(
                    strategy=args.strategy,
                    poll_interval=args.poll_interval,
                    timeout=args.timeout,
                )
                else 1
            )
        else:
            emotion_parser.print_help()
            exit_code = 1
    elif args.command == "demo":
        if args.demo_command == "full":
            exit_code = (
                0
                if run_demo_full(
                    use_3d=args.use_3d,
                    open_in_browser=args.open_browser,
                    timeout=args.timeout,
                )
                else 1
            )
        else:
            demo_parser.print_help()
            exit_code = 1
    elif args.command == "setup-nltk":
        exit_code = run_setup_nltk()
    else:
        parser.print_help()
        exit_code = 1

    return exit_code


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main(sys.argv[1:]))


def _wait_for_condition(
    description: str,
    predicate: Callable[[], bool],
    timeout: float = 60.0,
    poll_interval: float = 0.5,
) -> bool:
    """Poll ``predicate`` until it returns ``True`` or the timeout elapses."""

    end_time = time.time() + timeout
    while time.time() < end_time:
        try:
            if predicate():
                LOGGER.info("Completed: %s", description)
                return True
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.debug("Condition '%s' raised %s", description, exc)
        time.sleep(poll_interval)

    LOGGER.error("Timed out waiting for %s after %.1fs", description, timeout)
    return False


def run_graph_build(
    *,
    graph_manager: Optional["GraphManager"] = None,
    poll_interval: float = 1.0,
    timeout: float = 120.0,
) -> bool:
    """Run :class:`GraphWorker` until a full update cycle completes."""

    _setup_logging()
    from word_forge.database.database_manager import DBManager
    from word_forge.graph.graph_manager import GraphManager
    from word_forge.graph.graph_worker import GraphWorker
    from word_forge.queue.worker_manager import WorkerManager

    owns_manager = graph_manager is None
    db_manager: Optional[DBManager] = None
    if graph_manager is None:
        db_manager = DBManager()
        graph_manager = GraphManager(db_manager=db_manager)
    else:
        db_manager = graph_manager.db_manager

    worker = GraphWorker(
        graph_manager=graph_manager, poll_interval=poll_interval, daemon=False
    )
    manager = WorkerManager(logger=LOGGER)
    manager.register(worker)
    LOGGER.info("Starting graph build worker")

    try:
        manager.start_all()
        completed = _wait_for_condition(
            "graph build",
            lambda: worker.get_status()["update_count"] > 0,
            timeout=timeout,
        )
        return completed
    finally:
        manager.stop_all()
        worker.join(timeout=5)
        if owns_manager and db_manager is not None:
            db_manager.close()


def run_graph_visualization(
    *,
    graph_manager: Optional["GraphManager"] = None,
    output_path: Optional[str] = None,
    use_3d: bool = False,
    open_in_browser: bool = False,
) -> bool:
    """Build the graph (if needed) and emit a visualization file."""

    _setup_logging()
    from word_forge.database.database_manager import DBManager
    from word_forge.graph.graph_manager import GraphManager

    owns_manager = graph_manager is None
    db_manager: Optional[DBManager] = None
    if graph_manager is None:
        db_manager = DBManager()
        graph_manager = GraphManager(db_manager=db_manager)

    try:
        graph_manager.build_graph()
        graph_manager.visualize(
            output_path=output_path,
            use_3d=use_3d if use_3d else None,
            open_in_browser=open_in_browser,
        )
        LOGGER.info("Graph visualization ready")
        return True
    except Exception as exc:  # pragma: no cover - visualization dependent
        LOGGER.error("Graph visualization failed: %s", exc)
        return False
    finally:
        if owns_manager and db_manager is not None:
            db_manager.close()


def run_vector_index(
    *,
    db_manager: Optional["DBManager"] = None,
    embedder: str = "MiniLM",
    poll_interval: float = 0.25,
    timeout: float = 120.0,
) -> bool:
    """Run :class:`VectorWorker` long enough to finish an indexing cycle."""

    _setup_logging()
    from word_forge.database.database_manager import DBManager
    from word_forge.queue.worker_manager import WorkerManager
    from word_forge.vectorizer.vector_store import VectorStore
    from word_forge.vectorizer.vector_worker import VectorWorker

    owns_db = db_manager is None
    db = db_manager or DBManager()
    db.create_tables()

    vector_store = VectorStore(db_manager=db)
    worker = VectorWorker(
        db=db,
        vector_store=vector_store,
        embedder=embedder,
        poll_interval=poll_interval,
        daemon=False,
        logger=LOGGER,
    )
    manager = WorkerManager(logger=LOGGER)
    manager.register(worker)
    LOGGER.info("Starting vector indexing worker")

    try:
        manager.start_all()
        completed = _wait_for_condition(
            "vector indexing",
            lambda: worker.last_processed is not None,
            timeout=timeout,
        )
        return completed
    finally:
        manager.stop_all()
        worker.join(timeout=5)
        if owns_db:
            db.close()


def _remaining_unemotioned_words(db_manager: "DBManager") -> int:
    """Return number of words lacking emotion annotations."""

    query = """
        SELECT COUNT(*)
        FROM words w
        LEFT JOIN word_emotion we ON w.id = we.word_id
        WHERE we.word_id IS NULL
    """
    with db_manager.get_connection() as conn:
        cursor = conn.execute(query)
        (count,) = cursor.fetchone()
        return int(count)


def run_emotion_annotation(
    *,
    db_manager: Optional["DBManager"] = None,
    strategy: str = "random",
    poll_interval: float = 0.5,
    timeout: float = 180.0,
) -> bool:
    """Run :class:`EmotionWorker` until all words have annotations."""

    _setup_logging()
    from word_forge.database.database_manager import DBManager
    from word_forge.emotion.emotion_manager import EmotionManager
    from word_forge.emotion.emotion_worker import EmotionWorker
    from word_forge.queue.worker_manager import WorkerManager

    owns_db = db_manager is None
    db = db_manager or DBManager()
    db.create_tables()

    emotion_manager = EmotionManager(db)
    worker = EmotionWorker(
        db=db,
        emotion_manager=emotion_manager,
        poll_interval=poll_interval,
        strategy=strategy,
        daemon=False,
    )
    manager = WorkerManager(logger=LOGGER)
    manager.register(worker)
    LOGGER.info("Starting emotion annotation worker")

    def _all_tagged() -> bool:
        return _remaining_unemotioned_words(db) == 0

    try:
        manager.start_all()
        completed = _wait_for_condition(
            "emotion annotation",
            lambda: _all_tagged(),
            timeout=timeout,
        )
        return completed
    finally:
        manager.stop_all()
        worker.join(timeout=5)
        if owns_db:
            db.close()


def run_demo_full(
    *,
    use_3d: bool = False,
    open_in_browser: bool = False,
    timeout: float = 300.0,
) -> bool:
    """Generate sample data, vectors, and a visualization for demos."""

    _setup_logging()
    from word_forge.database.database_manager import DBManager
    from word_forge.graph.graph_manager import GraphManager

    db_manager = DBManager()
    try:
        db_manager.create_tables()
        graph_manager = GraphManager(db_manager=db_manager)
        graph_manager.ensure_sample_data()

        vector_ok = run_vector_index(
            db_manager=db_manager, poll_interval=0.25, timeout=timeout
        )
        graph_ok = run_graph_build(
            graph_manager=graph_manager, poll_interval=0.25, timeout=timeout
        )
        viz_ok = run_graph_visualization(
            graph_manager=graph_manager,
            use_3d=use_3d,
            open_in_browser=open_in_browser,
        )
        return vector_ok and graph_ok and viz_ok
    finally:
        db_manager.close()
