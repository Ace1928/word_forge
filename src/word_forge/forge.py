"""Top-level orchestration utilities for Word Forge.

This module exposes a minimal CLI allowing users to start the lexical
processing pipeline with a single command:

    word_forge start

Running the command launches the queue based parser/refiner which
recursively builds lexical entries for all discovered terms.  The
pipeline relies on :class:`ParserRefiner` which in turn uses the
:func:`create_lexical_dataset` function from
:mod:`word_forge.parser.lexical_functions` to pull in data from
WordNet and other sources and to generate additional lexical insight
using a language model.

The CLI is intentionally lightweight so that it can be used as a quick
entry point.  More advanced control flows can still be achieved by
using the underlying classes directly.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sqlite3
import sys
import tempfile
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from word_forge._version import __version__
from word_forge.observability import eidosian
from word_forge.utils import metrics

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from word_forge.database.database_manager import DBManager
    from word_forge.graph.graph_manager import GraphManager


LOGGER = logging.getLogger("word_forge")

# =============================================================================
# Processing Constants
# =============================================================================

# Main loop timing intervals (seconds)
MAIN_LOOP_SLEEP_INTERVAL: float = 0.5
PROGRESS_REPORT_INTERVAL: float = 5.0

# Default timeout values (seconds)
DEFAULT_TIMEOUT: float = 120.0
DEFAULT_POLL_INTERVAL: float = 0.5
WORKER_JOIN_TIMEOUT: float = 5.0

# Default worker counts
DEFAULT_WORKER_COUNT: int = 4

# Search defaults
DEFAULT_SEARCH_RESULTS: int = 5
DEFAULT_CONVERSATION_LIMIT: int = 10
DEFAULT_MESSAGE_LIMIT: int = 20


def _get_version() -> str:
    """Get the package version string.

    Returns:
        Version string in format 'word_forge VERSION'
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        return f"word_forge {version('word_forge')}"
    except (ImportError, PackageNotFoundError):
        return f"word_forge {__version__}"


def _setup_logging(level: str = "INFO") -> None:
    """Configure basic console logging."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format="%(asctime)s - %(levelname)s - %(message)s")


@eidosian()
def start(
    seed_words: Iterable[str] | None = None,
    run_minutes: float | None = None,
    worker_count: int = 4,
    db_path: str | None = None,
    vector_model: str | None = None,
    llm_model: str | None = None,
    enable_vector: bool = False,
    enable_llm: bool = False,
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
    db_path:
        Optional override for the SQLite database path.
    vector_model:
        Optional override for the sentence-transformer model used by vector storage
        and the vector worker embedder.
    llm_model:
        Optional override for the language model used to generate example sentences.
    enable_vector:
        Whether to launch the vector indexing worker. Disable for constrained smoke runs.
    enable_llm:
        Whether to create the language-model enrichment backend.
    """

    from word_forge.config import config
    from word_forge.configs.config_essentials import measure_execution
    from word_forge.database.database_manager import DBManager
    from word_forge.graph.graph_manager import GraphManager
    from word_forge.graph.graph_worker import GraphWorker
    from word_forge.parser.parser_refiner import ParserRefiner
    from word_forge.queue.queue_manager import QueueManager
    from word_forge.queue.queue_worker import (
        ParallelWordProcessor,
        WordProcessor,
        WorkerPoolConfig,
    )
    from word_forge.queue.worker_manager import WorkerManager

    _setup_logging()
    LOGGER.info("Starting Word Forge")

    # Initialize metrics
    metrics.initialize_metrics(port=getattr(config, "metrics_port", 8000))

    db_manager = DBManager(db_path=db_path)
    queue_manager: QueueManager[str] = QueueManager()
    queue_manager.start()
    parser_refiner = ParserRefiner(
        db_manager=db_manager,
        queue_manager=queue_manager,
        model_name=llm_model,
        enable_llm=enable_llm,
    )
    processor = WordProcessor(db_manager=db_manager, parser_refiner=parser_refiner, logger=LOGGER)
    pool_config = WorkerPoolConfig(worker_count=worker_count)
    worker_pool = ParallelWordProcessor(processor, config=pool_config, logger=LOGGER)

    graph_manager = GraphManager(db_manager=db_manager)
    graph_worker = GraphWorker(graph_manager=graph_manager)

    manager = WorkerManager(logger=LOGGER)
    manager.register(worker_pool)
    manager.register(graph_worker)
    if enable_vector:
        from word_forge.configs.config_essentials import StorageType
        from word_forge.vectorizer.vector_store import VectorStore
        from word_forge.vectorizer.vector_worker import VectorWorker

        vector_store = VectorStore(
            db_manager=db_manager,
            model_name=vector_model,
            storage_type=StorageType.MEMORY,
            demo_mode=True,
        )
        vector_worker = VectorWorker(
            db=db_manager,
            vector_store=vector_store,
            embedder=vector_model or "sentence-transformers/all-MiniLM-L6-v2",
        )
        manager.register(vector_worker)

    seeds = list(seed_words) if seed_words is not None else ["language", "knowledge", "system"]
    for term in seeds:
        queue_manager.enqueue(term)

    with measure_execution("forge.start", {"workers": worker_count}) as exec_metrics:
        manager.start_all()
        LOGGER.info(
            "Workers started in %.1fms",
            exec_metrics.duration_ms,
        )

    start_time = time.time()
    last_report = start_time
    try:
        while True:
            current_time = time.time()
            elapsed_min = (current_time - start_time) / 60.0

            if run_minutes is not None and elapsed_min >= run_minutes:
                LOGGER.info("Time limit reached (%.1f min). Shutting down.", run_minutes)
                break

            time.sleep(MAIN_LOOP_SLEEP_INTERVAL)

            # Autonomous health monitoring
            manager.monitor_health()

            # Update queue metrics
            metrics.update_queue_size(queue_manager.size)

            if current_time - last_report >= PROGRESS_REPORT_INTERVAL:
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
                if graph_status.get("last_new_nodes") or graph_status.get("last_new_edges"):
                    LOGGER.info(
                        "Graph updates - nodes:+%d edges:+%d state:%s",
                        graph_status.get("last_new_nodes", 0),
                        graph_status.get("last_new_edges", 0),
                        graph_status.get("state", "unknown"),
                    )
                last_report = time.time()

            if run_minutes is None and queue_manager.is_empty and not manager.any_alive():
                LOGGER.info("Queue empty and all workers finished. Shutting down.")
                break
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user")
    finally:
        LOGGER.info("Initiating graceful shutdown...")
        manager.stop_all()
        queue_manager.stop()
        parser_refiner.shutdown()
        db_manager.close()
        LOGGER.info("Word Forge stopped")


@eidosian()
def run_setup_nltk() -> int:
    """Ensure all required NLTK corpora are installed locally."""

    _setup_logging()
    LOGGER.info("Checking NLTK dependencies")
    try:
        from word_forge.utils.nltk_utils import ensure_nltk_data
    except ImportError:
        print('ERROR: NLTK support is not installed; run `pip install "word-forge[nlp]"`')
        return 2

    downloaded = ensure_nltk_data(logger=LOGGER)
    if downloaded:
        LOGGER.info("Downloaded NLTK corpora: %s", ", ".join(downloaded))
    else:
        LOGGER.info("NLTK corpora already installed; no downloads required")
    return 0


def _add_database_argument(parser: argparse.ArgumentParser) -> None:
    """Add the shared optional database argument to a command parser."""

    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        metavar="FILE",
        help="Use this SQLite lexicon instead of the user-data default",
    )


def _add_json_argument(parser: argparse.ArgumentParser) -> None:
    """Add the stable machine-readable output option to a command parser."""

    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Print UTF-8 JSON instead of the human-readable view",
    )


def _port_number(raw_value: str) -> int:
    """Parse a valid TCP port for argparse."""

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= value <= 65_535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return value


def _nonnegative_int(raw_value: str) -> int:
    """Parse a non-negative integer for optimistic revision flags."""

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("revision must be an integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("revision cannot be negative")
    return value


def _register_lexicon_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register the approachable local lexicon command group."""

    init_parser = subparsers.add_parser(
        "init",
        help="Create an empty local lexicon",
        description="Create the database schema and print the resulting file location.",
    )
    _add_database_argument(init_parser)
    _add_json_argument(init_parser)
    init_parser.set_defaults(local_action="init")

    add_parser = subparsers.add_parser(
        "add",
        help="Add or update one lexical entry",
    )
    add_parser.add_argument("term", help="Word or phrase to store")
    add_parser.add_argument("-d", "--definition", default="", help="Plain-language definition")
    add_parser.add_argument(
        "-p", "--part-of-speech", default="", help="For example: noun, verb, adjective"
    )
    add_parser.add_argument(
        "-l", "--language", default="en", help="Language tag, such as en, es, or zh-Hans"
    )
    add_parser.add_argument(
        "-e", "--example", action="append", default=[], help="Usage example; repeat for several"
    )
    add_parser.add_argument("--source", default="manual", help="Where this information came from")
    add_parser.add_argument(
        "--base-term", default="", help="English alignment for a non-English entry"
    )
    add_parser.add_argument(
        "--if-revision",
        type=_nonnegative_int,
        default=None,
        help="Write only if the current revision matches; use 0 for create-only",
    )
    _add_database_argument(add_parser)
    _add_json_argument(add_parser)
    add_parser.set_defaults(local_action="add")

    show_parser = subparsers.add_parser(
        "show",
        aliases=["lookup"],
        help="Show one exact lexical entry",
    )
    show_parser.add_argument("term", help="Word or phrase to retrieve")
    show_parser.add_argument("-l", "--language", default="en", help="Language tag; defaults to en")
    _add_database_argument(show_parser)
    _add_json_argument(show_parser)
    show_parser.set_defaults(local_action="show")

    find_parser = subparsers.add_parser(
        "find",
        aliases=["search"],
        help="Find entries by term or definition",
    )
    find_parser.add_argument("query", help="Text fragment to find")
    find_parser.add_argument("-l", "--language", default=None, help="Optional language tag")
    find_parser.add_argument(
        "-n", "--limit", type=int, default=20, help="Maximum results; defaults to 20"
    )
    _add_database_argument(find_parser)
    _add_json_argument(find_parser)
    find_parser.set_defaults(local_action="find")

    relate_parser = subparsers.add_parser(
        "relate",
        help="Connect two base-language terms",
    )
    relate_parser.add_argument("term", help="Source term already in the lexicon")
    relate_parser.add_argument("target", help="Related term; created if missing")
    relate_parser.add_argument(
        "--kind", default="related", help="Relationship type, such as synonym or hypernym"
    )
    _add_database_argument(relate_parser)
    _add_json_argument(relate_parser)
    relate_parser.set_defaults(local_action="relate")

    translate_parser = subparsers.add_parser(
        "translate",
        help="Connect equivalent terms in two languages",
    )
    translate_parser.add_argument("term", help="Source-language term")
    translate_parser.add_argument("target", help="Translated term")
    translate_parser.add_argument(
        "--from", dest="source_language", required=True, help="Source language tag"
    )
    translate_parser.add_argument(
        "--to", dest="target_language", required=True, help="Target language tag"
    )
    translate_parser.add_argument(
        "--source", default="manual", help="Where this translation came from"
    )
    translate_parser.add_argument(
        "--one-way",
        action="store_true",
        help="Store only source-to-target instead of both directions",
    )
    _add_database_argument(translate_parser)
    _add_json_argument(translate_parser)
    translate_parser.set_defaults(local_action="translate")

    sense_parser = subparsers.add_parser(
        "sense",
        help="Add or revise one independently reviewable meaning",
    )
    sense_parser.add_argument("term", help="Entry receiving the sense")
    sense_parser.add_argument("definition", help="Plain-language meaning")
    sense_parser.add_argument("-l", "--language", default="en", help="Language tag")
    sense_parser.add_argument("-p", "--part-of-speech", default="")
    sense_parser.add_argument("--domain", default="", help="For example: finance or botany")
    sense_parser.add_argument("--register", default="", help="For example: formal or slang")
    sense_parser.add_argument("--source", default="manual")
    sense_parser.add_argument("--confidence", type=float, default=1.0)
    sense_parser.add_argument(
        "--review-state",
        choices=["unreviewed", "reviewed", "rejected"],
        default="unreviewed",
    )
    sense_parser.add_argument("--sense-id", default=None, help="Existing sense UUID to revise")
    sense_parser.add_argument(
        "--if-revision",
        type=_nonnegative_int,
        default=None,
        help="Write only if this sense revision matches; use 0 for create-only",
    )
    _add_database_argument(sense_parser)
    _add_json_argument(sense_parser)
    sense_parser.set_defaults(local_action="sense")

    delete_sense_parser = subparsers.add_parser(
        "delete-sense",
        help="Tombstone a sense while preserving its audit history",
    )
    delete_sense_parser.add_argument("sense_id", help="Sense UUID to remove")
    delete_sense_parser.add_argument(
        "--if-revision",
        type=_nonnegative_int,
        default=None,
        help="Delete only if this revision still matches",
    )
    _add_database_argument(delete_sense_parser)
    _add_json_argument(delete_sense_parser)
    delete_sense_parser.set_defaults(local_action="delete-sense")

    stats_parser = subparsers.add_parser(
        "stats",
        help="Show exact lexicon record counts",
    )
    _add_database_argument(stats_parser)
    _add_json_argument(stats_parser)
    stats_parser.set_defaults(local_action="stats")

    integrity_parser = subparsers.add_parser(
        "integrity",
        help="Check database structure, foreign keys, and lifecycle queues",
    )
    _add_database_argument(integrity_parser)
    _add_json_argument(integrity_parser)
    integrity_parser.set_defaults(maintenance_action="integrity")

    backup_parser = subparsers.add_parser(
        "backup",
        help="Create an atomic, integrity-checked database snapshot",
    )
    backup_parser.add_argument("destination", type=Path, help="Snapshot file to create")
    _add_database_argument(backup_parser)
    backup_parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing snapshot; required for destructive overwrite",
    )
    _add_json_argument(backup_parser)
    backup_parser.set_defaults(maintenance_action="backup")

    restore_parser = subparsers.add_parser(
        "restore",
        help="Restore an integrity-checked snapshot",
    )
    restore_parser.add_argument("backup", type=Path, help="Snapshot file to restore")
    _add_database_argument(restore_parser)
    restore_parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing database; required for destructive overwrite",
    )
    _add_json_argument(restore_parser)
    restore_parser.set_defaults(maintenance_action="restore")


def _print_json(payload: object) -> None:
    """Write deterministic, multilingual-safe JSON to standard output."""

    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _print_entry(entry: object) -> None:
    """Print a compact human view of a public LexicalEntry."""

    from word_forge.models import LexicalEntry

    if not isinstance(entry, LexicalEntry):
        raise TypeError("entry must be a LexicalEntry")
    print(f"{entry.term} [{entry.language}] (revision {entry.revision})")
    print(f"ID: {entry.id}")
    if entry.part_of_speech:
        print(f"Part of speech: {entry.part_of_speech}")
    print(f"Definition: {entry.definition or '(not recorded)'}")
    if entry.base_term:
        print(f"Base term: {entry.base_term}")
    if entry.examples:
        print("Examples:")
        for example in entry.examples:
            print(f"  - {example}")
    if entry.relationships:
        print("Relationships:")
        for relationship in entry.relationships:
            language = f" [{relationship.language}]" if relationship.language else ""
            print(f"  - {relationship.kind}: {relationship.target}{language}")
    if entry.senses:
        print("Senses:")
        for sense in entry.senses:
            qualifiers = ", ".join(
                value for value in (sense.part_of_speech, sense.domain, sense.register) if value
            )
            suffix = f" ({qualifiers})" if qualifiers else ""
            print(f"  - {sense.definition}{suffix} [{sense.id}, revision {sense.revision}]")


def _run_local_lexicon_command(args: argparse.Namespace) -> int:
    """Execute a local lexicon command with stable errors and exit codes."""

    from word_forge.database.database_manager import (
        DatabaseError,
        RevisionConflictError,
        TermNotFoundError,
    )
    from word_forge.lexicon import Lexicon

    try:
        with Lexicon(args.db_path) as lexicon:
            action = args.local_action
            if action == "init":
                payload = {"database": str(lexicon.path), "created": True}
                if args.json_output:
                    _print_json(payload)
                else:
                    print(f"Word Forge lexicon ready: {lexicon.path}")
                return 0

            if action == "add":
                entry = lexicon.add(
                    args.term,
                    definition=args.definition,
                    part_of_speech=args.part_of_speech,
                    language=args.language,
                    examples=args.example,
                    source=args.source,
                    base_term=args.base_term,
                    expected_revision=args.if_revision,
                )
                _print_json(entry.as_dict()) if args.json_output else _print_entry(entry)
                return 0

            if action == "sense":
                sense = lexicon.add_sense(
                    args.term,
                    args.definition,
                    language=args.language,
                    part_of_speech=args.part_of_speech,
                    domain=args.domain,
                    register=args.register,
                    source=args.source,
                    confidence=args.confidence,
                    review_state=args.review_state,
                    sense_id=args.sense_id,
                    expected_revision=args.if_revision,
                )
                if args.json_output:
                    _print_json(sense.as_dict())
                else:
                    print(f"Sense stored: {sense.id} (revision {sense.revision})")
                return 0

            if action == "delete-sense":
                deleted = lexicon.remove_sense(
                    args.sense_id,
                    expected_revision=args.if_revision,
                )
                payload = {"sense_id": args.sense_id, "deleted": deleted}
                if args.json_output:
                    _print_json(payload)
                else:
                    print("Sense removed." if deleted else "Sense was not found.")
                return 0 if deleted else 3

            if action == "show":
                entry = lexicon.get(args.term, language=args.language)
                _print_json(entry.as_dict()) if args.json_output else _print_entry(entry)
                return 0

            if action == "find":
                entries = lexicon.find(args.query, language=args.language, limit=args.limit)
                if args.json_output:
                    _print_json([entry.as_dict() for entry in entries])
                elif not entries:
                    print(f"No entries matched {args.query!r}.")
                else:
                    for index, entry in enumerate(entries, start=1):
                        definition = entry.definition or "(not recorded)"
                        print(f"{index:>3}. {entry.term} [{entry.language}] - {definition}")
                return 0

            if action == "relate":
                created = lexicon.relate(args.term, args.target, kind=args.kind)
                payload = {
                    "term": args.term,
                    "target": args.target,
                    "kind": args.kind,
                    "created": created,
                }
                if args.json_output:
                    _print_json(payload)
                else:
                    verb = "Created" if created else "Already present"
                    print(f"{verb}: {args.term} --{args.kind}--> {args.target}")
                return 0

            if action == "translate":
                lexicon.translate(
                    args.term,
                    source_language=args.source_language,
                    target_language=args.target_language,
                    target_term=args.target,
                    source=args.source,
                    bidirectional=not args.one_way,
                )
                payload = {
                    "term": args.term,
                    "language": args.source_language,
                    "target": args.target,
                    "target_language": args.target_language,
                    "bidirectional": not args.one_way,
                }
                if args.json_output:
                    _print_json(payload)
                else:
                    print(
                        f"Translation stored: {args.term} [{args.source_language}] = "
                        f"{args.target} [{args.target_language}]"
                    )
                return 0

            if action == "stats":
                statistics = lexicon.statistics()
                if args.json_output:
                    _print_json(statistics.as_dict())
                else:
                    print(f"Database:      {lexicon.path}")
                    print(f"Words:         {statistics.words}")
                    print(f"Lexemes:       {statistics.lexemes}")
                    print(f"Languages:     {statistics.languages}")
                    print(f"Relationships: {statistics.relationships}")
                    print(f"Translations:  {statistics.translations}")
                return 0
    except TermNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    except RevisionConflictError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 4
    except (DatabaseError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"ERROR: unsupported lexicon action {args.local_action!r}", file=sys.stderr)
    return 2


def _default_database_path() -> Path:
    """Resolve the configured database without opening or mutating it."""

    from word_forge.paths import DATABASE_PATH

    return DATABASE_PATH


def _run_maintenance_command(args: argparse.Namespace) -> int:
    """Execute database maintenance without leaving an open application connection."""

    from word_forge.maintenance import backup_database, inspect_database, restore_database

    database_path = (
        Path(args.db_path).expanduser().resolve() if args.db_path else _default_database_path()
    )
    try:
        if args.maintenance_action == "integrity":
            report = inspect_database(database_path)
            if args.json_output:
                _print_json(report.as_dict())
            else:
                status = "healthy" if report.ok else "unhealthy"
                print(f"Database: {report.database}")
                print(f"Status: {status}")
                print(f"SQLite integrity: {report.integrity}")
                print(f"Schema version: {report.schema_version}")
                print(f"Foreign-key violations: {report.foreign_key_violations}")
                print(f"Pending invalidations: {report.pending_invalidations}")
                print(f"Pending jobs: {report.pending_jobs}")
            return 0 if report.ok else 1
        if args.maintenance_action == "backup":
            destination = backup_database(database_path, args.destination, force=args.force)
            payload = {"database": str(database_path), "backup": str(destination)}
            _print_json(payload) if args.json_output else print(f"Backup created: {destination}")
            return 0
        if args.maintenance_action == "restore":
            destination = restore_database(args.backup, database_path, force=args.force)
            payload = {"backup": str(args.backup), "database": str(destination)}
            _print_json(payload) if args.json_output else print(f"Database restored: {destination}")
            return 0
    except FileExistsError as exc:
        print(f"ERROR: {str(exc).replace('force=True', '--force')}", file=sys.stderr)
        return 2
    except (OSError, RuntimeError, sqlite3.Error, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"ERROR: unsupported maintenance action {args.maintenance_action!r}", file=sys.stderr)
    return 2


def run_doctor(*, json_output: bool = False) -> int:
    """Inspect local prerequisites without downloading or changing user data."""

    from word_forge.paths import DATA_ROOT, LOGS_ROOT

    checks: list[dict[str, object]] = []

    def record(name: str, ok: bool, detail: str, *, required: bool) -> None:
        checks.append({"name": name, "ok": ok, "required": required, "detail": detail})

    try:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        LOGS_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix="doctor-", dir=DATA_ROOT):
            pass
        record("data-directory", True, str(DATA_ROOT), required=True)
    except OSError as exc:
        record("data-directory", False, str(exc), required=True)

    try:
        connection = sqlite3.connect(":memory:")
        try:
            connection.execute("CREATE VIRTUAL TABLE search_probe USING fts5(content)")
        finally:
            connection.close()
        record("sqlite-fts5", True, sqlite3.sqlite_version, required=True)
    except sqlite3.Error as exc:
        record("sqlite-fts5", False, str(exc), required=True)

    nltk_available = importlib.util.find_spec("nltk") is not None
    record(
        "optional-nlp",
        nltk_available,
        "installed" if nltk_available else 'install with `pip install "word-forge[nlp]"`',
        required=False,
    )
    if nltk_available:
        import nltk

        for resource, lookup_path in (
            ("wordnet", "corpora/wordnet"),
            ("omw-1.4", "corpora/omw-1.4"),
            ("vader_lexicon", "sentiment/vader_lexicon.zip"),
        ):
            try:
                nltk.data.find(lookup_path)
                record(f"nltk-{resource}", True, "installed", required=False)
            except LookupError:
                record(
                    f"nltk-{resource}",
                    False,
                    "not installed; run `word-forge setup-nltk` when needed",
                    required=False,
                )

    for extra, module_name in (
        ("graph", "networkx"),
        ("gui", "fastapi"),
        ("vector", "sentence_transformers"),
        ("visualization", "pyvis"),
        ("llm", "transformers"),
    ):
        available = importlib.util.find_spec(module_name) is not None
        record(
            f"optional-{extra}",
            available,
            "installed" if available else f'install with `pip install "word-forge[{extra}]"`',
            required=False,
        )

    success = all(bool(item["ok"]) for item in checks if item["required"])
    if json_output:
        _print_json({"ok": success, "checks": checks})
    else:
        print("Word Forge doctor")
        for check in checks:
            marker = "OK" if check["ok"] else ("ERROR" if check["required"] else "INFO")
            print(f"[{marker:>5}] {check['name']}: {check['detail']}")
    return 0 if success else 1


def run_gui(
    *,
    database_path: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> int:
    """Launch the secured, loopback-only Word Forge workspace."""

    try:
        from word_forge.web.server import serve

        return serve(
            database_path=database_path,
            host=host,
            port=port,
            open_browser=open_browser,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


@eidosian()
def main(argv: list[str] | None = None) -> int:
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
        help="Path to a JSON or TOML configuration file",
    )
    subparsers = parser.add_subparsers(dest="command")

    _register_lexicon_commands(subparsers)

    gui_parser = subparsers.add_parser(
        "gui",
        aliases=["serve"],
        help="Open the secure local Word Forge workspace",
    )
    _add_database_argument(gui_parser)
    gui_parser.add_argument(
        "--host",
        default="127.0.0.1",
        choices=["127.0.0.1", "localhost", "::1"],
        help="Loopback address to bind; defaults to 127.0.0.1",
    )
    gui_parser.add_argument(
        "--port",
        type=_port_number,
        default=8765,
        help="Local TCP port; defaults to 8765",
    )
    gui_parser.add_argument(
        "--no-open",
        action="store_true",
        help="Print the authenticated URL without opening a browser",
    )

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
    start_parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Override the default SQLite database path",
    )
    start_parser.add_argument(
        "--vector-model",
        type=str,
        default=None,
        help="Override the default sentence-transformer model for vector storage/indexing",
    )
    start_parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Override the default language model for example sentence generation",
    )
    vector_feature_group = start_parser.add_mutually_exclusive_group()
    vector_feature_group.add_argument(
        "--with-vector",
        action="store_true",
        help="Enable optional neural vector indexing (may download a model)",
    )
    vector_feature_group.add_argument(
        "--no-vector",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    llm_feature_group = start_parser.add_mutually_exclusive_group()
    llm_feature_group.add_argument(
        "--with-llm",
        action="store_true",
        help="Enable optional language-model enrichment (may download a model)",
    )
    llm_feature_group.add_argument(
        "--no-llm",
        action="store_true",
        help=argparse.SUPPRESS,
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
    graph_build.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Override the default SQLite database path",
    )

    graph_visualize = graph_sub.add_parser("visualize", help="Generate a graph visualization")
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

    vector_parser = subparsers.add_parser("vector", help="Vector index management commands")
    vector_sub = vector_parser.add_subparsers(dest="vector_command")
    vector_index = vector_sub.add_parser(
        "index", help="Run the vector worker until one cycle completes"
    )
    vector_index.add_argument(
        "--embedder",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformer model name (HuggingFace model path)",
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
    vector_index.add_argument(
        "--db-path",
        default=None,
        help="Override the lexicon SQLite database path for this indexing pass",
    )

    # Vector search command
    vector_search = vector_sub.add_parser(
        "search", help="Search the vector index for similar terms"
    )
    vector_search.add_argument(
        "query",
        nargs="+",
        help="Query text to search for",
    )
    vector_search.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    vector_search.add_argument(
        "--content-type",
        choices=["word", "definition", "example", "all"],
        default="all",
        help="Filter by content type (default: all)",
    )

    # Conversation commands
    conversation_parser = subparsers.add_parser(
        "conversation", help="Conversation management commands"
    )
    conversation_sub = conversation_parser.add_subparsers(dest="conversation_command")

    conversation_start = conversation_sub.add_parser("start", help="Start a new conversation")
    conversation_start.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title for the conversation",
    )

    conversation_list = conversation_sub.add_parser("list", help="List all conversations")
    conversation_list.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of conversations to list (default: 10)",
    )

    conversation_show = conversation_sub.add_parser("show", help="Show messages in a conversation")
    conversation_show.add_argument(
        "conversation_id",
        type=int,
        help="ID of the conversation to show",
    )
    conversation_show.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of messages to show (default: 20)",
    )

    emotion_parser = subparsers.add_parser("emotion", help="Emotion annotation utilities")
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
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check local runtime and optional features",
    )
    _add_json_argument(doctor_parser)

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1

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

    if args.command in {"gui", "serve"}:
        exit_code = run_gui(
            database_path=args.db_path,
            host=args.host,
            port=args.port,
            open_browser=not args.no_open,
        )
    elif hasattr(args, "maintenance_action"):
        exit_code = _run_maintenance_command(args)
    elif hasattr(args, "local_action"):
        exit_code = _run_local_lexicon_command(args)
    elif args.command == "doctor":
        exit_code = run_doctor(json_output=args.json_output)
    elif args.command == "start":
        start(
            args.words,
            run_minutes=args.minutes,
            worker_count=args.workers,
            db_path=args.db_path,
            vector_model=args.vector_model,
            llm_model=args.llm_model,
            enable_vector=args.with_vector and not args.no_vector,
            enable_llm=args.with_llm and not args.no_llm,
        )
    elif args.command == "graph":
        if args.graph_command == "build":
            exit_code = (
                0
                if run_graph_build(
                    db_path=args.db_path, poll_interval=args.poll_interval, timeout=args.timeout
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
            vector_db = None
            if args.db_path:
                from word_forge.database.database_manager import DBManager

                vector_db = DBManager(db_path=Path(args.db_path))
            try:
                exit_code = (
                    0
                    if run_vector_index(
                        db_manager=vector_db,
                        embedder=args.embedder,
                        poll_interval=args.poll_interval,
                        timeout=args.timeout,
                    )
                    else 1
                )
            finally:
                if vector_db is not None:
                    vector_db.close()
        elif args.vector_command == "search":
            query_text = " ".join(args.query)
            content_type = None if args.content_type == "all" else args.content_type
            exit_code = (
                0
                if run_vector_search(
                    query=query_text,
                    k=args.top_k,
                    content_type=content_type,
                )
                else 1
            )
        else:
            vector_parser.print_help()
            exit_code = 1
    elif args.command == "conversation":
        if args.conversation_command == "start":
            exit_code = 0 if run_conversation_start(title=args.title) else 1
        elif args.conversation_command == "list":
            exit_code = 0 if run_conversation_list(limit=args.limit) else 1
        elif args.conversation_command == "show":
            exit_code = (
                0
                if run_conversation_show(conversation_id=args.conversation_id, limit=args.limit)
                else 1
            )
        else:
            conversation_parser.print_help()
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


@eidosian()
def run_graph_build(
    *,
    graph_manager: GraphManager | None = None,
    db_path: str | None = None,
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
    db_manager: DBManager | None = None
    if graph_manager is None:
        db_manager = DBManager(db_path=db_path)
        graph_manager = GraphManager(db_manager=db_manager)
    else:
        db_manager = graph_manager.db_manager

    worker = GraphWorker(
        graph_manager=graph_manager,
        poll_interval=poll_interval,
        daemon=False,
        generate_visualization=False,
    )
    manager = WorkerManager(logger=LOGGER)
    manager.register(worker)
    LOGGER.info("Starting graph build worker")

    try:
        manager.start_all()
        completed = _wait_for_condition(
            "graph build",
            lambda: worker.get_status()["update_count"] > 0 or graph_manager.get_node_count() > 0,
            timeout=timeout,
        )
        return completed
    finally:
        manager.stop_all()
        worker.join(timeout=5)
        if owns_manager and db_manager is not None:
            db_manager.close()


@eidosian()
def run_graph_visualization(
    *,
    graph_manager: GraphManager | None = None,
    output_path: str | None = None,
    use_3d: bool = False,
    open_in_browser: bool = False,
) -> bool:
    """Build the graph (if needed) and emit a visualization file."""

    _setup_logging()
    from word_forge.database.database_manager import DBManager
    from word_forge.graph.graph_manager import GraphManager

    owns_manager = graph_manager is None
    db_manager: DBManager | None = None
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


@eidosian()
def run_vector_index(
    *,
    db_manager: DBManager | None = None,
    embedder: str = "sentence-transformers/all-MiniLM-L6-v2",
    poll_interval: float = 0.25,
    timeout: float = 120.0,
) -> bool:
    """Run one synchronous vector-indexing pass over the current lexicon."""

    _setup_logging()
    from word_forge.database.database_manager import DBManager

    owns_db = db_manager is None
    db = db_manager or DBManager()
    db.create_tables()

    raw_words = db.get_all_words() if hasattr(db, "get_all_words") else None
    if raw_words == []:
        LOGGER.info("Vector indexing skipped: no words require indexing")
        if owns_db:
            db.close()
        return True

    from word_forge.vectorizer.vector_store import SQLITE_DB_FILENAME, VectorStore
    from word_forge.vectorizer.vector_worker import VectorWorker

    vector_store = VectorStore(db_manager=db, model_name=embedder)
    worker = VectorWorker(
        db=db,
        vector_store=vector_store,
        embedder=embedder,
        poll_interval=poll_interval,
        daemon=False,
        logger=LOGGER,
    )
    LOGGER.info("Starting synchronous vector indexing pass")

    try:
        started_at = time.time()
        worker.stats.clear()
        words = worker._get_all_words()
        existing_original_ids: set[int] = set()
        if str(getattr(vector_store, "backend_name", "")).startswith("sqlite-"):
            sqlite_path = vector_store.index_path / SQLITE_DB_FILENAME
            if sqlite_path.exists():
                try:
                    with sqlite3.connect(str(sqlite_path)) as conn:
                        rows = conn.execute("SELECT metadata FROM vectors").fetchall()
                    for (metadata_blob,) in rows:
                        if not metadata_blob:
                            continue
                        try:
                            metadata = json.loads(str(metadata_blob))
                        except json.JSONDecodeError:
                            continue
                        original_id = metadata.get("original_id")
                        if isinstance(original_id, int):
                            existing_original_ids.add(original_id)
                except Exception as exc:
                    LOGGER.warning(
                        "Failed to inspect existing vector index for incremental skip: %s", exc
                    )

        if existing_original_ids:
            words = [word for word in words if int(word.id) not in existing_original_ids]

        total = len(words)
        if total == 0:
            worker.last_processed = time.time()
            LOGGER.info("Vector indexing skipped: no new words require indexing")
            return True

        chunk_size = 100
        for start in range(0, total, chunk_size):
            if time.time() - started_at > timeout:
                LOGGER.warning(
                    "Vector indexing timed out after %.2fs at %s/%s words",
                    time.time() - started_at,
                    min(start, total),
                    total,
                )
                return False
            chunk = words[start : start + chunk_size]
            worker._process_words(chunk)
            LOGGER.info(
                "Vector indexing progress: %s/%s words (%s successful, %s failed)",
                min(start + len(chunk), total),
                total,
                worker.stats.successful,
                worker.stats.failed,
            )

        worker.last_processed = time.time()
        LOGGER.info(
            "Vector indexing completed in %.2fs: %s/%s successful, %s failed",
            worker.last_processed - started_at,
            worker.stats.successful,
            total,
            worker.stats.failed,
        )
        return worker.stats.failed == 0
    finally:
        if owns_db:
            db.close()


def _remaining_unemotioned_words(db_manager: DBManager) -> int:
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


@eidosian()
def run_emotion_annotation(
    *,
    db_manager: DBManager | None = None,
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


@eidosian()
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

        vector_ok = run_vector_index(db_manager=db_manager, poll_interval=0.25, timeout=timeout)
        graph_ok = run_graph_build(graph_manager=graph_manager, poll_interval=0.25, timeout=timeout)
        viz_ok = run_graph_visualization(
            graph_manager=graph_manager,
            use_3d=use_3d,
            open_in_browser=open_in_browser,
        )
        return vector_ok and graph_ok and viz_ok
    finally:
        db_manager.close()


@eidosian()
def run_vector_search(
    *,
    query: str,
    k: int = 5,
    content_type: str | None = None,
) -> bool:
    """Search the vector index for terms similar to the query.

    Parameters
    ----------
    query:
        The search query text.
    k:
        Number of results to return.
    content_type:
        Optional filter for content type (word, definition, example).

    Returns
    -------
    bool
        True if search completed successfully, False otherwise.
    """
    _setup_logging()
    from word_forge.database.database_manager import DBManager
    from word_forge.vectorizer.vector_store import VectorStore

    db_manager = DBManager()
    try:
        db_manager.create_tables()
        vector_store = VectorStore(db_manager=db_manager)

        LOGGER.info("Searching for: '%s' (top %d results)", query, k)

        # Prepare filter metadata if content_type specified
        filter_metadata = None
        if content_type:
            filter_metadata = {"content_type": content_type}

        try:
            results = vector_store.search(
                query_text=query,
                k=k,
                filter_metadata=filter_metadata,
            )

            if not results:
                print(f"No results found for query: '{query}'")
                return True

            print(f"\nSearch Results for '{query}':")
            print("-" * 60)
            for i, result in enumerate(results, 1):
                distance = result.get("distance", 0.0)
                metadata = result.get("metadata") or {}
                text = result.get("text") or ""

                term = metadata.get("term", "")
                definition = metadata.get("definition", "")
                ctype = metadata.get("content_type", "")

                print(f"\n{i}. {term or text[:50]}")
                print(f"   Type: {ctype} | Distance: {distance:.4f}")
                if definition:
                    print(f"   Definition: {definition[:100]}")

            print("-" * 60)
            return True

        except Exception as e:
            LOGGER.error("Vector search failed: %s", e)
            print(f"Search error: {e}")
            return False

    finally:
        db_manager.close()


@eidosian()
def run_conversation_start(*, title: str | None = None) -> bool:
    """Start a new conversation session.

    Parameters
    ----------
    title:
        Optional title for the conversation.

    Returns
    -------
    bool
        True if conversation was created successfully, False otherwise.
    """
    _setup_logging()
    from word_forge.database.database_manager import DBManager

    db_manager = DBManager()
    try:
        db_manager.create_tables()

        # Create conversation tables if they don't exist
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    status TEXT DEFAULT 'ACTIVE' NOT NULL,
                    created_at REAL DEFAULT (strftime('%s','now')) NOT NULL,
                    updated_at REAL DEFAULT (strftime('%s','now')) NOT NULL
                );
                """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    speaker TEXT NOT NULL,
                    text TEXT NOT NULL,
                    timestamp REAL DEFAULT (strftime('%s','now')) NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                """)

            cursor.execute("INSERT INTO conversations (status) VALUES ('ACTIVE');")
            conv_id = cursor.lastrowid
            conn.commit()

            print(f"Started new conversation with ID: {conv_id}")
            if title:
                print(f"Title: {title}")
            return True

    except Exception as e:
        LOGGER.error("Failed to start conversation: %s", e)
        print(f"Error: {e}")
        return False
    finally:
        db_manager.close()


@eidosian()
def run_conversation_list(*, limit: int = 10) -> bool:
    """List recent conversations.

    Parameters
    ----------
    limit:
        Maximum number of conversations to list.

    Returns
    -------
    bool
        True if listing completed successfully, False otherwise.
    """
    _setup_logging()
    from word_forge.database.database_manager import DBManager

    db_manager = DBManager()
    try:
        db_manager.create_tables()

        with db_manager.get_connection() as conn:
            import sqlite3

            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    c.id,
                    c.status,
                    datetime(c.created_at, 'unixepoch') as created_at,
                    COUNT(cm.id) as message_count
                FROM conversations c
                LEFT JOIN conversation_messages cm ON c.id = cm.conversation_id
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT ?;
                """,
                (limit,),
            )
            rows = cursor.fetchall()

            if not rows:
                print("No conversations found.")
                return True

            print(f"\nConversations (showing up to {limit}):")
            print("-" * 60)
            for row in rows:
                print(
                    f"  ID: {row['id']} | Messages: {row['message_count']} | Created: {row['created_at']}"
                )
            print("-" * 60)
            return True

    except Exception as e:
        LOGGER.error("Failed to list conversations: %s", e)
        print(f"Error: {e}")
        return False
    finally:
        db_manager.close()


@eidosian()
def run_conversation_show(*, conversation_id: int, limit: int = 20) -> bool:
    """Show messages in a conversation.

    Parameters
    ----------
    conversation_id:
        ID of the conversation to show.
    limit:
        Maximum number of messages to display.

    Returns
    -------
    bool
        True if display completed successfully, False otherwise.
    """
    _setup_logging()
    from word_forge.database.database_manager import DBManager

    db_manager = DBManager()
    try:
        db_manager.create_tables()

        with db_manager.get_connection() as conn:
            import sqlite3

            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # First verify conversation exists
            cursor.execute("SELECT id FROM conversations WHERE id = ?;", (conversation_id,))
            if cursor.fetchone() is None:
                print(f"Conversation {conversation_id} not found.")
                return False

            cursor.execute(
                """
                SELECT
                    speaker as role,
                    text as content,
                    datetime(timestamp, 'unixepoch') as created_at
                FROM conversation_messages
                WHERE conversation_id = ?
                ORDER BY timestamp ASC
                LIMIT ?;
                """,
                (conversation_id, limit),
            )
            rows = cursor.fetchall()

            if not rows:
                print(f"No messages found in conversation {conversation_id}.")
                return True

            print(f"\nConversation {conversation_id} (showing up to {limit} messages):")
            print("-" * 60)
            for row in rows:
                role_display = "User" if row["role"] == "user" else "Assistant"
                print(f"\n[{role_display}] ({row['created_at']})")
                print(f"  {row['content']}")
            print("-" * 60)
            return True

    except Exception as e:
        LOGGER.error("Failed to get messages: %s", e)
        print(f"Error: {e}")
        return False
    finally:
        db_manager.close()


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main(sys.argv[1:]))
