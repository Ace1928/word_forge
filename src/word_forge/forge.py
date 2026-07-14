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
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, List, Optional

from word_forge.vectorizer.embedding_models import DEFAULT_EMBEDDING_MODEL

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from word_forge.database.database_manager import DBManager
    from word_forge.graph.graph_config import RelationshipDimension
    from word_forge.graph.graph_manager import GraphManager


LOGGER = logging.getLogger("word_forge")

# Package version - dynamically retrieved from package metadata
__version__ = "0.1.0"

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
GRAPH_DIMENSION_CHOICES = (
    "lexical",
    "emotional",
    "affective",
    "connotative",
    "contextual",
)


def _positive_int(value: str) -> int:
    """Parse a strictly positive command-line integer."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _non_negative_int(value: str) -> int:
    """Parse a non-negative command-line integer."""

    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value cannot be negative")
    return parsed


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
    logging.basicConfig(
        level=numeric_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def start(
    seed_words: Optional[Iterable[str]] = None,
    run_minutes: Optional[float] = None,
    worker_count: int = 4,
    db_path: Optional[str] = None,
    vector_model: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_profile: Optional[str] = None,
    enable_vector: Optional[bool] = None,
    language: Optional[str] = None,
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
    llm_profile:
        Optional named local-model profile (``portable``, ``gemma3-tiny``,
        ``gemma4-edge``, ``auto``, or ``off``).
    enable_vector:
        ``True`` requires vector indexing, ``False`` disables it, and ``None``
        enables it when the optional vector dependencies are available.
    language:
        BCP 47 language tag applied to seeds and same-language discoveries.
    """

    from word_forge.configs.config_essentials import measure_execution
    from word_forge.database.database_manager import DBManager
    from word_forge.graph.graph_manager import GraphManager
    from word_forge.graph.graph_worker import GraphWorker
    from word_forge.parser.linguistics import canonicalize_language_tag
    from word_forge.parser.parser_refiner import ParserRefiner
    from word_forge.queue.queue_manager import QueueManager
    from word_forge.queue.queue_worker import (
        ParallelWordProcessor,
        WordProcessor,
        WorkerPoolConfig,
    )
    from word_forge.queue.worker_manager import WorkerManager

    _setup_logging()
    from word_forge.config import config

    selected_language = canonicalize_language_tag(
        language or config.parser.default_language
    )
    seeds = list(seed_words) if seed_words is not None else []
    if not seeds:
        if selected_language.split("-", 1)[0] != "en":
            raise ValueError(
                "Explicit seed words are required when --language is not English"
            )
        seeds = ["language", "knowledge", "system"]
    LOGGER.info("Starting Word Forge for language %s", selected_language)

    requested_profile = llm_profile
    if llm_model is None and requested_profile is None:
        if config.parser.enable_model and config.parser.model_name is None:
            requested_profile = config.parser.model_profile

    if requested_profile is not None:
        from word_forge.parser.model_profiles import (
            detect_runtime_resources,
            resolve_model_profile,
        )

        if requested_profile.strip().casefold() == "off":
            selected_profile = resolve_model_profile(
                requested_profile, require_ready=True
            )
        else:
            runtime = detect_runtime_resources()
            selected_profile = resolve_model_profile(
                requested_profile,
                runtime,
                require_ready=True,
            )
            for warning in selected_profile.warnings(runtime):
                LOGGER.warning("Model profile '%s': %s", selected_profile.name, warning)
        llm_profile = selected_profile.name

    db_manager = DBManager(db_path=db_path)
    queue_manager: QueueManager[str] = QueueManager()
    queue_manager.start()
    parser_refiner = ParserRefiner(
        db_manager=db_manager,
        queue_manager=queue_manager,
        model_name=llm_model,
        model_profile=llm_profile,
        language=selected_language,
    )
    processor = WordProcessor(
        db_manager=db_manager, parser_refiner=parser_refiner, logger=LOGGER
    )
    pool_config = WorkerPoolConfig(worker_count=worker_count)
    worker_pool = ParallelWordProcessor(processor, config=pool_config, logger=LOGGER)

    graph_manager = GraphManager(db_manager=db_manager)
    graph_worker = GraphWorker(graph_manager=graph_manager)

    vector_worker = None
    if enable_vector is not False:
        from word_forge.vectorizer.vector_store import VectorStore, VectorStoreError
        from word_forge.vectorizer.vector_worker import VectorWorker

        try:
            vector_store = VectorStore(db_manager=db_manager, model_name=vector_model)
            vector_worker = VectorWorker(
                db=db_manager,
                vector_store=vector_store,
            )
        except VectorStoreError as exc:
            vector_required = enable_vector is True or vector_model is not None
            if vector_required:
                raise
            LOGGER.info(
                "Vector indexing unavailable; continuing with lexical and graph "
                "workers (%s)",
                exc,
            )

    manager = WorkerManager(logger=LOGGER)
    manager.register(worker_pool)
    manager.register(graph_worker)
    if vector_worker is not None:
        manager.register(vector_worker)

    for term in seeds:
        queue_manager.enqueue(term)

    with measure_execution(
        "forge.start", {"workers": worker_count, "language": selected_language}
    ) as metrics:
        manager.start_all()
        LOGGER.info(
            "Workers started in %.1fms",
            metrics.duration_ms,
        )

    start_time = time.time()
    last_report = start_time
    try:
        while True:
            time.sleep(MAIN_LOOP_SLEEP_INTERVAL)
            if time.time() - last_report >= PROGRESS_REPORT_INTERVAL:
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
            if (
                run_minutes is None
                and queue_manager.is_empty
                and not manager.any_alive()
            ):
                break
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user")
    finally:
        manager.stop_all()
        try:
            parser_refiner.shutdown()
            final_graph_metrics = graph_worker.refresh()
            LOGGER.info(
                "Final graph refresh complete: +%d nodes, +%d edges",
                final_graph_metrics.new_nodes,
                final_graph_metrics.new_edges,
            )
        except Exception as exc:
            LOGGER.error("Final graph refresh failed: %s", exc, exc_info=True)
        finally:
            queue_manager.stop()
            db_manager.close()
        LOGGER.info("Word Forge stopped")


def run_setup_nltk(
    *, include_multilingual: bool = False, accept_source_licenses: bool = False
) -> int:
    """Ensure all required NLTK corpora are installed locally."""

    _setup_logging()
    LOGGER.info("Checking NLTK dependencies")
    from word_forge.utils.nltk_utils import (
        LexicalDataLicenseError,
        ensure_nltk_data,
    )

    try:
        downloaded = ensure_nltk_data(
            logger=LOGGER,
            include_multilingual=include_multilingual,
            accept_source_licenses=accept_source_licenses,
        )
    except LexicalDataLicenseError as exc:
        LOGGER.error("Unable to install multilingual lexical data: %s", exc)
        return 2
    if downloaded:
        LOGGER.info("Downloaded NLTK corpora: %s", ", ".join(downloaded))
    else:
        LOGGER.info("NLTK corpora already installed; no downloads required")
    return 0


def run_model_catalog(action: str = "list", json_output: bool = False) -> int:
    """List local-model profiles or recommend one for the current runtime."""
    from word_forge.parser.model_profiles import (
        detect_runtime_resources,
        iter_model_profiles,
        recommend_model_profile,
    )

    resources = detect_runtime_resources()
    profiles = list(iter_model_profiles())
    recommended = recommend_model_profile(resources)
    report = {
        "runtime": resources.to_dict(),
        "recommended": recommended.name,
        "profiles": [profile.to_dict(resources) for profile in profiles],
    }

    if json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    print(
        "Runtime: "
        f"{resources.accelerator}, {resources.available_ram_gib:.1f} GiB available / "
        f"{resources.total_ram_gib:.1f} GiB total, {resources.cpu_threads} CPU threads"
    )
    print(f"Recommended profile: {recommended.name} ({recommended.display_name})")
    if action == "recommend":
        if recommended.model_id is None:
            print("Install word_forge[llm] to enable optional generative enrichment.")
        else:
            print(f"Model: {recommended.model_id}")
            print(
                "Run: word_forge start --llm-profile "
                f"{recommended.name} <seed words>"
            )
        return 0

    print()
    print("PROFILE         READY  RAM MIN/REC  MODEL")
    for profile in profiles:
        ready, _issues = profile.readiness(resources)
        ready_label = "yes*" if ready and profile.warnings(resources) else "yes"
        if not ready:
            ready_label = "no"
        model_id = profile.model_id or "disabled"
        print(
            f"{profile.name:<15} {ready_label:<6} "
            f"{profile.minimum_available_ram_gib:g}/{profile.recommended_available_ram_gib:g} GiB  "
            f"{model_id}"
        )
    print("\n* Ready with operational warnings; use --json for details.")
    return 0


def run_source_catalog(
    *, json_output: bool = False, unattended_only: bool = False
) -> int:
    """Render the governed lexical-source catalog."""

    from word_forge.sources import source_catalog_report

    report = source_catalog_report(unattended_only=unattended_only)
    if json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    sources = report["sources"]
    if not isinstance(sources, list):  # pragma: no cover - internal invariant
        raise TypeError("Lexical source catalog must contain a source list")
    print("SOURCE                         STATUS       BOOTSTRAP             LICENSE")
    for item in sources:
        if not isinstance(item, dict):  # pragma: no cover - internal invariant
            raise TypeError("Lexical source entries must be mappings")
        license_data = item.get("license", {})
        if not isinstance(license_data, dict):  # pragma: no cover
            raise TypeError("Lexical source license metadata must be a mapping")
        print(
            f"{str(item['id']):<30} "
            f"{str(item['integration_status']):<12} "
            f"{str(item['bootstrap_tier']):<21} "
            f"{str(license_data.get('name', 'unknown'))}"
        )
    print(f"\n{report['notice']}")
    return 0


def run_kaikki_import(
    artifact_path: Path,
    *,
    source_version: str,
    source_url: str,
    accept_source_license: bool,
    db_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    disable_checkpoint: bool = False,
    expected_sha256: Optional[str] = None,
    batch_size: int = 500,
    languages: Iterable[str] = (),
    max_entries: Optional[int] = None,
    json_output: bool = False,
) -> int:
    """Inspect and import one local Kaikki artifact through the public CLI path."""

    from word_forge.database.database_manager import DatabaseError, DBManager
    from word_forge.database.lexical_repository import LexicalRepository
    from word_forge.sources.kaikki import (
        KaikkiImporter,
        KaikkiImportError,
        inspect_artifact,
    )

    if not accept_source_license:
        LOGGER.error(
            "Kaikki contains Wiktionary-derived data. Review the exact source "
            "terms, then pass --accept-source-license."
        )
        return 2

    database: Optional[DBManager] = None
    try:
        artifact = inspect_artifact(
            artifact_path,
            expected_sha256=expected_sha256,
        )
        effective_checkpoint = None
        if not disable_checkpoint:
            effective_checkpoint = checkpoint_path or artifact.path.with_name(
                f"{artifact.path.name}.word-forge.checkpoint.json"
            )

        database = DBManager(db_path=db_path)
        importer = KaikkiImporter(
            LexicalRepository(database),
            source_version=source_version,
            source_url=source_url,
            accept_source_license=True,
            batch_size=batch_size,
            languages=tuple(languages),
        )
        report = importer.import_artifact(
            artifact,
            checkpoint_path=effective_checkpoint,
            max_entries=max_entries,
        )
        payload = {
            "schema_version": 1,
            "source_id": "kaikki-wiktionary",
            "database_path": str(database.db_path.expanduser().resolve()),
            "checkpoint_path": (
                str(effective_checkpoint.expanduser().resolve())
                if effective_checkpoint is not None
                else None
            ),
            "report": report.to_dict(),
        }
        if json_output:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            writes = report.write_report
            print(
                f"Imported {writes.attempted:,} entries "
                f"({writes.inserted:,} inserted, {writes.updated:,} updated) "
                f"in {report.elapsed_seconds:.2f}s."
            )
            print(
                f"Read {report.lines_read:,} lines in {report.batches:,} batches; "
                f"skipped {report.skipped_entries:,}."
            )
            print(f"Snapshot: {report.snapshot_id}")
            print(f"SHA-256: {report.artifact_sha256}")
            print(f"Database: {payload['database_path']}")
            if effective_checkpoint is not None:
                print(f"Checkpoint: {payload['checkpoint_path']}")
        return 0
    except ValueError as exc:
        LOGGER.error("Invalid Kaikki import options: %s", exc)
        return 2
    except KaikkiImportError as exc:
        location = ""
        if exc.line_number is not None:
            location = (
                f" (line {exc.line_number}, committed through "
                f"{exc.committed_through})"
            )
        LOGGER.error("Kaikki import failed%s: %s", location, exc)
        return 1
    except DatabaseError as exc:
        LOGGER.error("Kaikki database operation failed: %s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.warning("Kaikki import interrupted; resume with the same checkpoint")
        return 130
    finally:
        if database is not None:
            database.close()


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
    start_parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Override the default SQLite database path",
    )
    start_parser.add_argument(
        "--language",
        type=str,
        default=None,
        metavar="BCP47",
        help="Language tag for seeds and recursive expansion (default: en)",
    )
    start_parser.add_argument(
        "--vector-model",
        type=str,
        default=None,
        help="Override the default sentence-transformer model for vector storage/indexing",
    )
    start_parser.add_argument(
        "--vector",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable vector indexing (default: enable when available)",
    )
    llm_group = start_parser.add_mutually_exclusive_group()
    llm_group.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Use a custom Hugging Face language model for missing examples",
    )
    llm_group.add_argument(
        "--llm-profile",
        choices=["auto", "off", "portable", "gemma3-tiny", "gemma4-edge"],
        default=None,
        help="Use a resource profile for optional generative enrichment",
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
        type=Path,
        default=None,
        help="Override the configured SQLite database path",
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
    graph_visualize.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Override the configured SQLite database path",
    )
    graph_visualize.add_argument(
        "--term",
        dest="focus_term",
        default=None,
        help="Render a bounded neighborhood around this term",
    )
    graph_visualize.add_argument(
        "--language",
        dest="focus_language",
        default=None,
        metavar="BCP47",
        help="Language tag used to disambiguate --term",
    )
    graph_visualize.add_argument(
        "--depth",
        type=_non_negative_int,
        default=1,
        help="Maximum hop distance from --term (default: 1)",
    )
    graph_visualize.add_argument(
        "--dimension",
        dest="dimensions",
        action="append",
        choices=GRAPH_DIMENSION_CHOICES,
        default=None,
        help="Relationship dimension to show; repeat to select several",
    )
    graph_visualize.add_argument(
        "--max-nodes",
        type=_positive_int,
        default=None,
        help="Maximum nodes rendered (default: configured graph limit)",
    )
    graph_visualize.add_argument(
        "--max-edges",
        type=_positive_int,
        default=None,
        help="Maximum edges rendered (default: configured graph limit)",
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
        default=None,
        help=(
            "Sentence transformer model name (default: configured model, "
            f"initially {DEFAULT_EMBEDDING_MODEL})"
        ),
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

    conversation_start = conversation_sub.add_parser(
        "start", help="Start a new conversation"
    )
    conversation_start.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional title for the conversation",
    )

    conversation_list = conversation_sub.add_parser(
        "list", help="List all conversations"
    )
    conversation_list.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of conversations to list (default: 10)",
    )

    conversation_show = conversation_sub.add_parser(
        "show", help="Show messages in a conversation"
    )
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

    setup_nltk_parser = subparsers.add_parser(
        "setup-nltk",
        help="Download the NLTK corpora required by Word Forge",
    )
    setup_nltk_parser.add_argument(
        "--multilingual",
        action="store_true",
        help="Also install Open Multilingual Wordnet component datasets",
    )
    setup_nltk_parser.add_argument(
        "--accept-source-licenses",
        action="store_true",
        help="Acknowledge responsibility for optional dataset license terms",
    )

    doctor_parser = subparsers.add_parser(
        "doctor", help="Check installation and optional feature readiness"
    )
    doctor_parser.add_argument(
        "--fix",
        action="store_true",
        help="Download missing NLTK parser resources",
    )
    doctor_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )

    models_parser = subparsers.add_parser(
        "models", help="List or recommend local language-model profiles"
    )
    models_parser.add_argument(
        "models_action",
        nargs="?",
        choices=["list", "recommend"],
        default="list",
        help="List every profile or recommend one for this machine",
    )
    models_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable profile and runtime details",
    )

    sources_parser = subparsers.add_parser(
        "sources", help="Inspect lexical source licenses and importer readiness"
    )
    sources_parser.add_argument(
        "sources_action",
        nargs="?",
        choices=["list"],
        default="list",
        help="List registered lexical data sources",
    )
    sources_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the versioned machine-readable source catalog",
    )
    sources_parser.add_argument(
        "--unattended-eligible",
        action="store_true",
        help="Show only core and permissive sources eligible for automation",
    )

    data_parser = subparsers.add_parser(
        "data", help="Import and manage governed lexical data"
    )
    data_sub = data_parser.add_subparsers(dest="data_command")
    kaikki_parser = data_sub.add_parser(
        "import-kaikki",
        help="Stream a local Kaikki/Wiktextract JSONL artifact into SQLite",
    )
    kaikki_parser.add_argument(
        "artifact",
        type=Path,
        help="Path to a plain, gzip, or bzip2 JSON Lines artifact",
    )
    kaikki_parser.add_argument(
        "--source-version",
        required=True,
        help="Exact upstream release date or immutable source version",
    )
    kaikki_parser.add_argument(
        "--source-url",
        required=True,
        help="HTTPS URL from which this exact artifact was obtained",
    )
    kaikki_parser.add_argument(
        "--accept-source-license",
        action="store_true",
        help="Acknowledge the artifact's attribution/share-alike source terms",
    )
    kaikki_parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Override the configured SQLite database path",
    )
    checkpoint_group = kaikki_parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (default: a sidecar next to the artifact)",
    )
    checkpoint_group.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable resumable checkpoint writes",
    )
    kaikki_parser.add_argument(
        "--expected-sha256",
        default=None,
        metavar="HEX",
        help="Reject the artifact unless its exact-byte SHA-256 matches",
    )
    kaikki_parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Entries per atomic database transaction (1-10000; default: 500)",
    )
    kaikki_parser.add_argument(
        "--language",
        dest="languages",
        action="append",
        default=[],
        metavar="BCP47",
        help="Import only this language; repeat to select multiple languages",
    )
    kaikki_parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Stop after this many matching entries (useful for staged imports)",
    )
    kaikki_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a stable machine-readable import report",
    )

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

    if args.command == "start":
        from word_forge.parser.model_profiles import ModelProfileError

        try:
            start(
                args.words,
                run_minutes=args.minutes,
                worker_count=args.workers,
                db_path=args.db_path,
                vector_model=args.vector_model,
                llm_model=args.llm_model,
                llm_profile=args.llm_profile,
                enable_vector=args.vector,
                language=args.language,
            )
        except (ModelProfileError, ValueError) as exc:
            LOGGER.error("Unable to start Word Forge: %s", exc)
            exit_code = 2
    elif args.command == "models":
        exit_code = run_model_catalog(args.models_action, args.json)
    elif args.command == "sources":
        exit_code = run_source_catalog(
            json_output=args.json,
            unattended_only=args.unattended_eligible,
        )
    elif args.command == "data":
        if args.data_command == "import-kaikki":
            exit_code = run_kaikki_import(
                args.artifact,
                source_version=args.source_version,
                source_url=args.source_url,
                accept_source_license=args.accept_source_license,
                db_path=args.db_path,
                checkpoint_path=args.checkpoint,
                disable_checkpoint=args.no_checkpoint,
                expected_sha256=args.expected_sha256,
                batch_size=args.batch_size,
                languages=args.languages,
                max_entries=args.max_entries,
                json_output=args.json,
            )
        else:
            data_parser.print_help()
            exit_code = 1
    elif args.command == "graph":
        if args.graph_command == "build":
            exit_code = (
                0
                if run_graph_build(
                    poll_interval=args.poll_interval,
                    timeout=args.timeout,
                    db_path=args.db_path,
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
                    db_path=args.db_path,
                    focus_term=args.focus_term,
                    focus_language=args.focus_language,
                    depth=args.depth,
                    dimensions=args.dimensions,
                    max_nodes=args.max_nodes,
                    max_edges=args.max_edges,
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
                if run_conversation_show(
                    conversation_id=args.conversation_id, limit=args.limit
                )
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
        exit_code = run_setup_nltk(
            include_multilingual=args.multilingual,
            accept_source_licenses=args.accept_source_licenses,
        )
    elif args.command == "doctor":
        import json

        from word_forge.diagnostics import render_diagnostics, run_diagnostics

        report = run_diagnostics(fix=args.fix)
        output = (
            json.dumps(report.to_dict(), indent=2)
            if args.json
            else render_diagnostics(report)
        )
        print(output)
        exit_code = 0 if report.ok else 1
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


def run_graph_build(
    *,
    graph_manager: Optional["GraphManager"] = None,
    poll_interval: float = 1.0,
    timeout: float = 120.0,
    db_path: Optional[Path] = None,
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
        db_manager = DBManager(db_path=db_path) if db_path is not None else DBManager()
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
    db_path: Optional[Path] = None,
    focus_term: Optional[str] = None,
    focus_language: Optional[str] = None,
    depth: int = 1,
    dimensions: Optional[List["RelationshipDimension"]] = None,
    max_nodes: Optional[int] = None,
    max_edges: Optional[int] = None,
) -> bool:
    """Build a bounded graph view and emit a standalone visualization file."""

    _setup_logging()
    from word_forge.database.database_manager import DBManager
    from word_forge.graph.graph_manager import GraphManager

    owns_manager = graph_manager is None
    db_manager: Optional[DBManager] = None
    if graph_manager is None:
        db_manager = DBManager(db_path=db_path) if db_path is not None else DBManager()
        graph_manager = GraphManager(db_manager=db_manager)

    try:
        if focus_language is not None and focus_term is None:
            raise ValueError("--language requires --term")
        graph_manager.build_graph(compute_layout=False)
        graph_manager.visualize(
            output_path=output_path,
            use_3d=use_3d if use_3d else None,
            dimensions_filter=dimensions,
            open_in_browser=open_in_browser,
            focus_term=focus_term,
            focus_language=focus_language,
            depth=depth,
            max_nodes=max_nodes,
            max_edges=max_edges,
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
    embedder: Optional[str] = None,
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

    vector_store = VectorStore(db_manager=db, model_name=embedder)
    worker = VectorWorker(
        db=db,
        vector_store=vector_store,
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


def run_vector_search(
    *,
    query: str,
    k: int = 5,
    content_type: Optional[str] = None,
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
                metadata = result.get("metadata", {})
                text = result.get("text", "")

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


def run_conversation_start(*, title: Optional[str] = None) -> bool:
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
            cursor.execute(
                "SELECT id FROM conversations WHERE id = ?;", (conversation_id,)
            )
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
