#!/usr/bin/env python3
"""Run a deterministic, offline benchmark of Word Forge's stable API.

The benchmark creates an isolated temporary lexicon, so it never modifies a
user database and never downloads a model. Results are hardware-specific and
are intended for comparing two revisions on the same machine.

Example:
    python scripts/benchmark_core.py --entries 2000 --searches 500 --json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Permit direct use from a source checkout without requiring an editable install.
sys.path.insert(0, str(Path(__file__).parents[1].resolve() / "src"))

from word_forge import Lexicon  # noqa: E402


def _measure(operation: Callable[[], object], repetitions: int) -> dict[str, float]:
    """Measure one operation repeatedly and return stable latency summaries."""

    samples: list[float] = []
    started = time.perf_counter()
    for _ in range(repetitions):
        call_started = time.perf_counter()
        operation()
        samples.append((time.perf_counter() - call_started) * 1_000.0)
    elapsed = time.perf_counter() - started
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    return {
        "operations": float(repetitions),
        "seconds": elapsed,
        "operations_per_second": repetitions / elapsed if elapsed else float("inf"),
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[p95_index],
    }


def run_benchmark(entries: int, searches: int) -> dict[str, Any]:
    """Build and query one temporary lexicon and return benchmark results."""

    with tempfile.TemporaryDirectory(prefix="word-forge-benchmark-") as temporary:
        with Lexicon(Path(temporary) / "benchmark.sqlite") as lexicon:
            next_entry = 0

            def add_entry() -> None:
                nonlocal next_entry
                term = f"word_{next_entry:06d}"
                lexicon.add(
                    term,
                    definition=f"Deterministic benchmark definition {next_entry}.",
                    part_of_speech="noun",
                )
                next_entry += 1

            insertion = _measure(add_entry, entries)

            next_edge = 1

            def add_relationship() -> None:
                nonlocal next_edge
                lexicon.relate(
                    f"word_{next_edge - 1:06d}",
                    f"word_{next_edge:06d}",
                    kind="next",
                )
                next_edge += 1

            relationships = _measure(add_relationship, entries - 1)

            lookup_index = 0

            def exact_lookup() -> None:
                nonlocal lookup_index
                lexicon.get(f"word_{lookup_index % entries:06d}")
                lookup_index += 7919

            lookups = _measure(exact_lookup, searches)

            search_index = 0

            def text_search() -> None:
                nonlocal search_index
                lexicon.find(f"definition {search_index % entries}", limit=5)
                search_index += 1543

            text_searches = _measure(text_search, searches)

            network_started = time.perf_counter()
            snapshot = lexicon.network(limit=min(entries, 2_000))
            network_seconds = time.perf_counter() - network_started
            counts = lexicon.statistics().as_dict()

    return {
        "entries": entries,
        "searches": searches,
        "counts": counts,
        "network": {
            "nodes": len(snapshot.nodes),
            "edges": len(snapshot.edges),
            "seconds": network_seconds,
            "truncated": snapshot.truncated,
        },
        "operations": {
            "insert": insertion,
            "relationship": relationships,
            "exact_lookup": lookups,
            "text_search": text_searches,
        },
    }


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Word Forge's offline SQLite and graph operations."
    )
    parser.add_argument(
        "--entries",
        type=_positive_integer,
        default=1_000,
        help="Number of entries to create; defaults to 1000.",
    )
    parser.add_argument(
        "--searches",
        type=_positive_integer,
        default=200,
        help="Number of exact and text searches; defaults to 200.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.entries < 2:
        print("ERROR: --entries must be at least 2", file=sys.stderr)
        return 2

    report = run_benchmark(args.entries, args.searches)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    print(f"Word Forge core benchmark: {args.entries} entries, {args.searches} searches")
    print(f"{'operation':<18} {'ops/sec':>12} {'median ms':>12} {'p95 ms':>12}")
    for name, measurement in report["operations"].items():
        print(
            f"{name:<18} {measurement['operations_per_second']:>12.1f} "
            f"{measurement['median_ms']:>12.3f} {measurement['p95_ms']:>12.3f}"
        )
    network = report["network"]
    print(
        f"network snapshot: {network['nodes']} nodes, {network['edges']} edges "
        f"in {network['seconds']:.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
