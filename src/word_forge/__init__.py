"""Word Forge: a local-first multilingual living lexicon.

The package root is intentionally lightweight. Public objects are imported on
first access so ``python -m word_forge``, shell completion, health checks, and
applications that only need version metadata do not initialize graph, neural,
or linguistic dependencies.
"""

from __future__ import annotations

import importlib
from typing import Any

from word_forge._version import __version__

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "Config": ("word_forge.config", "Config"),
    "GraphBuilder": ("word_forge.graph.graph_builder", "GraphBuilder"),
    "GraphManager": ("word_forge.graph.graph_manager", "GraphManager"),
    "GraphQuery": ("word_forge.graph.graph_query", "GraphQuery"),
    "IdempotencyConflictError": ("word_forge.jobs", "IdempotencyConflictError"),
    "Job": ("word_forge.jobs", "Job"),
    "JobError": ("word_forge.jobs", "JobError"),
    "JobLeaseError": ("word_forge.jobs", "JobLeaseError"),
    "JobQueue": ("word_forge.jobs", "JobQueue"),
    "LexicalEntry": ("word_forge.models", "LexicalEntry"),
    "Lexicon": ("word_forge.lexicon", "Lexicon"),
    "LexiconStatistics": ("word_forge.models", "LexiconStatistics"),
    "NetworkEdge": ("word_forge.models", "NetworkEdge"),
    "NetworkNode": ("word_forge.models", "NetworkNode"),
    "NetworkSnapshot": ("word_forge.models", "NetworkSnapshot"),
    "Relationship": ("word_forge.models", "Relationship"),
    "Sense": ("word_forge.models", "Sense"),
    "create_lexical_dataset": (
        "word_forge.parser.lexical_functions",
        "create_lexical_dataset",
    ),
    "get_synsets": ("word_forge.parser.lexical_functions", "get_synsets"),
    "get_wordnet_data": ("word_forge.parser.lexical_functions", "get_wordnet_data"),
}


def __getattr__(name: str) -> Any:
    """Load one compatibility export only when an application requests it."""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(importlib.import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy public names to shells, IDEs, and documentation tools."""

    return sorted({*globals(), *_LAZY_EXPORTS})


__all__ = [
    "Config",
    "GraphBuilder",
    "GraphManager",
    "GraphQuery",
    "IdempotencyConflictError",
    "Job",
    "JobError",
    "JobLeaseError",
    "JobQueue",
    "LexicalEntry",
    "Lexicon",
    "LexiconStatistics",
    "NetworkEdge",
    "NetworkNode",
    "NetworkSnapshot",
    "Relationship",
    "Sense",
    "__version__",
    "create_lexical_dataset",
    "get_synsets",
    "get_wordnet_data",
]
