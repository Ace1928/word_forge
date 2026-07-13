"""Public command-line entry point for Word Forge.

The implementation lives in :mod:`word_forge.forge` so the console script,
``python -m word_forge``, and Python callers execute the same parser and return
the same stable exit codes.
"""

from __future__ import annotations

from collections.abc import Sequence

from word_forge.forge import main as _main


def main(argv: Sequence[str] | None = None) -> int:
    """Run Word Forge and return a process-compatible exit code."""

    return _main(list(argv) if argv is not None else None)


__all__ = ["main"]
