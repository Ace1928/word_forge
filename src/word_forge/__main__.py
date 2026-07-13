"""Run the Word Forge CLI with ``python -m word_forge``."""

from __future__ import annotations

import sys

from word_forge.cli import main

if __name__ == "__main__":  # pragma: no cover - covered by wheel smoke tests
    raise SystemExit(main(sys.argv[1:]))
