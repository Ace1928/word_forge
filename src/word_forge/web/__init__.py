"""Optional local web interface for Word Forge."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def create_app(database_path: str | Path | None = None, **options: Any) -> Any:
    """Create the web application, or explain how to install GUI support."""

    try:
        from word_forge.web.app import create_app as _create_app
    except ImportError as exc:  # pragma: no cover - optional environment
        raise RuntimeError(
            "Word Forge GUI dependencies are missing. Install them with "
            '`pip install "word-forge[gui]"`.'
        ) from exc
    return _create_app(database_path=database_path, **options)


__all__ = ["create_app"]
