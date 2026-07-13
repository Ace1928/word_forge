"""Cross-platform filesystem locations for Word Forge runtime data.

Installed applications must not write into their package directory: it may be
read-only, shared by multiple users, or replaced during an upgrade.  Word Forge
therefore stores its database, indexes, generated graphs, and logs in the
current user's application-data directory.

The complete location can be moved with ``WORD_FORGE_HOME``.  Advanced users
may override only data or logs with ``WORD_FORGE_DATA_DIR`` and
``WORD_FORGE_LOG_DIR`` respectively.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final

PACKAGE_ROOT: Final[Path] = Path(__file__).resolve().parent
PROJECT_ROOT: Final[Path] = PACKAGE_ROOT.parents[1]


def _expanded_path(raw_path: str) -> Path:
    """Expand user markers and return an absolute, normalized path."""

    return Path(raw_path).expanduser().resolve()


def default_home_directory(
    *,
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
    home: Path | None = None,
) -> Path:
    """Return the operating system's conventional Word Forge data directory.

    Parameters are injectable to make platform behavior deterministic in tests;
    normal callers should use the defaults.
    """

    environment = os.environ if environ is None else environ
    current_platform = sys.platform if platform is None else platform
    user_home = Path.home() if home is None else home

    configured_home = environment.get("WORD_FORGE_HOME", "").strip()
    if configured_home:
        return _expanded_path(configured_home)

    if current_platform == "win32":
        local_app_data = environment.get("LOCALAPPDATA", "").strip()
        base = _expanded_path(local_app_data) if local_app_data else user_home / "AppData" / "Local"
        return base / "WordForge"
    if current_platform == "darwin":
        return user_home / "Library" / "Application Support" / "WordForge"

    xdg_data_home = environment.get("XDG_DATA_HOME", "").strip()
    base = _expanded_path(xdg_data_home) if xdg_data_home else user_home / ".local" / "share"
    return base / "word-forge"


def _configured_subdirectory(environment_key: str, fallback: Path) -> Path:
    """Resolve an optional directory override with a stable fallback."""

    configured = os.environ.get(environment_key, "").strip()
    return _expanded_path(configured) if configured else fallback


HOME_ROOT: Final[Path] = default_home_directory()
DATA_ROOT: Final[Path] = _configured_subdirectory("WORD_FORGE_DATA_DIR", HOME_ROOT / "data")
LOGS_ROOT: Final[Path] = _configured_subdirectory("WORD_FORGE_LOG_DIR", HOME_ROOT / "logs")
MODELS_ROOT: Final[Path] = _configured_subdirectory("WORD_FORGE_MODELS_DIR", HOME_ROOT / "models")
DATABASE_PATH: Final[Path] = _expanded_path(
    os.environ.get("WORD_FORGE_DB_PATH", str(DATA_ROOT / "word_forge.sqlite"))
)


__all__ = [
    "DATA_ROOT",
    "DATABASE_PATH",
    "HOME_ROOT",
    "LOGS_ROOT",
    "MODELS_ROOT",
    "PACKAGE_ROOT",
    "PROJECT_ROOT",
    "default_home_directory",
]
