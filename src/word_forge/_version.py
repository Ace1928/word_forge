"""Single source of truth for the Word Forge package version."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

_FALLBACK_VERSION = "1.0.0"


def package_version() -> str:
    """Return installed distribution metadata, or the source-tree version."""

    try:
        return version("word-forge")
    except PackageNotFoundError:
        return _FALLBACK_VERSION


__version__ = package_version()

__all__ = ["__version__", "package_version"]
