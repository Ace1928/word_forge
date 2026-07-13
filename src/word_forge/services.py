"""Portable endpoint resolution for optional local model services."""

from __future__ import annotations

import os
import re
from urllib.parse import urlsplit, urlunsplit


def _environment_key(service_key: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", service_key).strip("_").upper()
    return f"WORD_FORGE_{normalized}_URL"


def _normalize_override(raw_url: str, default_path: str) -> str:
    candidate = raw_url.strip().rstrip("/")
    if not candidate:
        raise ValueError("Service URL must not be empty")
    if "://" not in candidate:
        candidate = f"http://{candidate}"
    parsed = urlsplit(candidate)
    if not parsed.hostname:
        raise ValueError(f"Service URL has no host: {raw_url!r}")
    path = parsed.path or default_path
    if path and not path.startswith("/"):
        path = f"/{path}"
    return urlunsplit((parsed.scheme or "http", parsed.netloc, path, parsed.query, parsed.fragment))


def get_service_url(
    service_key: str,
    *,
    default_port: int,
    default_host: str = "127.0.0.1",
    default_protocol: str = "http",
    default_path: str = "",
    registry_path: str | None = None,
) -> str:
    """Resolve a service URL from environment overrides or safe defaults.

    ``registry_path`` remains accepted for source compatibility but is ignored;
    standalone Word Forge never reads a private monorepo registry.
    """

    del registry_path
    if not service_key.strip():
        raise ValueError("service_key must not be empty")
    if not 1 <= default_port <= 65_535:
        raise ValueError("default_port must be between 1 and 65535")

    override = os.environ.get(_environment_key(service_key))
    if override is None and service_key.casefold() == "ollama_http":
        override = os.environ.get("OLLAMA_HOST")
    if override:
        return _normalize_override(override, default_path)

    path = default_path
    if path and not path.startswith("/"):
        path = f"/{path}"
    return f"{default_protocol}://{default_host}:{default_port}{path}"


__all__ = ["get_service_url"]
