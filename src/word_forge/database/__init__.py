"""Lazy public exports for Word Forge persistence components."""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_IMPORTS = {
    "ConnectionError": ("word_forge.database.database_manager", "ConnectionError"),
    "DBManager": ("word_forge.database.database_manager", "DBManager"),
    "DatabaseConfig": ("word_forge.database.database_config", "DatabaseConfig"),
    "DatabaseError": ("word_forge.database.database_manager", "DatabaseError"),
    "QueryError": ("word_forge.database.database_manager", "QueryError"),
    "RelationshipTypeManager": (
        "word_forge.database.database_manager",
        "RelationshipTypeManager",
    ),
    "RevisionConflictError": (
        "word_forge.database.database_manager",
        "RevisionConflictError",
    ),
    "SchemaError": ("word_forge.database.database_manager", "SchemaError"),
    "TermNotFoundError": ("word_forge.database.database_manager", "TermNotFoundError"),
    "TransactionError": ("word_forge.database.database_manager", "TransactionError"),
}


def __getattr__(name: str) -> Any:
    """Load one database component only when requested."""

    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(importlib.import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy names to interactive help and documentation tools."""

    return sorted({*globals(), *_LAZY_IMPORTS})


__all__ = sorted(_LAZY_IMPORTS)
