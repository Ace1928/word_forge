"""Lazy utility exports that keep the core lexicon dependency-free."""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_EXPORTS = {
    "Err": ("word_forge.utils.result", "Err"),
    "Ok": ("word_forge.utils.result", "Ok"),
    "Result": ("word_forge.utils.result", "Result"),
    "ensure_nltk_data": ("word_forge.utils.nltk_utils", "ensure_nltk_data"),
    "failure": ("word_forge.utils.result", "failure"),
    "success": ("word_forge.utils.result", "success"),
}


def __getattr__(name: str) -> Any:
    """Load one helper on demand."""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(importlib.import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy utility names to help and completion tools."""

    return sorted({*globals(), *_LAZY_EXPORTS})


__all__ = sorted(_LAZY_EXPORTS)
