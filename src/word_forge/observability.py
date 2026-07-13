"""Dependency-free, opt-in timing instrumentation for Word Forge.

Set ``WORD_FORGE_TRACE_CALLS=1`` to log decorated call durations at debug
level.  Set ``WORD_FORGE_SLOW_CALL_MS`` to a positive number to warn about
slow calls.  Arguments and results are never logged.
"""

from __future__ import annotations

import functools
import logging
import os
import time
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar, overload

P = ParamSpec("P")
R = TypeVar("R")


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().casefold() in {"1", "true", "yes", "on"}


def _threshold_ms() -> float | None:
    raw = os.environ.get("WORD_FORGE_SLOW_CALL_MS", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        logging.getLogger(__name__).warning(
            "Ignoring invalid WORD_FORGE_SLOW_CALL_MS value %r", raw
        )
        return None
    return value if value > 0 else None


@overload
def eidosian(function: Callable[P, R], /) -> Callable[P, R]: ...


@overload
def eidosian(
    *, log: bool | None = None, **_: Any
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def eidosian(
    function: Callable[P, R] | None = None,
    /,
    *,
    log: bool | None = None,
    **_: Any,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Preserve the historical decorator API with safe local instrumentation."""

    def decorate(target: Callable[P, R]) -> Callable[P, R]:
        trace = _enabled("WORD_FORGE_TRACE_CALLS") if log is None else log
        threshold = _threshold_ms()
        metadata = {"log": trace, "slow_call_ms": threshold, "standalone": True}
        if not trace and threshold is None:
            target.__eidosian_decorated__ = True  # type: ignore[attr-defined]
            target.__eidosian_config__ = metadata  # type: ignore[attr-defined]
            return target

        logger = logging.getLogger(target.__module__)

        @functools.wraps(target)
        def measured(*args: P.args, **kwargs: P.kwargs) -> R:
            started = time.perf_counter()
            try:
                return target(*args, **kwargs)
            finally:
                elapsed = (time.perf_counter() - started) * 1_000.0
                if trace:
                    logger.debug("%s completed in %.3f ms", target.__qualname__, elapsed)
                if threshold is not None and elapsed >= threshold:
                    logger.warning(
                        "%s exceeded the %.3f ms slow-call threshold (%.3f ms)",
                        target.__qualname__,
                        threshold,
                        elapsed,
                    )

        measured.__eidosian_decorated__ = True  # type: ignore[attr-defined]
        measured.__eidosian_config__ = metadata  # type: ignore[attr-defined]
        return measured

    return decorate(function) if function is not None else decorate


__all__ = ["eidosian"]
