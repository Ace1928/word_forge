"""Async coordination for normal API work and exclusive database replacement."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class OperationGate:
    """Allow concurrent normal work while making restore fully exclusive."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._active = 0
        self._exclusive = False

    @asynccontextmanager
    async def operation(self) -> AsyncIterator[None]:
        """Enter one normal operation, waiting while an exclusive task runs."""

        async with self._condition:
            await self._condition.wait_for(lambda: not self._exclusive)
            self._active += 1
        try:
            yield
        finally:
            async with self._condition:
                self._active -= 1
                if self._active == 0:
                    self._condition.notify_all()

    @asynccontextmanager
    async def exclusive(self) -> AsyncIterator[None]:
        """Wait for normal work to drain, then block new work until completion."""

        async with self._condition:
            await self._condition.wait_for(lambda: not self._exclusive)
            self._exclusive = True
            try:
                await self._condition.wait_for(lambda: self._active == 0)
            except BaseException:
                # Cancellation while waiting must not leave the gate locked
                # forever. Reset state before propagating the cancellation.
                self._exclusive = False
                self._condition.notify_all()
                raise
        try:
            yield
        finally:
            async with self._condition:
                self._exclusive = False
                self._condition.notify_all()


__all__ = ["OperationGate"]
