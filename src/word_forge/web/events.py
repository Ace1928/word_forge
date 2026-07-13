"""Bounded in-process event fan-out for live GUI updates."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any


class EventHub:
    """Publish invalidation events to connected browser sessions.

    Slow subscribers lose their oldest event rather than blocking writers. The
    GUI refetches authoritative state after every event, so the newest signal is
    more useful than an obsolete backlog.
    """

    def __init__(self, *, queue_size: int = 128, history_size: int = 512) -> None:
        if queue_size < 1 or history_size < 1:
            raise ValueError("queue and history sizes must be positive")
        self._queue_size = queue_size
        self._history: deque[dict[str, Any]] = deque(maxlen=history_size)
        self._sequence = 0
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._lock = asyncio.Lock()

    async def subscribe(self, *, since: int | None = None) -> asyncio.Queue[dict[str, Any]]:
        """Register a subscriber and replay a recoverable event gap."""

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(self._queue_size)
        async with self._lock:
            self._subscribers.add(queue)
            if since is not None and since < self._sequence:
                available = [event for event in self._history if int(event["sequence"]) > since]
                first_sequence = int(self._history[0]["sequence"]) if self._history else 0
                if since < first_sequence - 1 or len(available) > self._queue_size:
                    queue.put_nowait({"type": "resync.required", "sequence": self._sequence})
                else:
                    for event in available:
                        queue.put_nowait(dict(event))
        return queue

    async def sequence(self) -> int:
        """Return the latest published event sequence."""

        async with self._lock:
            return self._sequence

    async def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Remove a subscriber queue if it remains registered."""

        async with self._lock:
            self._subscribers.discard(queue)

    async def publish(self, event: dict[str, Any]) -> None:
        """Deliver a copy of an event without blocking on slow clients."""

        async with self._lock:
            self._sequence += 1
            sequenced = {**event, "sequence": self._sequence}
            self._history.append(sequenced)
            for queue in tuple(self._subscribers):
                if queue.full():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:  # pragma: no cover - defensive race guard
                        pass
                    queue.put_nowait({"type": "resync.required", "sequence": self._sequence})
                else:
                    queue.put_nowait(dict(sequenced))


__all__ = ["EventHub"]
