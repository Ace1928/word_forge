"""Tests for sequenced live-event replay and overflow recovery."""

from __future__ import annotations

import asyncio

from word_forge.web.events import EventHub


def test_event_sequences_and_replay_are_monotonic() -> None:
    async def scenario() -> None:
        hub = EventHub(queue_size=4, history_size=8)
        await hub.publish({"type": "entry.changed", "term": "one"})
        await hub.publish({"type": "entry.changed", "term": "two"})
        queue = await hub.subscribe(since=1)
        replay = queue.get_nowait()
        assert replay["term"] == "two"
        assert replay["sequence"] == 2
        await hub.unsubscribe(queue)

    asyncio.run(scenario())


def test_history_gap_requests_authoritative_resync() -> None:
    async def scenario() -> None:
        hub = EventHub(queue_size=2, history_size=2)
        for index in range(5):
            await hub.publish({"type": "entry.changed", "index": index})
        queue = await hub.subscribe(since=1)
        assert queue.get_nowait() == {"type": "resync.required", "sequence": 5}
        await hub.unsubscribe(queue)

    asyncio.run(scenario())


def test_slow_subscriber_is_not_allowed_to_block_writers() -> None:
    async def scenario() -> None:
        hub = EventHub(queue_size=1, history_size=8)
        queue = await hub.subscribe()
        await hub.publish({"type": "entry.changed", "term": "one"})
        await hub.publish({"type": "entry.changed", "term": "two"})
        assert queue.get_nowait() == {"type": "resync.required", "sequence": 2}
        await hub.unsubscribe(queue)

    asyncio.run(scenario())
