"""Concurrency tests for exclusive database replacement coordination."""

from __future__ import annotations

import asyncio

from word_forge.web.operations import OperationGate


def test_exclusive_work_waits_for_readers_and_blocks_new_operations() -> None:
    async def scenario() -> None:
        gate = OperationGate()
        exclusive_entered = asyncio.Event()

        async def use_exclusive() -> None:
            async with gate.exclusive():
                exclusive_entered.set()

        async with gate.operation():
            exclusive_task = asyncio.create_task(use_exclusive())
            await asyncio.sleep(0)
            assert not exclusive_entered.is_set()
        await asyncio.wait_for(exclusive_entered.wait(), timeout=1.0)
        await exclusive_task

        operation_entered = asyncio.Event()

        async def use_operation() -> None:
            async with gate.operation():
                operation_entered.set()

        async with gate.exclusive():
            operation_task = asyncio.create_task(use_operation())
            await asyncio.sleep(0)
            assert not operation_entered.is_set()
        await asyncio.wait_for(operation_entered.wait(), timeout=1.0)
        await operation_task

    asyncio.run(scenario())


def test_cancelled_exclusive_waiter_releases_the_gate() -> None:
    async def scenario() -> None:
        gate = OperationGate()

        async def wait_exclusively() -> None:
            async with gate.exclusive():
                pass

        async with gate.operation():
            waiter = asyncio.create_task(wait_exclusively())
            await asyncio.sleep(0)
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)

        async def use_operation() -> None:
            async with gate.operation():
                pass

        await asyncio.wait_for(use_operation(), timeout=1.0)

    asyncio.run(scenario())
