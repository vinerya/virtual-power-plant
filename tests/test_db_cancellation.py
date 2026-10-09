"""Cancelling a background task mid-write must never leave SQLite locked.

Cancelling SQLAlchemy mid-statement invalidates the connection without a
rollback; on SQLite it then keeps the write lock until it is garbage
collected and every other writer fails with "database is locked". Background
writers therefore run each transaction through
:func:`vpp.db.engine.run_shielded` / :func:`~vpp.db.engine.run_in_session`.
These tests cancel writers at many points and check that another connection
can always write straight away.
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from vpp.db.base import Base
from vpp.db.engine import run_in_session, run_shielded
from vpp.db.models import BatteryStateModel, ResourceModel

CANCEL_POINTS = 40


@pytest.fixture
async def sqlite_db(
    tmp_path: Path,
) -> AsyncIterator[tuple[Path, async_sessionmaker[AsyncSession]]]:
    path = tmp_path / "cancel.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as s:
            s.add(ResourceModel(id="b1", name="b1", resource_type="battery", rated_power=10.0))
            await s.commit()
        yield path, factory
    finally:
        await engine.dispose()


def _assert_writable(path: Path) -> None:
    other = sqlite3.connect(path, timeout=0.2)
    try:
        other.execute("UPDATE resources SET online = 1")
        other.commit()
    finally:
        other.close()


async def _cancel_at_many_points(path: Path, start: Callable[[], Awaitable[object]]) -> None:
    """Start the write, cancel it at evenly spread points of its run, check the DB is free.

    One uncancelled run measures how long the write takes; the cancellations
    then sweep from "before the first statement" to "after the commit".
    """
    loop = asyncio.get_running_loop()
    began = loop.time()
    await start()
    duration = loop.time() - began
    for step in range(CANCEL_POINTS):
        task = asyncio.ensure_future(start())
        await asyncio.sleep(duration * 1.2 * step / (CANCEL_POINTS - 1))
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        _assert_writable(path)


# ---------------------------------------------------------------------------
# The helper itself
# ---------------------------------------------------------------------------


async def test_run_in_session_cancelled_anywhere_never_leaves_the_write_lock_held(sqlite_db):
    path, factory = sqlite_db

    async def work(session: AsyncSession) -> None:
        # Read, ORM insert (flushed at commit) and a Core update: a typical writer.
        await session.get(ResourceModel, "b1")
        session.add(BatteryStateModel(resource_id="b1", soc=0.5))
        await session.flush()
        await session.execute(update(ResourceModel).values(current_power=1.0))

    await _cancel_at_many_points(path, lambda: run_in_session(factory, work))


async def test_run_in_session_commits_and_returns(sqlite_db):
    _path, factory = sqlite_db

    async def work(session: AsyncSession) -> int:
        await session.execute(update(ResourceModel).values(current_power=4.0))
        return 7

    assert await run_in_session(factory, work) == 7
    async with factory() as s:
        assert (await s.get(ResourceModel, "b1")).current_power == pytest.approx(4.0)


async def test_run_in_session_rolls_back_on_error(sqlite_db):
    path, factory = sqlite_db

    async def work(session: AsyncSession) -> None:
        await session.execute(update(ResourceModel).values(current_power=9.0))
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await run_in_session(factory, work)
    _assert_writable(path)
    async with factory() as s:
        assert (await s.get(ResourceModel, "b1")).current_power == pytest.approx(0.0)


async def test_run_shielded_finishes_the_work_before_the_cancellation_proceeds():
    done = asyncio.Event()

    async def work() -> None:
        await asyncio.sleep(0.02)
        done.set()

    task = asyncio.ensure_future(run_shielded(work))
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0.001)
    task.cancel()  # cancelled again while waiting: still waits for the work
    with pytest.raises(asyncio.CancelledError):
        await task
    assert done.is_set()


# ---------------------------------------------------------------------------
# Real background writers
# ---------------------------------------------------------------------------


async def test_cancelled_mqtt_ingestion_never_leaves_the_write_lock_held(sqlite_db):
    from vpp.protocols.base import ProtocolMessage
    from vpp.protocols.telemetry_ingestion import MQTTTelemetryIngestor

    path, factory = sqlite_db
    ingestor = MQTTTelemetryIngestor(None, factory)  # type: ignore[arg-type]
    message = ProtocolMessage(
        topic="vpp/site/battery/b1/state", payload={"soc": 0.5}, source="mqtt"
    )
    await _cancel_at_many_points(path, lambda: ingestor.ingest_message(message))


async def test_cancelled_forwarded_call_never_leaves_the_write_lock_held(sqlite_db):
    from vpp.cluster import rpc

    path, factory = sqlite_db

    async def handler(payload: dict, session: AsyncSession) -> int:
        session.add(BatteryStateModel(resource_id="b1", soc=0.5))
        await session.flush()
        return 1

    rpc.register_handler("test-cancel", "write", handler)
    try:

        async def submit_and_run() -> int:
            await rpc.submit("test-cancel", "write", {}, session_factory=factory)
            return await rpc.execute_pending(factory, ["test-cancel"])

        await _cancel_at_many_points(path, submit_and_run)
        # A call the executor claimed always got its final status written.
        from sqlalchemy import func, select

        from vpp.db.models import ClusterCallModel

        async with factory() as s:
            running = await s.scalar(
                select(func.count()).where(ClusterCallModel.status == "running")
            )
        assert running == 0
    finally:
        rpc._handlers.pop(("test-cancel", "write"), None)
