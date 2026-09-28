"""Relay WebSocket broadcasts between API worker processes through the database.

A browser's WebSocket is connected to one worker, but the events it wants
often originate in another: market data and fills on the ``trading-venue``
leader, telemetry on the ingestion leader, fired alerts on the
``alert-evaluator`` leader. With ``VPP_API_WORKERS > 1`` every worker runs a
:class:`WebSocketRelay`:

* each local broadcast is also queued and written (in batches) to
  ``cluster_events`` with this process as ``origin``;
* a reader tails ``cluster_events`` and re-broadcasts rows from other
  origins to this process's clients only (``broadcast_local``), so nothing
  loops.

Latency is about one poll interval (``VPP_CLUSTER_POLL_INTERVAL_SECONDS``).
Delivery is best-effort, like the in-process path: the write queue drops
messages when full, and rows are purged after :data:`RETENTION`. Rows whose
ids commit out of order (PostgreSQL sequences) are still picked up by
re-reading a small window below the newest id seen.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import logging
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import delete, func, select

from vpp.cluster.node import node_id, utcnow
from vpp.cluster.rpc import dumps
from vpp.db.models import ClusterEventModel

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from vpp.api.websocket import ConnectionManager

logger = logging.getLogger(__name__)

RETENTION = timedelta(minutes=5)
REREAD_WINDOW = 200  # ids below the newest seen that are re-checked
QUEUE_SIZE = 5000
BATCH = 500


class WebSocketRelay:
    def __init__(
        self,
        manager: ConnectionManager,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        poll_s: float = 0.25,
        origin: str | None = None,
    ) -> None:
        self._manager = manager
        self._factory = session_factory
        self._poll_s = poll_s
        self.origin = origin or node_id()
        self._queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue(maxsize=QUEUE_SIZE)
        self._tasks: list[asyncio.Task] = []
        self._last_id = 0
        self._seen: collections.deque[int] = collections.deque(maxlen=4 * BATCH)
        self._seen_set: set[int] = set()
        self.dropped = 0
        self.relayed_in = 0

    async def start(self) -> None:
        async with self._factory() as session:
            newest = (await session.execute(select(func.max(ClusterEventModel.id)))).scalar()
        self._last_id = int(newest or 0)
        self._manager.relay = self.publish
        self._tasks = [
            asyncio.create_task(self._writer(), name="vpp-ws-relay-writer"),
            asyncio.create_task(self._reader(), name="vpp-ws-relay-reader"),
        ]

    async def stop(self) -> None:
        if self._manager.relay == self.publish:
            self._manager.relay = None
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks = []
        with contextlib.suppress(Exception):
            await self._flush()  # best effort: what was queued before shutdown

    # -- outbound ----------------------------------------------------------

    async def publish(self, channel: str, data: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait((channel, dumps(data)))
        except asyncio.QueueFull:
            self.dropped += 1
            if self.dropped % 100 == 1:
                logger.warning("WebSocket relay queue full; dropped %d messages", self.dropped)

    async def _writer(self) -> None:
        while True:
            first = await self._queue.get()
            batch = [first]
            while len(batch) < BATCH and not self._queue.empty():
                batch.append(self._queue.get_nowait())
            try:
                await self._insert(batch)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("WebSocket relay write failed; %d messages lost", len(batch))

    async def _flush(self) -> None:
        batch = []
        while not self._queue.empty():
            batch.append(self._queue.get_nowait())
        if batch:
            await self._insert(batch)

    async def _insert(self, batch: list[tuple[str, str]]) -> None:
        now = utcnow()
        async with self._factory() as session:
            session.add_all(
                ClusterEventModel(origin=self.origin, channel=ch, payload_json=p, created_at=now)
                for ch, p in batch
            )
            await session.commit()

    # -- inbound -----------------------------------------------------------

    async def poll_once(self) -> int:
        """Re-broadcast new rows from other workers. Returns how many."""
        async with self._factory() as session:
            rows = (
                (
                    await session.execute(
                        select(ClusterEventModel)
                        .where(ClusterEventModel.id > self._last_id - REREAD_WINDOW)
                        .order_by(ClusterEventModel.id)
                        .limit(BATCH + REREAD_WINDOW)
                    )
                )
                .scalars()
                .all()
            )
        delivered = 0
        for row in rows:
            if row.id in self._seen_set:
                continue
            self._remember(row.id)
            self._last_id = max(self._last_id, row.id)
            if row.origin == self.origin:
                continue
            try:
                data = json.loads(row.payload_json)
            except ValueError:
                continue
            await self._manager.broadcast_local(row.channel, data)
            delivered += 1
        self.relayed_in += delivered
        return delivered

    def _remember(self, event_id: int) -> None:
        if len(self._seen) == self._seen.maxlen:
            self._seen_set.discard(self._seen[0])
        self._seen.append(event_id)
        self._seen_set.add(event_id)

    async def purge(self) -> int:
        async with self._factory() as session:
            result = await session.execute(
                delete(ClusterEventModel)
                .where(ClusterEventModel.created_at < utcnow() - RETENTION)
                .execution_options(synchronize_session=False)
            )
            await session.commit()
            return int(getattr(result, "rowcount", 0) or 0)

    async def _reader(self) -> None:
        loop = asyncio.get_running_loop()
        next_purge = loop.time() + 60.0
        while True:
            try:
                await self.poll_once()
                if loop.time() >= next_purge:
                    next_purge = loop.time() + 60.0
                    await self.purge()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("WebSocket relay read failed")
            await asyncio.sleep(self._poll_s)
