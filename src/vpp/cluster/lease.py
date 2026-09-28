"""DB-backed leadership leases for work that must run once per deployment.

Every API process runs the same lifespan. Work that must happen once
cluster-wide (the simulated venue's market-data tick, the degradation
updater, alert evaluation, MQTT/Modbus ingestion, protocol adapters + the
DR orchestrator) is wrapped in a :class:`LeaderElector`: the process holding
the named lease runs the *leader* role, every other process runs the
optional *follower* role (or nothing) and takes over once the lease expires.

The lease is one row in ``cluster_leases``. Acquire and renew are the same
atomic statement, valid on SQLite and PostgreSQL::

    UPDATE cluster_leases SET holder = :me, expires_at = :now + ttl
     WHERE name = :name AND (holder = :me OR expires_at < :now)

falling back to an ``INSERT`` for a lease that does not exist yet (a primary
key conflict means another process won the race). Times come from the
processes' clocks (UTC), so hosts need NTP-synchronised clocks; skew must
stay well below the TTL.

A lone process acquires every lease on its first attempt (made during
startup, before the first request), so single-process behaviour is unchanged.
After a crash, a restarted process on the same host takes over leases held
by its dead predecessor immediately (see
:func:`~vpp.cluster.node.holder_is_dead_local_process`); otherwise a lease
is free once its TTL has passed.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Protocol

from sqlalchemy import case, delete, or_, select, update
from sqlalchemy.exc import IntegrityError

from vpp.cluster.node import as_utc, holder_is_dead_local_process, node_id, utcnow
from vpp.db.models import ClusterLeaseModel

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logger = logging.getLogger(__name__)

DEFAULT_TTL_S = 15.0


# ---------------------------------------------------------------------------
# Lease primitives
# ---------------------------------------------------------------------------


async def try_acquire(
    session_factory: async_sessionmaker[AsyncSession],
    name: str,
    holder: str,
    ttl_s: float,
    *,
    now: datetime | None = None,
) -> bool:
    """Acquire or renew lease *name* for *holder*. True when *holder* now holds it."""
    now = now or utcnow()
    expires = now + timedelta(seconds=ttl_s)
    lease = ClusterLeaseModel
    values = {
        "holder": holder,
        "expires_at": expires,
        "acquired_at": case((lease.holder == holder, lease.acquired_at), else_=now),
    }
    async with session_factory() as session:
        result = await session.execute(
            update(lease)
            .where(lease.name == name, or_(lease.holder == holder, lease.expires_at < now))
            .values(**values)
            .execution_options(synchronize_session=False)
        )
        if _rowcount(result) == 1:
            await session.commit()
            return True

        current = (
            await session.execute(select(lease.holder).where(lease.name == name))
        ).scalar_one_or_none()
        if current is None:
            session.add(lease(name=name, holder=holder, acquired_at=now, expires_at=expires))
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                return False
            return True

        if holder_is_dead_local_process(current):
            result = await session.execute(
                update(lease)
                .where(lease.name == name, lease.holder == current)
                .values(holder=holder, expires_at=expires, acquired_at=now)
                .execution_options(synchronize_session=False)
            )
            await session.commit()
            if _rowcount(result) == 1:
                logger.warning(
                    "Took over lease %r from %s (process on this host no longer exists)",
                    name,
                    current,
                )
                return True
            return False

        await session.rollback()
        return False


async def release(
    session_factory: async_sessionmaker[AsyncSession], name: str, holder: str
) -> bool:
    """Give up lease *name* if *holder* holds it (so a follower takes over at once)."""
    async with session_factory() as session:
        result = await session.execute(
            delete(ClusterLeaseModel)
            .where(ClusterLeaseModel.name == name, ClusterLeaseModel.holder == holder)
            .execution_options(synchronize_session=False)
        )
        await session.commit()
        return _rowcount(result) == 1


async def lease_holders(
    session_factory: async_sessionmaker[AsyncSession],
) -> dict[str, dict[str, object]]:
    """Every lease row: ``{name: {"holder", "expires_at", "expired"}}``."""
    now = utcnow()
    async with session_factory() as session:
        rows = (await session.execute(select(ClusterLeaseModel))).scalars().all()
    out: dict[str, dict[str, object]] = {}
    for row in rows:
        expires = as_utc(row.expires_at)
        out[row.name] = {
            "holder": row.holder,
            "expires_at": expires.isoformat() if expires else None,
            "expired": expires is None or expires < now,
        }
    return out


def _rowcount(result: object) -> int:
    return int(getattr(result, "rowcount", 0) or 0)


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------


class Role(Protocol):
    """Something started when a process takes a role and stopped when it leaves it."""

    async def start(self) -> None: ...

    async def stop(self) -> None: ...


class TaskRole:
    """Runs ``factory()`` as a task while the role is held; cancels it on stop."""

    def __init__(self, factory: Callable[[], Awaitable[None]], *, name: str) -> None:
        self._factory = factory
        self.name = name
        self.task: asyncio.Task | None = None

    async def start(self) -> None:
        self.task = asyncio.create_task(self._run(), name=self.name)

    async def _run(self) -> None:
        await self._factory()

    async def stop(self) -> None:
        task, self.task = self.task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("%s raised during shutdown", self.name)


class CallbackRole:
    """A role made of two async callbacks."""

    def __init__(
        self,
        start: Callable[[], Awaitable[None]],
        stop: Callable[[], Awaitable[None]],
    ) -> None:
        self._start = start
        self._stop = stop

    async def start(self) -> None:
        await self._start()

    async def stop(self) -> None:
        await self._stop()


# ---------------------------------------------------------------------------
# Elector
# ---------------------------------------------------------------------------

_electors: dict[str, LeaderElector] = {}


def get_elector(name: str) -> LeaderElector | None:
    return _electors.get(name)


def is_local(name: str) -> bool:
    """Whether work guarded by lease *name* should run in this process.

    True when this process leads *name*, and also when no elector for it is
    running (library use, tests without a lifespan): there is nobody else.
    """
    elector = _electors.get(name)
    return elector is None or elector.is_leader


def leadership() -> dict[str, bool]:
    """``{lease name: this process leads it}`` for every running elector."""
    return {name: e.is_leader for name, e in sorted(_electors.items())}


class LeaderElector:
    """Holds lease *name* when it can; runs *leader* while holding it, *follower* otherwise.

    :meth:`start` makes the first acquisition attempt inline, so a lone
    process is already leader (and its leader role started) when ``start``
    returns. The lease is renewed every ``ttl_s / 3``; if renewal keeps
    failing (DB unreachable) the leader role is stopped before the lease can
    have expired in the database, so two processes never run it at once.
    """

    def __init__(
        self,
        name: str,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        leader: Role | None = None,
        follower: Role | None = None,
        holder: str | None = None,
        ttl_s: float = DEFAULT_TTL_S,
        renew_interval_s: float | None = None,
    ) -> None:
        if ttl_s <= 0:
            raise ValueError("ttl_s must be positive")
        self.name = name
        self.holder = holder or node_id()
        self.ttl_s = ttl_s
        self.renew_interval_s = renew_interval_s or ttl_s / 3.0
        self._factory = session_factory
        self._leader = leader
        self._follower = follower
        self._leading = False
        self._follower_active = False
        self._valid_until = 0.0
        self._loop_task: asyncio.Task | None = None
        self._step_lock = asyncio.Lock()
        self.transitions = 0

    # -- state ---------------------------------------------------------------

    @property
    def is_leader(self) -> bool:
        return self._leading and time.monotonic() < self._valid_until

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        _electors[self.name] = self
        await self.step()
        self._loop_task = asyncio.create_task(self._loop(), name=f"vpp-lease-{self.name}")

    async def stop(self) -> None:
        task, self._loop_task = self._loop_task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        async with self._step_lock:
            was_leading = self._leading
            await self._stop_leader()
            await self._stop_follower()
            if was_leading:
                try:
                    await release(self._factory, self.name, self.holder)
                except Exception:
                    logger.exception("Could not release lease %r", self.name)
        if _electors.get(self.name) is self:
            del _electors[self.name]

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.renew_interval_s)
            try:
                await self.step()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Lease %r step failed", self.name)

    async def step(self) -> None:
        """One acquire/renew attempt and the resulting role transition."""
        async with self._step_lock:
            started = time.monotonic()
            try:
                held: bool | None = await try_acquire(
                    self._factory, self.name, self.holder, self.ttl_s
                )
            except Exception:
                logger.exception("Lease %r: acquire/renew failed", self.name)
                held = None
            if held:
                self._valid_until = started + self.ttl_s
            elif held is None and self._leading and time.monotonic() < self._valid_until:
                return  # DB hiccup: keep leading until our lease could have expired

            if held and not self._leading:
                await self._stop_follower()
                self._leading = True
                self.transitions += 1
                logger.info(
                    "Acquired lease %r (holder %s): running leader work", self.name, self.holder
                )
                if self._leader is not None:
                    try:
                        await self._leader.start()
                    except Exception:
                        logger.exception("Leader role for %r failed to start", self.name)
            elif not held and self._leading:
                logger.warning(
                    "Lost lease %r (holder %s): stopping leader work", self.name, self.holder
                )
                await self._stop_leader()
                self.transitions += 1
                await self._start_follower()
            elif not held and not self._follower_active:
                await self._start_follower()

    async def _stop_leader(self) -> None:
        if not self._leading:
            return
        self._leading = False
        self._valid_until = 0.0
        if self._leader is not None:
            try:
                await self._leader.stop()
            except Exception:
                logger.exception("Leader role for %r failed to stop", self.name)

    async def _start_follower(self) -> None:
        if self._follower_active:
            return
        self._follower_active = True
        if self._follower is not None:
            logger.info("Lease %r is held elsewhere: running as follower", self.name)
            try:
                await self._follower.start()
            except Exception:
                logger.exception("Follower role for %r failed to start", self.name)

    async def _stop_follower(self) -> None:
        if not self._follower_active:
            return
        self._follower_active = False
        if self._follower is not None:
            try:
                await self._follower.stop()
            except Exception:
                logger.exception("Follower role for %r failed to stop", self.name)
