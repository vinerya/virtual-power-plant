"""Rate-limit and login-throttle counters shared by every API worker.

With one worker the HTTP rate limiter (:mod:`vpp.auth.middleware`) and the
login throttle (:mod:`vpp.auth.throttle`) keep their state in memory. With
several workers that state would be per process -- a client would get
``workers x`` the request limit and a password guesser ``workers x`` the
attempts -- so they keep it in the ``shared_rate_limits`` table instead.
``VPP_RATE_LIMIT_BACKEND`` picks the store: ``memory``, ``database`` or
``auto`` (the default: ``database`` when ``VPP_API_WORKERS > 1``, the same
test :mod:`vpp.cluster` uses to start the WebSocket relay).

Every update is one atomic ``INSERT ... ON CONFLICT (key) DO UPDATE`` (SQLite
and PostgreSQL), followed in the same transaction by a read or a
conditional ``UPDATE`` of the same row, which the upsert has already
locked. Times are Unix epoch seconds from the workers' clocks, so hosts
need NTP-synchronised clocks (as for :mod:`vpp.cluster.lease`).

* **HTTP requests** (``http:<client ip>``): a sliding-window counter over
  one minute. The estimate is ``prev_hits x (unused fraction of the
  previous window) + hits``; a request that would push it past the limit is
  refused and not counted, so a client sending at the limit rate keeps
  getting through.
* **Failed logins** (``login:<username>``): the exact semantics of
  :class:`vpp.auth.throttle.LoginThrottle` -- failures counted within
  ``VPP_LOGIN_LOCKOUT_SECONDS`` of the first one, a lockout of that length
  after ``VPP_LOGIN_MAX_FAILURES`` of them, a success clears the row.

Rows whose ``expires_at`` has passed carry no state; each process deletes
them at most once a minute, opportunistically after an update.

What happens when the database fails is up to the callers: the rate limiter
lets the request through (fail open, logged), the login throttle falls back
to its in-memory state (throttling is never switched off).
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import Table, case, delete, literal, select, update

from vpp.db.models import SharedRateLimitModel

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logger = logging.getLogger(__name__)

MEMORY = "memory"
DATABASE = "database"

#: Longest stored key; longer identifiers are replaced by their SHA-256.
MAX_KEY_LENGTH = 255
PURGE_INTERVAL_S = 60.0

_TABLE = cast("Table", SharedRateLimitModel.__table__)


def resolve_backend(settings: Any) -> str:
    """``"memory"`` or ``"database"`` for ``VPP_RATE_LIMIT_BACKEND`` (``auto`` resolved)."""
    choice = str(getattr(settings, "rate_limit_backend", "auto") or "auto").strip().lower()
    if choice in (MEMORY, DATABASE):
        return choice
    workers = int(getattr(settings, "api_workers", 1) or 1)
    return DATABASE if workers > 1 else MEMORY


def storage_key(kind: str, ident: str) -> str:
    """``<kind>:<ident>``, hashed when it would not fit the key column."""
    key = f"{kind}:{ident}"
    if len(key) <= MAX_KEY_LENGTH:
        return key
    return f"{kind}:sha256:{hashlib.sha256(ident.encode('utf-8')).hexdigest()}"


def _insert_for(session: AsyncSession) -> Any:
    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        return pg_insert
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        return sqlite_insert
    raise NotImplementedError(f"shared rate limits need SQLite or PostgreSQL, not {dialect}")


class SharedLimitStore:
    """Counters in ``shared_rate_limits``; one instance per process is enough.

    ``session_factory`` defaults to the application's (:func:`vpp.db.engine.
    get_session_factory`, looked up on every call because the database is
    initialised after the middleware is built). ``clock`` returns epoch
    seconds and must agree across workers.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._factory = session_factory
        self._clock = clock
        self._next_purge = 0.0

    def _sessions(self) -> async_sessionmaker[AsyncSession]:
        if self._factory is not None:
            return self._factory
        from vpp.db.engine import get_session_factory

        return get_session_factory()

    # -- HTTP requests ---------------------------------------------------

    async def hit(self, key: str, limit: int, window_s: float = 60.0) -> bool:
        """Count one request under *key*; False (and not counted) when over *limit*."""
        now = self._clock()
        start = math.floor(now / window_s) * window_s
        t = _TABLE
        async with self._sessions()() as session:
            stmt = _insert_for(session)(t).values(
                key=key,
                hits=1,
                prev_hits=0,
                window_start=start,
                locked_until=0.0,
                expires_at=start + 2 * window_s,
            )
            same = t.c.window_start == stmt.excluded.window_start
            follows = t.c.window_start + window_s == stmt.excluded.window_start
            stmt = stmt.on_conflict_do_update(
                index_elements=[t.c.key],
                set_={
                    "hits": case((same, t.c.hits + 1), else_=1),
                    "prev_hits": case((same, t.c.prev_hits), (follows, t.c.hits), else_=0),
                    "window_start": stmt.excluded.window_start,
                    "expires_at": stmt.excluded.expires_at,
                },
            )
            await session.execute(stmt)
            row = (
                await session.execute(
                    select(t.c.hits, t.c.prev_hits, t.c.window_start).where(t.c.key == key)
                )
            ).one()
            # Another worker with a clock a window ahead may have moved the
            # row on; judge by the window the row is in.
            elapsed = min(max(now - float(row.window_start), 0.0), window_s)
            estimate = row.prev_hits * (1.0 - elapsed / window_s) + row.hits
            allowed = bool(estimate <= limit)
            if not allowed:
                await session.execute(
                    update(t)
                    .where(t.c.key == key, t.c.window_start == row.window_start)
                    .values(hits=t.c.hits - 1)
                )
            await session.commit()
        await self._maybe_purge(now)
        return allowed

    # -- Failed logins ---------------------------------------------------

    async def login_retry_after(self, key: str) -> int | None:
        """Seconds until *key* may try again, or None if not locked."""
        now = self._clock()
        async with self._sessions()() as session:
            locked_until = (
                await session.execute(select(_TABLE.c.locked_until).where(_TABLE.c.key == key))
            ).scalar_one_or_none()
        if locked_until is None or locked_until <= now:
            return None
        return max(1, math.ceil(locked_until - now))

    async def login_failure(self, key: str, max_failures: int, window_s: float) -> None:
        """Record one failure; lock *key* for *window_s* after *max_failures* in the window."""
        now = self._clock()
        t = _TABLE
        async with self._sessions()() as session:
            stmt = _insert_for(session)(t).values(
                key=key,
                hits=1,
                prev_hits=0,
                window_start=now,
                locked_until=0.0,
                expires_at=now + window_s,
            )
            # The window of the stored failures has passed: start a new one.
            stale = literal(now) - t.c.window_start > window_s
            stmt = stmt.on_conflict_do_update(
                index_elements=[t.c.key],
                set_={
                    "hits": case((stale, 1), else_=t.c.hits + 1),
                    "window_start": case((stale, now), else_=t.c.window_start),
                    "locked_until": case((stale, 0.0), else_=t.c.locked_until),
                    # Both the window and a lockout set now end by now + window.
                    "expires_at": stmt.excluded.expires_at,
                },
            )
            await session.execute(stmt)
            await session.execute(
                update(t)
                .where(t.c.key == key, t.c.hits >= max_failures)
                .values(hits=0, window_start=now, locked_until=now + window_s)
            )
            await session.commit()
        await self._maybe_purge(now)

    async def clear(self, key: str) -> None:
        async with self._sessions()() as session:
            await session.execute(delete(_TABLE).where(_TABLE.c.key == key))
            await session.commit()

    # -- Housekeeping ----------------------------------------------------

    async def purge(self, now: float | None = None) -> int:
        """Delete rows whose ``expires_at`` has passed; returns how many."""
        now = self._clock() if now is None else now
        async with self._sessions()() as session:
            result = await session.execute(delete(_TABLE).where(_TABLE.c.expires_at < now))
            await session.commit()
        return int(getattr(result, "rowcount", 0) or 0)

    async def _maybe_purge(self, now: float) -> None:
        if now < self._next_purge:
            return
        self._next_purge = now + PURGE_INTERVAL_S
        try:
            await self.purge(now)
        except Exception as exc:  # housekeeping must never fail a request
            logger.debug("Purging expired shared_rate_limits rows failed: %s", exc)


class WarnOnce:
    """Log a warning at most once per *interval_s* (DB outages hit every request)."""

    def __init__(self, log: logging.Logger, interval_s: float = 60.0) -> None:
        self._log = log
        self._interval = interval_s
        self._next = 0.0

    def __call__(self, msg: str, *args: object) -> None:
        now = time.monotonic()
        if now >= self._next:
            self._next = now + self._interval
            self._log.warning(msg, *args)
