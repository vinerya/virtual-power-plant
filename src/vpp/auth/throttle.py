"""Per-username failed-login throttle.

After ``VPP_LOGIN_MAX_FAILURES`` failed logins for one username, further
attempts for that username are refused with ``429`` for
``VPP_LOGIN_LOCKOUT_SECONDS`` -- without checking the password, so a
correct guess during the lockout reveals nothing. Failures older than the
lockout window are forgotten; a successful login clears the counter.

Where the state lives follows ``VPP_RATE_LIMIT_BACKEND`` (see
:mod:`vpp.auth.shared_limits`): with one worker it is in memory
(:class:`LoginThrottle`); with several it is in the ``shared_rate_limits``
table, so the limit holds across workers and replicas
(:class:`SharedLoginThrottle`, same semantics). If the database fails, the
throttle falls back to this process's in-memory state -- it is never
switched off -- and a lockout recorded either way is honoured. It
complements, not replaces, the per-IP rate limiter. Unknown usernames are
throttled exactly like real ones, so the 429 does not reveal whether an
account exists.

A lockout can be triggered by anyone who knows a username (that is the
trade-off of every account lockout); keep the lockout short. Set
``VPP_LOGIN_MAX_FAILURES=0`` to disable the throttle.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from vpp.auth.shared_limits import (
    DATABASE,
    SharedLimitStore,
    WarnOnce,
    resolve_backend,
    storage_key,
)
from vpp.settings import get_settings

logger = logging.getLogger(__name__)

#: Hard cap on tracked usernames so a flood of random names cannot grow
#: memory without bound; the oldest entries are dropped first.
MAX_TRACKED = 10_000


@dataclass
class _Entry:
    failures: int
    window_start: float
    locked_until: float = 0.0


class LoginThrottle:
    """In-memory, per-process throttle (the single-worker backend and the fallback)."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(username: str) -> str:
        return username.strip().lower()

    def retry_after(self, username: str) -> int | None:
        """Seconds until ``username`` may try again, or None if not locked."""
        settings = get_settings()
        if settings.login_max_failures <= 0:
            return None
        now = self._clock()
        with self._lock:
            entry = self._entries.get(self._key(username))
            if entry is None or entry.locked_until <= now:
                return None
            return max(1, math.ceil(entry.locked_until - now))

    def record_failure(self, username: str) -> None:
        settings = get_settings()
        if settings.login_max_failures <= 0:
            return
        window = max(1, settings.login_lockout_seconds)
        now = self._clock()
        key = self._key(username)
        with self._lock:
            entry = self._entries.pop(key, None)  # re-insert: most recent last
            if entry is None or now - entry.window_start > window:
                entry = _Entry(failures=0, window_start=now)
            entry.failures += 1
            if entry.failures >= settings.login_max_failures:
                entry.locked_until = now + window
                entry.failures = 0
                entry.window_start = now
            self._entries[key] = entry
            if len(self._entries) > MAX_TRACKED:
                self._prune(now, window)

    def record_success(self, username: str) -> None:
        with self._lock:
            self._entries.pop(self._key(username), None)

    def reset(self) -> None:
        with self._lock:
            self._entries.clear()

    def _prune(self, now: float, window: float) -> None:
        stale = [
            k
            for k, e in self._entries.items()
            if e.locked_until <= now and now - e.window_start > window
        ]
        for k in stale:
            del self._entries[k]
        while len(self._entries) > MAX_TRACKED:
            del self._entries[next(iter(self._entries))]


class SharedLoginThrottle:
    """The throttle the routes use: in memory or in the database.

    The backend is resolved from the settings on every call (``backend``
    forces one). In database mode the counters are shared by every worker;
    when the database fails, the failure is recorded in ``local`` (the
    in-memory :class:`LoginThrottle`) instead and a warning is logged.
    :meth:`retry_after` consults both, so a lockout recorded during an
    outage still holds, and :meth:`record_success` clears both.
    """

    def __init__(
        self,
        local: LoginThrottle,
        store: SharedLimitStore | None = None,
        *,
        backend: str | None = None,
    ) -> None:
        self.local = local
        self.store = store if store is not None else SharedLimitStore()
        self._backend = backend
        self._warn = WarnOnce(logger)

    def _shared(self) -> bool:
        backend = self._backend or resolve_backend(get_settings())
        return backend == DATABASE

    @staticmethod
    def _db_key(username: str) -> str:
        return storage_key("login", LoginThrottle._key(username))

    def _fallback(self, exc: Exception) -> None:
        self._warn(
            "Shared login throttle unavailable (%s: %s); throttling logins with this "
            "worker's in-memory state until the database answers again",
            type(exc).__name__,
            exc,
        )

    async def retry_after(self, username: str) -> int | None:
        """Seconds until ``username`` may try again, or None if not locked."""
        if get_settings().login_max_failures <= 0:
            return None
        local = self.local.retry_after(username)
        if not self._shared():
            return local
        try:
            shared = await self.store.login_retry_after(self._db_key(username))
        except Exception as exc:
            self._fallback(exc)
            return local
        if shared is None:
            return local
        return shared if local is None else max(shared, local)

    async def record_failure(self, username: str) -> None:
        settings = get_settings()
        if settings.login_max_failures <= 0:
            return
        if self._shared():
            try:
                await self.store.login_failure(
                    self._db_key(username),
                    settings.login_max_failures,
                    float(max(1, settings.login_lockout_seconds)),
                )
                return
            except Exception as exc:
                self._fallback(exc)
        self.local.record_failure(username)

    async def record_success(self, username: str) -> None:
        self.local.record_success(username)
        if not self._shared():
            return
        try:
            await self.store.clear(self._db_key(username))
        except Exception as exc:
            self._fallback(exc)

    def reset(self) -> None:
        """Forget the in-memory state (tests); database rows are left alone."""
        self.local.reset()


#: In-memory state of this process (the single-worker backend and the fallback).
login_throttle = LoginThrottle()

#: Used by ``POST /api/v1/auth/token`` and ``POST /api/v1/auth/password``.
shared_login_throttle = SharedLoginThrottle(login_throttle)
