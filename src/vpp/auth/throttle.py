"""Per-username failed-login throttle.

After ``VPP_LOGIN_MAX_FAILURES`` failed logins for one username, further
attempts for that username are refused with ``429`` for
``VPP_LOGIN_LOCKOUT_SECONDS`` -- without checking the password, so a
correct guess during the lockout reveals nothing. Failures older than the
lockout window are forgotten; a successful login clears the counter.

The state is **in memory and per process**: with several uvicorn workers
or replicas an attacker gets ``max_failures`` tries per process per
window. It complements, not replaces, the per-IP rate limiter. Unknown
usernames are throttled exactly like real ones, so the 429 does not reveal
whether an account exists.

A lockout can be triggered by anyone who knows a username (that is the
trade-off of every account lockout); keep the lockout short. Set
``VPP_LOGIN_MAX_FAILURES=0`` to disable the throttle.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from vpp.settings import get_settings

#: Hard cap on tracked usernames so a flood of random names cannot grow
#: memory without bound; the oldest entries are dropped first.
MAX_TRACKED = 10_000


@dataclass
class _Entry:
    failures: int
    window_start: float
    locked_until: float = 0.0


class LoginThrottle:
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


#: Process-wide instance used by ``POST /api/v1/auth/token``.
login_throttle = LoginThrottle()
