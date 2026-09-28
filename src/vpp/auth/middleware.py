"""ASGI middleware for rate-limiting and CORS."""

from __future__ import annotations

import logging
import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from vpp.auth.shared_limits import DATABASE, MEMORY, SharedLimitStore, WarnOnce, storage_key
from vpp.client_ip import parse_trusted_proxies, resolve_client_ip

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-client-IP rate limiter.

    ``backend="memory"`` (one worker): a token bucket per IP in this
    process, no database round trip. ``backend="database"`` (several
    workers, see :mod:`vpp.auth.shared_limits`): a one-minute sliding-window
    counter per IP in ``shared_rate_limits``, shared by every worker. If the
    database fails the request is let through and a warning is logged (at
    most once a minute): an outage must not take the API down with it.

    Parameters
    ----------
    app : ASGI app
    requests_per_minute : int
        Maximum sustained requests per minute per IP.
    trusted_proxies : list of str
        CIDRs/addresses whose ``X-Forwarded-For`` / ``X-Real-IP`` headers
        identify the client (``VPP_TRUSTED_PROXIES``). Empty: headers are
        ignored and the TCP peer address is the key.
    backend : "memory" or "database"
        Resolved ``VPP_RATE_LIMIT_BACKEND``.
    store : SharedLimitStore, optional
        The database store (default: one on the application's database).
    """

    def __init__(
        self,
        app,
        *,
        requests_per_minute: int = 120,
        trusted_proxies: list[str] | tuple[str, ...] = (),
        backend: str = MEMORY,
        store: SharedLimitStore | None = None,
    ):
        super().__init__(app)
        if backend not in (MEMORY, DATABASE):
            raise ValueError(f"unknown rate limit backend {backend!r}")
        self.backend = backend
        self.store = store if store is not None else SharedLimitStore()
        self._warn = WarnOnce(logger)
        self.trusted = parse_trusted_proxies(trusted_proxies)
        self.rate = requests_per_minute / 60.0  # tokens per second
        self.capacity = requests_per_minute
        self._buckets: dict[str, list[float]] = defaultdict(
            lambda: [float(self.capacity), time.monotonic()]
        )

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        forwarded = request.headers.getlist("x-forwarded-for")
        ip = resolve_client_ip(
            request.client.host if request.client else None,
            ",".join(forwarded) if forwarded else None,
            request.headers.get("x-real-ip"),
            self.trusted,
        )

        allowed = await self._allow_shared(ip) if self.backend == DATABASE else self._allow(ip)
        if not allowed:
            return JSONResponse(
                {"detail": "Rate limit exceeded. Try again later."},
                status_code=429,
            )
        return await call_next(request)

    def _allow(self, ip: str) -> bool:
        tokens, last_time = self._buckets[ip]
        now = time.monotonic()
        elapsed = now - last_time
        tokens = min(self.capacity, tokens + elapsed * self.rate)

        if tokens < 1:
            return False
        self._buckets[ip] = [tokens - 1, now]
        return True

    async def _allow_shared(self, ip: str) -> bool:
        try:
            return await self.store.hit(storage_key("http", ip), self.capacity, 60.0)
        except Exception as exc:
            self._warn(
                "Shared rate limiter unavailable (%s: %s); letting requests through "
                "unthrottled until the database answers again",
                type(exc).__name__,
                exc,
            )
            return True
