"""ASGI middleware for rate-limiting and CORS."""

from __future__ import annotations

import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from vpp.client_ip import parse_trusted_proxies, resolve_client_ip


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple token-bucket rate limiter keyed by client IP.

    Parameters
    ----------
    app : ASGI app
    requests_per_minute : int
        Maximum sustained requests per minute per IP.
    trusted_proxies : list of str
        CIDRs/addresses whose ``X-Forwarded-For`` / ``X-Real-IP`` headers
        identify the client (``VPP_TRUSTED_PROXIES``). Empty: headers are
        ignored and the TCP peer address is the key.
    """

    def __init__(
        self,
        app,
        *,
        requests_per_minute: int = 120,
        trusted_proxies: list[str] | tuple[str, ...] = (),
    ):
        super().__init__(app)
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

        tokens, last_time = self._buckets[ip]
        now = time.monotonic()
        elapsed = now - last_time
        tokens = min(self.capacity, tokens + elapsed * self.rate)

        if tokens < 1:
            return JSONResponse(
                {"detail": "Rate limit exceeded. Try again later."},
                status_code=429,
            )

        self._buckets[ip] = [tokens - 1, now]
        return await call_next(request)
