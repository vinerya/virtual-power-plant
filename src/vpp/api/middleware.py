"""Observability middleware: request-id correlation and HTTP metrics.

Both are plain ASGI middleware (not ``BaseHTTPMiddleware``) so they add no
extra task / body-buffering overhead and see every response, including
ones produced by other middleware (e.g. rate-limit 429s) when installed
outermost.
"""

from __future__ import annotations

import re
import time
import uuid
from typing import TYPE_CHECKING, Any

import structlog

from vpp.metrics import MetricsCollector, metrics_collector

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_HEADER_BYTES = REQUEST_ID_HEADER.lower().encode("latin-1")

# Accept caller-supplied ids only if they are short and made of safe
# characters -- they end up in logs and response headers.
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:\-]{1,128}$")

UNMATCHED_ROUTE = "<unmatched>"

access_logger = structlog.get_logger("vpp.access")


def _header(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers") or []:
        if key == name:
            try:
                decoded: str = value.decode("latin-1")
                return decoded
            except UnicodeDecodeError:  # pragma: no cover - latin-1 never fails
                return None
    return None


def route_template(scope: Scope) -> str:
    """Return the matched route's path template, or ``<unmatched>``.

    Starlette's router stores the matched route on ``scope["route"]``; its
    ``path`` is the template (``/api/v1/resources/{resource_id}``), which
    keeps metric label cardinality bounded regardless of request paths.
    """
    route = scope.get("route")
    path = getattr(route, "path", None)
    if isinstance(path, str) and path:
        return path
    return UNMATCHED_ROUTE


class RequestIdMiddleware:
    """Assign every HTTP request an id and bind it into the log context.

    * Reuses an incoming ``X-Request-ID`` if it looks sane, otherwise
      generates a UUID4 hex.
    * Binds ``request_id`` into ``structlog.contextvars`` for the duration of
      the request, so every structlog line -- and, through
      :func:`vpp.logging.configure_logging`'s ``foreign_pre_chain``, every
      stdlib ``logging`` line -- carries it.
    * Echoes it back as the ``X-Request-ID`` response header and exposes it
      as ``request.state.request_id``.
    * Emits one ``vpp.access`` log line per request (method, route template,
      status, duration).
    """

    def __init__(self, app: ASGIApp, *, access_log: bool = True) -> None:
        self.app = app
        self.access_log = access_log

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = _header(scope, _REQUEST_ID_HEADER_BYTES)
        request_id = (
            incoming if incoming and _VALID_REQUEST_ID.match(incoming) else uuid.uuid4().hex
        )
        scope.setdefault("state", {})["request_id"] = request_id

        status_code = 500
        start = time.perf_counter()

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = [
                    (k, v)
                    for k, v in message.get("headers", [])
                    if k.lower() != _REQUEST_ID_HEADER_BYTES
                ]
                headers.append((_REQUEST_ID_HEADER_BYTES, request_id.encode("latin-1")))
                message["headers"] = headers
            await send(message)

        with structlog.contextvars.bound_contextvars(request_id=request_id):
            try:
                await self.app(scope, receive, send_wrapper)
            finally:
                if self.access_log:
                    access_logger.info(
                        "http_request",
                        method=scope.get("method"),
                        route=route_template(scope),
                        status=status_code,
                        duration_ms=round((time.perf_counter() - start) * 1000, 2),
                    )


class PrometheusMiddleware:
    """Record request count / latency per (method, route template, status)."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        collector: MetricsCollector | None = None,
        exclude_paths: tuple[str, ...] = (),
    ) -> None:
        self.app = app
        self.collector = collector or metrics_collector
        self.exclude_paths = set(exclude_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") in self.exclude_paths:
            await self.app(scope, receive, send)
            return

        status_code = 500
        start = time.perf_counter()

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        self.collector.request_started()
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            self.collector.request_finished()
            self.collector.record_api_request(
                scope.get("method", "GET"),
                route_template(scope),
                status_code,
                time.perf_counter() - start,
            )


class TrailingSlashMiddleware:
    """Serve ``/x`` and ``/x/`` from the same route without a redirect.

    Starlette's router answers a path that only matches with the trailing
    slash toggled with a ``307`` redirect. Browsers re-send the request
    (fine), but API clients that do not follow redirects -- or that drop the
    ``Authorization`` header on the hop, or sit behind a proxy that rewrites
    the ``Location`` host -- break on it. This middleware resolves the
    toggled path *internally* instead: when no route matches the request
    path but one matches it with the trailing slash added/removed, the scope
    is rewritten before routing. A path that matches as-is (even only with a
    different method, i.e. a 405) is never touched, so explicit routes win.
    """

    def __init__(self, app: ASGIApp, *, router: Any) -> None:
        self.app = app
        self.router = router

    def _matches(self, scope: Scope) -> bool:
        from starlette.routing import Match

        for route in self.router.routes:
            match, _ = route.matches(scope)
            if match is not Match.NONE:
                return True
        return False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            path: str = scope["path"]
            if path != "/" and not self._matches(scope):
                alt = path[:-1] if path.endswith("/") else path + "/"
                alt_scope = dict(scope)
                alt_scope["path"] = alt
                raw = scope.get("raw_path")
                if isinstance(raw, bytes):
                    alt_scope["raw_path"] = raw[:-1] if raw.endswith(b"/") else raw + b"/"
                if self._matches(alt_scope):
                    scope = alt_scope
        await self.app(scope, receive, send)
