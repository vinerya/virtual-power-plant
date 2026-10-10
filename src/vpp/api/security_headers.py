"""Security response headers for the API (``VPP_SECURITY_HEADERS_*``).

A plain ASGI middleware (like the observability ones) that adds, to every
HTTP response, unless the response already carries the header:

* ``X-Content-Type-Options: nosniff``
* ``X-Frame-Options: DENY``
* ``Referrer-Policy: no-referrer``
* ``Content-Security-Policy``: ``default-src 'none'; frame-ancestors 'none'``
  for API responses (JSON never needs to load anything), and a relaxed policy
  for the interactive docs pages (``/docs``, ``/docs/oauth2-redirect``,
  ``/redoc``), whose HTML loads Swagger UI / ReDoc from cdn.jsdelivr.net and
  runs an inline bootstrap script.
* ``Cache-Control: no-store`` on ``/api/*`` responses (tokens, API keys and
  per-user data must not land in browser or proxy caches).
* ``Strict-Transport-Security`` -- see :func:`hsts_wanted`.

WebSocket connections (``scope["type"] == "websocket"``) pass through
untouched, and CORS keeps working: the CORS middleware runs inside this one
and its headers (``Access-Control-*``, ``X-Total-Count`` exposure) are never
modified.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vpp.client_ip import is_trusted_peer, parse_trusted_proxies

if TYPE_CHECKING:
    from collections.abc import Iterable

    from starlette.types import ASGIApp, Message, Receive, Scope, Send

    from vpp.client_ip import IPNetwork

#: CSP for every response except the docs pages: an API response is data,
#: it never loads scripts, styles or frames, and must not be framed.
DEFAULT_API_CSP = "default-src 'none'; frame-ancestors 'none'"

#: CSP for the Swagger UI / ReDoc pages FastAPI renders. They pull their
#: bundle and stylesheet from cdn.jsdelivr.net, ReDoc its fonts from Google
#: Fonts, both a favicon from fastapi.tiangolo.com; Swagger UI boots from an
#: inline <script>, ReDoc runs its search in a blob: worker. The pages fetch
#: /openapi.json (and "Try it out" calls the API) from the same origin.
DEFAULT_DOCS_CSP = (
    "default-src 'none'; "
    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
    "font-src 'self' data: https://fonts.gstatic.com; "
    "img-src 'self' data: https://fastapi.tiangolo.com https://cdn.redoc.ly; "
    "connect-src 'self'; "
    "worker-src 'self' blob:; "
    "frame-ancestors 'none'; "
    "base-uri 'none'; "
    "form-action 'self'"
)

_HEADER_CSP = b"content-security-policy"
_HEADER_CACHE = b"cache-control"
_HEADER_HSTS = b"strict-transport-security"
_XFP = b"x-forwarded-proto"


def hsts_value(max_age: int, *, include_subdomains: bool, preload: bool) -> str:
    value = f"max-age={max_age}"
    if include_subdomains:
        value += "; includeSubDomains"
    if preload:
        value += "; preload"
    return value


def request_is_https(scope: Scope, trusted: tuple[IPNetwork, ...]) -> bool:
    """Whether the client reached us over TLS.

    True when the ASGI scheme is ``https`` (TLS terminated by uvicorn, or
    uvicorn's own ``--proxy-headers`` already applied ``X-Forwarded-Proto``),
    or when the TCP peer is one of ``VPP_TRUSTED_PROXIES`` and the nearest
    ``X-Forwarded-Proto`` entry is ``https``. The header is ignored from any
    other peer, since a client can send it.
    """
    if scope.get("scheme") in ("https", "wss"):
        return True
    client = scope.get("client")
    if not is_trusted_peer(client[0] if client else None, trusted):
        return False
    values: list[bytes] = [v for k, v in scope.get("headers") or [] if k == _XFP]
    if not values:
        return False
    nearest = values[-1].decode("latin-1").split(",")[-1].strip().lower()
    return nearest == "https"


class SecurityHeadersMiddleware:
    """Add the security headers described in the module docstring.

    ``hsts`` is ``True`` (always send), ``False`` (never) or ``None`` (send
    only on requests that arrived over HTTPS, see :func:`request_is_https`).
    Headers a route sets itself are left alone.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        csp: str = DEFAULT_API_CSP,
        docs_csp: str = DEFAULT_DOCS_CSP,
        docs_paths: Iterable[str] = (),
        hsts: bool | None = None,
        hsts_max_age: int = 31_536_000,
        hsts_include_subdomains: bool = False,
        hsts_preload: bool = False,
        trusted_proxies: Iterable[str] = (),
        no_store_prefixes: tuple[str, ...] = ("/api/",),
    ) -> None:
        self.app = app
        self.csp = csp.encode("latin-1")
        self.docs_csp = docs_csp.encode("latin-1")
        self.docs_paths = frozenset(p for p in docs_paths if p)
        self.hsts = hsts
        self.hsts_header = hsts_value(
            hsts_max_age, include_subdomains=hsts_include_subdomains, preload=hsts_preload
        ).encode("latin-1")
        self.trusted = parse_trusted_proxies(trusted_proxies)
        self.no_store_prefixes = no_store_prefixes
        self.static: list[tuple[bytes, bytes]] = [
            (b"x-content-type-options", b"nosniff"),
            (b"x-frame-options", b"DENY"),
            (b"referrer-policy", b"no-referrer"),
        ]

    def _send_hsts(self, scope: Scope) -> bool:
        if self.hsts is None:
            return request_is_https(scope, self.trusted)
        return self.hsts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path: str = scope.get("path", "")
        extra = list(self.static)
        is_docs = path in self.docs_paths or path.rstrip("/") in self.docs_paths
        extra.append((_HEADER_CSP, self.docs_csp if is_docs else self.csp))
        if path.startswith(self.no_store_prefixes):
            extra.append((_HEADER_CACHE, b"no-store"))
        if self._send_hsts(scope):
            extra.append((_HEADER_HSTS, self.hsts_header))

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                present = {k.lower() for k, _ in headers}
                headers.extend((k, v) for k, v in extra if k not in present)
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)
