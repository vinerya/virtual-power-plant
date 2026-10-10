"""Security response headers (vpp.api.security_headers)."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

import pytest
from fastapi import FastAPI, Response, WebSocket
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

from vpp.api.app import create_app
from vpp.api.security_headers import (
    DEFAULT_API_CSP,
    DEFAULT_DOCS_CSP,
    SecurityHeadersMiddleware,
)
from vpp.settings import get_settings

STATIC = {
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "no-referrer",
}


def _csp_directives(csp: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for part in csp.split(";"):
        tokens = part.split()
        if tokens:
            out[tokens[0]] = tokens[1:]
    return out


def _allowed(sources: list[str], url: str) -> bool:
    """Minimal CSP source matching for the absolute URLs the docs pages load."""
    if url.startswith("data:"):
        return "data:" in sources
    parts = urlsplit(url)
    if not parts.scheme:  # relative -> same origin
        return "'self'" in sources
    origin = f"{parts.scheme}://{parts.netloc}"
    return origin in sources


# -- the real app ------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_response_carries_static_headers_and_strict_csp(client: AsyncClient):
    r = await client.get("/health")
    assert r.status_code == 200
    for name, value in STATIC.items():
        assert r.headers[name] == value
    assert r.headers["content-security-policy"] == DEFAULT_API_CSP
    # Plain http request, no VPP_HSTS_ENABLED: no HSTS.
    assert "strict-transport-security" not in r.headers
    # /health is not under /api/: caching is left to the route.
    assert "cache-control" not in r.headers


@pytest.mark.asyncio
async def test_auth_and_api_responses_are_not_cached(client: AsyncClient):
    r = await client.post(
        "/api/v1/auth/token", data={"username": "nobody", "password": "wrong-password"}
    )
    assert r.status_code == 401
    assert r.headers["cache-control"] == "no-store"
    assert r.headers["content-security-policy"] == DEFAULT_API_CSP

    r = await client.get("/api/v1/auth/me")
    assert r.status_code == 401
    assert r.headers["cache-control"] == "no-store"


@pytest.mark.asyncio
async def test_authenticated_api_response_is_not_cached(client: AsyncClient, auth_headers):
    r = await client.get("/api/v1/auth/me", headers=auth_headers)
    assert r.status_code == 200
    assert r.headers["cache-control"] == "no-store"
    assert r.headers["x-content-type-options"] == "nosniff"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/docs", "/redoc", "/docs/oauth2-redirect"])
async def test_docs_pages_get_a_csp_that_allows_their_assets(client: AsyncClient, path):
    r = await client.get(path)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    csp = r.headers["content-security-policy"]
    assert csp == DEFAULT_DOCS_CSP
    directives = _csp_directives(csp)
    assert directives["frame-ancestors"] == ["'none'"]
    assert r.headers["x-frame-options"] == "DENY"

    html = r.text
    scripts = re.findall(r"<script[^>]*\ssrc=\"([^\"]+)\"", html)
    styles = re.findall(r"<link[^>]*rel=\"stylesheet\"[^>]*href=\"([^\"]+)\"", html)
    styles += re.findall(r"<link[^>]*href=\"([^\"]+)\"[^>]*rel=\"stylesheet\"", html)
    icons = re.findall(r"<link[^>]*rel=\"shortcut icon\"[^>]*href=\"([^\"]+)\"", html)
    for url in scripts:
        assert _allowed(directives["script-src"], url), url
    for url in styles:
        assert _allowed(directives["style-src"], url), url
    for url in icons:
        assert _allowed(directives["img-src"], url), url
    if re.search(r"<script>\s*\S", html):  # inline bootstrap script
        assert "'unsafe-inline'" in directives["script-src"]
    if "<style" in html:
        assert "'unsafe-inline'" in directives["style-src"]
    if path == "/redoc":
        assert scripts and styles  # the ReDoc bundle and Google Fonts
        assert "https://fonts.gstatic.com" in directives["font-src"]
        assert "blob:" in directives["worker-src"]
    if path == "/docs":
        assert scripts and styles
        assert "'self'" in directives["connect-src"]  # /openapi.json, "Try it out"


@pytest.mark.asyncio
async def test_openapi_json_keeps_the_strict_csp(client: AsyncClient):
    r = await client.get("/openapi.json")
    assert r.status_code == 200
    assert r.headers["content-security-policy"] == DEFAULT_API_CSP


@pytest.mark.asyncio
async def test_cors_preflight_and_exposed_headers_still_work(client: AsyncClient):
    origin = get_settings().cors_origins[0]
    r = await client.options(
        "/api/v1/resources",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization",
        },
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == origin
    assert r.headers["x-content-type-options"] == "nosniff"

    r = await client.get("/health", headers={"Origin": origin})
    assert r.headers["access-control-allow-origin"] == origin
    assert "X-Total-Count" in r.headers["access-control-expose-headers"]


def _app_with_env(monkeypatch, **env: str) -> FastAPI:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    try:
        return create_app()
    finally:
        get_settings.cache_clear()


async def _get(app, path: str = "/health", *, base_url: str = "http://test", **kw):
    transport = ASGITransport(app=app, client=kw.pop("peer", ("127.0.0.1", 40000)))
    async with AsyncClient(transport=transport, base_url=base_url) as c:
        return await c.get(path, **kw)


@pytest.mark.asyncio
async def test_settings_can_disable_the_headers(monkeypatch):
    app = _app_with_env(monkeypatch, VPP_SECURITY_HEADERS_ENABLED="false")
    r = await _get(app)
    assert r.status_code == 200
    assert "content-security-policy" not in r.headers
    assert "x-frame-options" not in r.headers


@pytest.mark.asyncio
async def test_settings_override_csp_and_force_hsts(monkeypatch):
    app = _app_with_env(
        monkeypatch,
        VPP_CONTENT_SECURITY_POLICY="default-src 'none'",
        VPP_HSTS_ENABLED="true",
        VPP_HSTS_MAX_AGE_SECONDS="600",
        VPP_HSTS_INCLUDE_SUBDOMAINS="true",
    )
    r = await _get(app)
    assert r.headers["content-security-policy"] == "default-src 'none'"
    assert r.headers["strict-transport-security"] == "max-age=600; includeSubDomains"


@pytest.mark.asyncio
async def test_hsts_auto_follows_trusted_proxy_forwarded_proto(monkeypatch):
    app = _app_with_env(monkeypatch, VPP_TRUSTED_PROXIES='["10.0.0.5/32"]')
    xfp = {"X-Forwarded-Proto": "https"}
    trusted = await _get(app, headers=xfp, peer=("10.0.0.5", 40000))
    assert trusted.headers["strict-transport-security"] == "max-age=31536000"
    spoofed = await _get(app, headers=xfp, peer=("203.0.113.9", 40000))
    assert "strict-transport-security" not in spoofed.headers
    plain = await _get(app, peer=("10.0.0.5", 40000))
    assert "strict-transport-security" not in plain.headers


@pytest.mark.asyncio
async def test_hsts_disabled_wins_over_https(monkeypatch):
    app = _app_with_env(monkeypatch, VPP_HSTS_ENABLED="false")
    r = await _get(app, base_url="https://test")
    assert "strict-transport-security" not in r.headers


# -- the middleware on its own ------------------------------------------------


def _mini_app(**kwargs) -> FastAPI:
    app = FastAPI()

    @app.get("/plain")
    async def plain() -> dict[str, str]:
        return {"ok": "yes"}

    @app.get("/api/v1/cached")
    async def cached(response: Response) -> dict[str, str]:
        response.headers["Cache-Control"] = "max-age=60"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        return {"ok": "yes"}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept(subprotocol="bearer")
        await websocket.send_text(await websocket.receive_text())
        await websocket.close()

    app.add_middleware(SecurityHeadersMiddleware, **kwargs)
    return app


@pytest.mark.asyncio
async def test_https_request_gets_hsts_by_default():
    r = await _get(_mini_app(), "/plain", base_url="https://test")
    assert r.headers["strict-transport-security"] == "max-age=31536000"
    r = await _get(_mini_app(), "/plain")
    assert "strict-transport-security" not in r.headers


@pytest.mark.asyncio
async def test_hsts_preload_value():
    app = _mini_app(hsts=True, hsts_include_subdomains=True, hsts_preload=True)
    r = await _get(app, "/plain")
    assert r.headers["strict-transport-security"] == (
        "max-age=31536000; includeSubDomains; preload"
    )


@pytest.mark.asyncio
async def test_route_set_headers_are_kept():
    r = await _get(_mini_app(), "/api/v1/cached")
    assert r.headers["cache-control"] == "max-age=60"
    assert r.headers["content-security-policy"] == "default-src 'self'"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers.get_list("cache-control") == ["max-age=60"]


@pytest.mark.asyncio
async def test_error_responses_get_headers():
    r = await _get(_mini_app(), "/api/v1/missing")
    assert r.status_code == 404
    assert r.headers["cache-control"] == "no-store"
    assert r.headers["content-security-policy"] == DEFAULT_API_CSP


def test_websocket_upgrade_is_unaffected():
    with (
        TestClient(_mini_app(hsts=True)) as tc,
        tc.websocket_connect("/ws", subprotocols=["bearer", "tok"]) as ws,
    ):
        assert ws.accepted_subprotocol == "bearer"
        ws.send_text("ping")
        assert ws.receive_text() == "ping"


def test_real_app_websocket_rejection_is_unaffected(monkeypatch):
    """The app's /api/v1/ws still refuses an unauthenticated socket with 1008."""
    from starlette.websockets import WebSocketDisconnect

    from vpp.api import websocket as ws_module

    app = _mini_app()
    app.add_api_websocket_route("/api/v1/ws", ws_module.websocket_endpoint)
    with (
        TestClient(app) as tc,
        pytest.raises(WebSocketDisconnect) as exc,
        tc.websocket_connect("/api/v1/ws") as ws,
    ):
        ws.receive_text()
    assert exc.value.code == ws_module.WS_POLICY_VIOLATION
