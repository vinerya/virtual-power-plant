"""Collection routes answer with or without a trailing slash -- no 307."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from _portal_helpers import uid

if TYPE_CHECKING:
    from httpx import AsyncClient


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/resources",
        "/api/v1/resources/",
        "/api/v1/optimization/history",
        "/api/v1/optimization/history/",
        "/api/v1/sites",
        "/api/v1/sites/",
        "/api/v1/protocols",
        "/api/v1/protocols/",
        "/api/v1/alerts",
        "/api/v1/alerts/",
    ],
)
async def test_collection_routes_without_redirect(
    client: AsyncClient, auth_headers: dict, path: str
):
    resp = await client.get(path, headers=auth_headers)
    assert resp.status_code == 200, (path, resp.status_code, resp.headers.get("location"))


@pytest.mark.asyncio
async def test_create_with_and_without_trailing_slash(client: AsyncClient, auth_headers: dict):
    for path in ("/api/v1/resources", "/api/v1/resources/"):
        resp = await client.post(
            path,
            json={"name": uid("slash"), "resource_type": "solar", "rated_power": 1.0},
            headers=auth_headers,
        )
        assert resp.status_code == 201, (path, resp.status_code)


@pytest.mark.asyncio
async def test_unknown_paths_still_404(client: AsyncClient, auth_headers: dict):
    assert (await client.get("/api/v1/nope", headers=auth_headers)).status_code == 404
    assert (await client.get("/api/v1/nope/", headers=auth_headers)).status_code == 404


@pytest.mark.asyncio
async def test_method_mismatch_is_405_not_rewritten(client: AsyncClient, auth_headers: dict):
    assert (await client.delete("/api/v1/resources", headers=auth_headers)).status_code == 405
    assert (await client.delete("/api/v1/resources/", headers=auth_headers)).status_code == 405


def test_route_match_answers_are_cached_and_bounded():
    """Regression (found with `vpp loadtest`): walking every route on every
    request cost ~0.8 ms, ~10 % of a worker's CPU at ~100 requests/s."""
    from fastapi import FastAPI

    from vpp.api.middleware import TrailingSlashMiddleware

    app = FastAPI()

    @app.get("/items/{item_id}")
    async def item(item_id: str) -> dict:  # pragma: no cover - not called
        return {}

    mw = TrailingSlashMiddleware(app, router=app.router)
    calls = 0
    original = mw._match_routes

    def counting(scope):
        nonlocal calls
        calls += 1
        return original(scope)

    mw._match_routes = counting  # type: ignore[method-assign]
    scope = {"type": "http", "path": "/items/1", "method": "GET", "root_path": ""}
    assert mw._matches(scope)
    assert mw._matches(dict(scope, method="DELETE"))  # any method counts
    assert not mw._matches(dict(scope, path="/nope"))
    assert calls == 2  # the second /items/1 lookup came from the cache

    mw.CACHE_SIZE = 3
    for i in range(10):
        mw._matches(dict(scope, path=f"/items/{i}"))
    assert len(mw._cache) == 3
