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
