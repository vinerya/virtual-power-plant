"""Tests for authentication and authorization."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_unauthenticated_access_rejected(client: AsyncClient):
    """Endpoints requiring auth should return 401 without credentials."""
    resp = await client.get("/api/v1/resources/")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_me_endpoint(client: AsyncClient, auth_headers: dict):
    resp = await client.get("/api/v1/auth/me", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["username"] == "testadmin"
    assert body["role"] == "admin"


@pytest.mark.asyncio
async def test_invalid_token_rejected(client: AsyncClient):
    resp = await client.get(
        "/api/v1/resources/",
        headers={"Authorization": "Bearer invalid.token.here"},
    )
    assert resp.status_code == 401
