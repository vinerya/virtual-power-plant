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


@pytest.mark.asyncio
async def test_viewer_cannot_self_issue_admin_api_key(client: AsyncClient, viewer_headers: dict):
    """A non-admin user must not be able to mint an API key with a higher role."""
    resp = await client.post(
        "/api/v1/auth/api-key",
        json={"name": "escalation-attempt", "role": "admin"},
        headers=viewer_headers,
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_viewer_can_self_issue_own_role_api_key(client: AsyncClient, viewer_headers: dict):
    """A non-admin user can still mint a key scoped to their own role."""
    resp = await client.post(
        "/api/v1/auth/api-key",
        json={"name": "own-role-key", "role": "viewer"},
        headers=viewer_headers,
    )
    assert resp.status_code == 201
    assert resp.json()["role"] == "viewer"


@pytest.mark.asyncio
async def test_admin_can_issue_lower_role_api_key(client: AsyncClient, auth_headers: dict):
    """Admins are unrestricted — they can still mint a key with any role."""
    resp = await client.post(
        "/api/v1/auth/api-key",
        json={"name": "scoped-down-key", "role": "viewer"},
        headers=auth_headers,
    )
    assert resp.status_code == 201
    assert resp.json()["role"] == "viewer"
