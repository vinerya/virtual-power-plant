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


# ---------------------------------------------------------------------------
# POST /api/v1/auth/token -- credential transport
# ---------------------------------------------------------------------------

_CREDS = {"username": "testadmin", "password": "adminpassword123"}


@pytest.mark.asyncio
async def test_token_form_body(client: AsyncClient, admin_user):
    resp = await client.post("/api/v1/auth/token", data={**_CREDS, "grant_type": "password"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["access_token"]
    assert "Deprecation" not in resp.headers


@pytest.mark.asyncio
async def test_token_json_body(client: AsyncClient, admin_user):
    resp = await client.post("/api/v1/auth/token", json=_CREDS)
    assert resp.status_code == 200, resp.text
    assert resp.json()["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_token_query_params_deprecated(client: AsyncClient, admin_user):
    resp = await client.post("/api/v1/auth/token", params=_CREDS)
    assert resp.status_code == 200, resp.text
    assert resp.headers["Deprecation"] == "true"


@pytest.mark.asyncio
async def test_token_rejects_bad_requests(client: AsyncClient, admin_user):
    bad_pw = await client.post("/api/v1/auth/token", data={**_CREDS, "password": "nope"})
    assert bad_pw.status_code == 401
    bad_json = await client.post("/api/v1/auth/token", json={**_CREDS, "password": "nope"})
    assert bad_json.status_code == 401
    missing = await client.post("/api/v1/auth/token", data={"username": "testadmin"})
    assert missing.status_code == 422
    grant = await client.post(
        "/api/v1/auth/token", data={**_CREDS, "grant_type": "client_credentials"}
    )
    assert grant.status_code == 400
    wrong_type = await client.post(
        "/api/v1/auth/token", content=b"x", headers={"content-type": "text/plain"}
    )
    assert wrong_type.status_code == 415
    nothing = await client.post("/api/v1/auth/token")
    assert nothing.status_code == 422


def test_token_openapi_documents_form_and_json(app):
    body = app.openapi()["paths"]["/api/v1/auth/token"]["post"]["requestBody"]["content"]
    assert set(body) == {"application/x-www-form-urlencoded", "application/json"}
