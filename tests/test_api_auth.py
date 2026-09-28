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


@pytest.mark.asyncio
async def test_api_key_is_limited_to_its_own_role(client: AsyncClient, auth_headers: dict):
    """A viewer-scoped key minted by an admin must not carry admin rights."""
    minted = await client.post(
        "/api/v1/auth/api-key",
        json={"name": "viewer-scope-enforced", "role": "viewer"},
        headers=auth_headers,
    )
    assert minted.status_code == 201
    key_headers = {"X-API-Key": minted.json()["key"]}

    # Reads a viewer may make still work ...
    assert (await client.get("/api/v1/resources", headers=key_headers)).status_code == 200
    me = await client.get("/api/v1/auth/me", headers=key_headers)
    assert me.status_code == 200
    assert me.json()["role"] == "viewer"

    # ... but admin/operator actions are refused.
    escalate = await client.post(
        "/api/v1/auth/register",
        json={"username": "minted-by-viewer-key", "password": "s3cret-pass!", "role": "admin"},
        headers=key_headers,
    )
    assert escalate.status_code == 403
    write = await client.post(
        "/api/v1/resources",
        json={"name": "viewer-key-batt", "resource_type": "battery", "rated_power": 5.0},
        headers=key_headers,
    )
    assert write.status_code == 403

    # The narrowing is per-request only: the owning admin keeps full rights.
    owner = await client.get("/api/v1/auth/me", headers=auth_headers)
    assert owner.json()["role"] == "admin"


def test_effective_api_key_role_takes_the_lesser_privilege():
    from vpp.auth.security import effective_api_key_role

    assert effective_api_key_role("viewer", "admin") == "viewer"
    assert effective_api_key_role("admin", "operator") == "operator"  # owner demoted
    assert effective_api_key_role("operator", "operator") == "operator"
    assert effective_api_key_role("admin", "customer") == "customer"
    assert effective_api_key_role("researcher", "admin") == "researcher"
    assert effective_api_key_role("bogus", "admin") == "bogus"  # unknown ranks lowest


@pytest.mark.asyncio
async def test_api_key_header_name_follows_setting(
    client: AsyncClient, auth_headers: dict, monkeypatch
):
    """``VPP_API_KEY_HEADER`` renames the header API keys are read from."""
    from vpp.settings import get_settings

    minted = await client.post(
        "/api/v1/auth/api-key",
        json={"name": "custom-header-key", "role": "viewer"},
        headers=auth_headers,
    )
    assert minted.status_code == 201
    key = minted.json()["key"]

    monkeypatch.setenv("VPP_API_KEY_HEADER", "X-VPP-Key")
    get_settings.cache_clear()
    try:
        custom = await client.get("/api/v1/auth/me", headers={"X-VPP-Key": key})
        assert custom.status_code == 200
        # The default header name is no longer consulted.
        default = await client.get("/api/v1/auth/me", headers={"X-API-Key": key})
        assert default.status_code == 401
    finally:
        monkeypatch.delenv("VPP_API_KEY_HEADER")
        get_settings.cache_clear()
    restored = await client.get("/api/v1/auth/me", headers={"X-API-Key": key})
    assert restored.status_code == 200
