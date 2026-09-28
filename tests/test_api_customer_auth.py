"""Customer role, JWT audience claim and deny-by-default RBAC."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from _portal_helpers import create_customer, isolated_client, user_headers
from jose import jwt

from vpp.auth.security import create_access_token

if TYPE_CHECKING:
    from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def client(app):
    """Per-test client IP so these API-heavy tests don't drain the shared rate limit."""
    async with isolated_client(app) as c:
        yield c


async def test_customer_login_issues_customer_audience(client: AsyncClient, auth_headers: dict):
    cid, headers = await create_customer(client, auth_headers)
    token = headers["Authorization"].split()[1]
    claims = jwt.get_unverified_claims(token)
    assert claims["aud"] == "customer"
    assert claims["role"] == "customer"

    me = await client.get("/api/v1/auth/me", headers=headers)
    assert me.status_code == 200, me.text
    assert me.json()["audience"] == "customer"
    assert me.json()["id"] == cid


async def test_operator_login_issues_operator_audience(client: AsyncClient, admin_user):
    resp = await client.post(
        "/api/v1/auth/token", params={"username": "testadmin", "password": "adminpassword123"}
    )
    assert resp.status_code == 200
    token = resp.json()["access_token"]
    assert jwt.get_unverified_claims(token)["aud"] == "operator"
    me = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["audience"] == "operator"
    assert me.json()["role"] == "admin"


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/resources/",
        "/api/v1/tariffs",
        "/api/v1/config",
        "/api/v1/config/schema",
        "/api/v1/customers",
        "/api/v1/programs",
    ],
)
async def test_customer_denied_on_operator_endpoints(
    client: AsyncClient, auth_headers: dict, path
):
    _cid, headers = await create_customer(client, auth_headers)
    resp = await client.get(path, headers=headers)
    assert resp.status_code == 403, (path, resp.text)


async def test_operator_denied_on_customer_portal(client: AsyncClient, db_session):
    headers = await user_headers(db_session, "operator")
    for path in (
        "/api/v1/customer/me",
        "/api/v1/customer/me/bill",
        "/api/v1/customer/me/devices",
        "/api/v1/customer/programs",
    ):
        resp = await client.get(path, headers=headers)
        assert resp.status_code == 403, path


async def test_audience_mismatch_rejected(client: AsyncClient, admin_user):
    # A token claiming the customer audience for an admin account is refused.
    token = create_access_token(
        {
            "sub": admin_user.id,
            "username": admin_user.username,
            "role": admin_user.role,
            "aud": "customer",
        }
    )
    resp = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


async def test_unknown_audience_rejected(client: AsyncClient, admin_user):
    token = create_access_token(
        {
            "sub": admin_user.id,
            "username": admin_user.username,
            "role": admin_user.role,
            "aud": "somebody-else",
        }
    )
    resp = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


async def test_legacy_token_without_audience_still_works(client: AsyncClient, auth_headers: dict):
    resp = await client.get("/api/v1/auth/me", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["audience"] == "operator"


async def test_inactive_user_cannot_log_in(client: AsyncClient, auth_headers: dict, db_session):
    from vpp.db.models import UserModel

    cid, _headers = await create_customer(client, auth_headers)
    user = await db_session.get(UserModel, cid)
    user.is_active = False
    await db_session.commit()
    resp = await client.post(
        "/api/v1/auth/token", params={"username": user.username, "password": "password123"}
    )
    assert resp.status_code == 401
