"""Shared helpers for the sites / customer-portal / metrics / config API tests."""

from __future__ import annotations

import contextlib
import random
import uuid

from httpx import ASGITransport, AsyncClient

from vpp.auth.security import create_access_token, get_password_hash
from vpp.db.repositories import UserRepository

FLAT_TARIFF_URDB = {
    "name": "Flat 20c + $10/mo",
    "utility": "Test Utility",
    "energyratestructure": [[{"rate": 0.2}]],
    "energyweekdayschedule": [[0] * 24] * 12,
    "energyweekendschedule": [[0] * 24] * 12,
    "fixedchargefirstmeter": 10,
    "fixedchargeunits": "$/month",
}


def uid(prefix: str = "t") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


async def user_headers(db_session, role: str) -> dict[str, str]:
    """Create a fresh user with ``role`` and return bearer headers for it."""
    user = await UserRepository.create_user(
        db_session,
        username=uid(role),
        hashed_password=get_password_hash("Grid-Battery-4217"),
        role=role,
    )
    await db_session.commit()
    token = create_access_token({"sub": user.id, "username": user.username, "role": user.role})
    return {"Authorization": f"Bearer {token}"}


async def create_customer(
    client: AsyncClient, admin_headers: dict, **profile
) -> tuple[str, dict[str, str]]:
    """Onboard a customer via the admin API, log in as them; return (id, headers)."""
    username = uid("cust")
    body = {
        "username": username,
        "password": "Grid-Battery-4217",
        "name": "Ada Lovelace",
        **profile,
    }
    resp = await client.post("/api/v1/customers", json=body, headers=admin_headers)
    assert resp.status_code == 201, resp.text
    login = await client.post(
        "/api/v1/auth/token", data={"username": username, "password": "Grid-Battery-4217"}
    )
    assert login.status_code == 200, login.text
    return resp.json()["id"], {"Authorization": f"Bearer {login.json()['access_token']}"}


async def create_resource(
    client: AsyncClient,
    admin_headers: dict,
    *,
    resource_type: str = "battery",
    rated_power: float = 5.0,
) -> str:
    resp = await client.post(
        "/api/v1/resources/",
        json={
            "name": uid(resource_type),
            "resource_type": resource_type,
            "rated_power": rated_power,
        },
        headers=admin_headers,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


async def create_site(client: AsyncClient, admin_headers: dict, **fields) -> dict:
    body = {"name": uid("site"), "lat": 30.27, "lon": -97.74, **fields}
    resp = await client.post("/api/v1/sites", json=body, headers=admin_headers)
    assert resp.status_code == 201, resp.text
    return resp.json()


async def create_tariff(client: AsyncClient, admin_headers: dict) -> str:
    resp = await client.post(
        "/api/v1/tariffs",
        json={"name": uid("tariff"), "utility": "Test Utility", "urdb_json": FLAT_TARIFF_URDB},
        headers=admin_headers,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


@contextlib.asynccontextmanager
async def isolated_client(app):
    """HTTP client with its own random client IP.

    The session-scoped app keeps one rate-limit bucket per client IP; these
    API-heavy tests would otherwise drain the shared bucket and make
    unrelated later tests fail with 429.
    """
    ip = f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
    transport = ASGITransport(app=app, client=(ip, 50000))
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
