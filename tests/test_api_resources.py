"""Tests for the /api/v1/resources endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_create_battery(client: AsyncClient, auth_headers: dict):
    resp = await client.post(
        "/api/v1/resources/",
        json={
            "name": "test-battery-1",
            "resource_type": "battery",
            "rated_power": 50.0,
            "capacity_kwh": 100.0,
            "current_charge_kwh": 50.0,
            "nominal_voltage": 48.0,
        },
        headers=auth_headers,
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "test-battery-1"
    assert body["resource_type"] == "battery"
    assert body["rated_power"] == 50.0


@pytest.mark.asyncio
async def test_list_resources(client: AsyncClient, auth_headers: dict):
    resp = await client.get("/api/v1/resources/", headers=auth_headers)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_create_resource_requires_auth(client: AsyncClient):
    resp = await client.post(
        "/api/v1/resources/", json={"name": "x", "resource_type": "battery", "rated_power": 10}
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_duplicate_name_rejected(client: AsyncClient, auth_headers: dict):
    name = "unique-dup-test"
    payload = {
        "name": name,
        "resource_type": "solar",
        "rated_power": 10.0,
        "panel_area_m2": 20.0,
        "panel_efficiency": 0.2,
    }
    resp1 = await client.post("/api/v1/resources/", json=payload, headers=auth_headers)
    assert resp1.status_code == 201
    resp2 = await client.post("/api/v1/resources/", json=payload, headers=auth_headers)
    assert resp2.status_code == 409


@pytest.mark.asyncio
async def test_update_resource_returns_fresh_row(client: AsyncClient, auth_headers: dict):
    """PUT used to 500 (MissingGreenlet) lazy-loading the expired updated_at."""
    created = await client.post(
        "/api/v1/resources/",
        json={"name": "put-regression", "resource_type": "battery", "rated_power": 5.0},
        headers=auth_headers,
    )
    rid = created.json()["id"]
    resp = await client.put(
        f"/api/v1/resources/{rid}", json={"online": False}, headers=auth_headers
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["online"] is False
    assert resp.json()["updated_at"]


@pytest.mark.asyncio
async def test_viewer_cannot_mutate_resources(client: AsyncClient, auth_headers, viewer_headers):
    body = {"name": "viewer-denied-batt", "resource_type": "battery", "rated_power": 10.0}
    denied = await client.post("/api/v1/resources", json=body, headers=viewer_headers)
    assert denied.status_code == 403

    created = await client.post("/api/v1/resources", json=body, headers=auth_headers)
    assert created.status_code == 201
    rid = created.json()["id"]
    put = await client.put(
        f"/api/v1/resources/{rid}", json={"rated_power": 5.0}, headers=viewer_headers
    )
    assert put.status_code == 403
    delete = await client.delete(f"/api/v1/resources/{rid}", headers=viewer_headers)
    assert delete.status_code == 403
    # Viewers keep read access.
    assert (await client.get(f"/api/v1/resources/{rid}", headers=viewer_headers)).status_code == 200
