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
    resp = await client.post("/api/v1/resources/", json={"name": "x", "resource_type": "battery", "rated_power": 10})
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
