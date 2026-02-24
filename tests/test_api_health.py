"""Tests for health and version endpoints (no auth required)."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_readiness(client: AsyncClient):
    resp = await client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


@pytest.mark.asyncio
async def test_version(client: AsyncClient):
    resp = await client.get("/version")
    assert resp.status_code == 200
    body = resp.json()
    assert "version" in body
    assert body["api_version"] == "v1"
