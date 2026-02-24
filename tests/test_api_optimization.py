"""Tests for the /api/v1/optimization endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_dispatch(client: AsyncClient, auth_headers: dict):
    resp = await client.post(
        "/api/v1/optimization/dispatch",
        json={"target_power_kw": 50.0},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "success" in body
    assert "solve_time_ms" in body
    assert "allocations" in body


@pytest.mark.asyncio
async def test_stochastic(client: AsyncClient, auth_headers: dict):
    resp = await client.post(
        "/api/v1/optimization/stochastic",
        json={
            "num_scenarios": 10,
            "time_horizon_hours": 24,
            "volatility": 0.2,
        },
        headers=auth_headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "solve_time_ms" in body


@pytest.mark.asyncio
async def test_realtime(client: AsyncClient, auth_headers: dict):
    resp = await client.post(
        "/api/v1/optimization/realtime",
        json={
            "grid_frequency_hz": 49.95,
            "grid_voltage_pu": 1.0,
            "active_power_demand_kw": 100.0,
        },
        headers=auth_headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body


@pytest.mark.asyncio
async def test_optimization_history(client: AsyncClient, auth_headers: dict):
    resp = await client.get("/api/v1/optimization/history", headers=auth_headers)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_optimization_stats(client: AsyncClient, auth_headers: dict):
    resp = await client.get("/api/v1/optimization/stats", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert "total_runs" in body
