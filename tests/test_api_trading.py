"""Tests for the /api/v1/trading endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_submit_order(client: AsyncClient, auth_headers: dict):
    resp = await client.post(
        "/api/v1/trading/orders",
        json={
            "order_type": "limit",
            "market": "day_ahead",
            "side": "buy",
            "quantity": 100.0,
            "price": 0.12,
        },
        headers=auth_headers,
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["order_type"] == "limit"
    assert body["market"] == "day_ahead"
    assert body["status"] == "pending"


@pytest.mark.asyncio
async def test_list_orders(client: AsyncClient, auth_headers: dict):
    resp = await client.get("/api/v1/trading/orders", headers=auth_headers)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_cancel_order(client: AsyncClient, auth_headers: dict):
    # Create an order first
    create_resp = await client.post(
        "/api/v1/trading/orders",
        json={"order_type": "market", "market": "real_time", "side": "sell", "quantity": 50.0},
        headers=auth_headers,
    )
    order_id = create_resp.json()["id"]

    # Cancel it
    resp = await client.delete(f"/api/v1/trading/orders/{order_id}", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"


@pytest.mark.asyncio
async def test_list_trades(client: AsyncClient, auth_headers: dict):
    resp = await client.get("/api/v1/trading/trades", headers=auth_headers)
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_portfolio(client: AsyncClient, auth_headers: dict):
    resp = await client.get("/api/v1/trading/portfolio", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert "cash" in body
    assert "equity" in body
