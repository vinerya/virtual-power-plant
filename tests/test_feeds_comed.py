"""Tests for the ComEd Hourly Pricing feed (M4)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest

from vpp.tariffs.feeds import ComEdHourlyFeed


@pytest.mark.asyncio
async def test_comed_5min_parses():
    """Standard 5-minute response shape parses to PricePoints."""
    base = datetime(2024, 7, 1, tzinfo=timezone.utc)
    payload = []
    for k in range(3):
        ts_ms = int((base + timedelta(minutes=5 * k)).timestamp() * 1000)
        payload.append({"millisUTC": str(ts_ms), "price": str(2.4 + k * 0.1)})

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["type"] == "5minutefeed"
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    feed = ComEdHourlyFeed(client=client)
    points = await feed.fetch(base, base + timedelta(minutes=15))
    await client.aclose()

    assert len(points) == 3
    # 2.4 cents/kWh -> 0.024 $/kWh
    assert points[0].price_per_kwh == pytest.approx(0.024)
    assert points[2].price_per_kwh == pytest.approx(0.026)


@pytest.mark.asyncio
async def test_comed_handles_missing_data():
    """An empty array shouldn't blow up — return zero points."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    feed = ComEdHourlyFeed(client=client)
    points = await feed.fetch(
        datetime(2024, 7, 1, tzinfo=timezone.utc),
        datetime(2024, 7, 1, 1, tzinfo=timezone.utc),
    )
    await client.aclose()
    assert points == []
