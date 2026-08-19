"""Rate limiting is registered on the FastAPI app and actually enforced."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from vpp.api.app import create_app


@pytest.mark.asyncio
async def test_excess_requests_are_rate_limited():
    """A client that exceeds the configured requests-per-minute gets a 429."""
    app = create_app(rate_limit_enabled=True, rate_limit_requests_per_minute=3)
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        statuses = [(await client.get("/health")).status_code for _ in range(5)]

    assert statuses[:3] == [200, 200, 200]
    assert 429 in statuses[3:]


@pytest.mark.asyncio
async def test_rate_limiting_can_be_disabled():
    """With rate limiting off, no amount of requests gets throttled."""
    app = create_app(rate_limit_enabled=False, rate_limit_requests_per_minute=1)
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        statuses = [(await client.get("/health")).status_code for _ in range(5)]

    assert statuses == [200, 200, 200, 200, 200]
