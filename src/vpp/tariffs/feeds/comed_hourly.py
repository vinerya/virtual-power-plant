"""ComEd Hourly Pricing API feed.

Endpoint: ``https://hourlypricing.comed.com/api`` returns 5-minute prices in
cents/kWh under the ``5minutefeed`` query.

Example query:
    GET https://hourlypricing.comed.com/api?type=5minutefeed&datestart=...&dateend=...

Response is a JSON array::

    [{"millisUTC": "1719797700000", "price": "2.4"}, ...]

Prices are in cents/kWh. We divide by 100 to normalize to $/kWh.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from .base import PriceFeed, PricePoint
from .cache import FeedCache


COMED_URL = "https://hourlypricing.comed.com/api"


class ComEdHourlyFeed(PriceFeed):
    """ComEd Hourly Pricing — 5-minute interval $/kWh feed."""

    name = "comed_hourly"
    timezone = "UTC"

    def __init__(
        self,
        feed_type: str = "5minutefeed",
        client: Optional[httpx.AsyncClient] = None,
        cache: Optional[FeedCache] = None,
        max_retries: int = 3,
        backoff_seconds: float = 0.05,
    ) -> None:
        self.feed_type = feed_type
        self._client = client
        self._cache = cache
        self.max_retries = int(max_retries)
        self.backoff_seconds = float(backoff_seconds)

    @staticmethod
    def _fmt(dt: datetime) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y%m%d%H%M")

    async def fetch(
        self, start: datetime, end: datetime, **kwargs: Any
    ) -> list[PricePoint]:
        if self._cache is not None:
            cached = self._cache.get(self.name, start, end)
            if cached is not None:
                return cached

        params = {
            "type": self.feed_type,
            "datestart": self._fmt(start),
            "dateend": self._fmt(end),
        }
        rows = await self._fetch_with_retry(params)
        points: list[PricePoint] = []
        for row in rows or []:
            try:
                ms = int(row["millisUTC"])
                cents = float(row["price"])
            except (KeyError, TypeError, ValueError):
                continue
            ts = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
            points.append(
                PricePoint(
                    timestamp=ts,
                    price_per_kwh=cents / 100.0,
                    feed=self.name,
                    metadata={"price_cents_per_kwh": cents, "feed_type": self.feed_type},
                )
            )
        points.sort(key=lambda p: p.timestamp)
        if self._cache is not None:
            self._cache.set(self.name, start, end, points)
        return points

    async def _fetch_with_retry(self, params: dict[str, str]) -> list[dict]:
        client = self._client
        owns_client = False
        if client is None:
            client = httpx.AsyncClient(timeout=30.0)
            owns_client = True
        try:
            attempt = 0
            while True:
                resp = await client.get(COMED_URL, params=params)
                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except ValueError:
                        return []
                if resp.status_code == 429 and attempt < self.max_retries:
                    delay = self.backoff_seconds * (2**attempt)
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                resp.raise_for_status()
                return []
        finally:
            if owns_client:
                await client.aclose()
