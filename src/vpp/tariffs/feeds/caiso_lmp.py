"""CAISO OASIS Locational Marginal Price feed.

Fetches DAM (day-ahead) or RTM (real-time) LMPs from the OASIS public
SingleZip endpoint:

    https://oasis.caiso.com/oasisapi/SingleZip

Query params (CAISO OASIS public docs)
--------------------------------------
* ``resultformat=6``                          — CSV inside a zip.
* ``queryname=PRC_LMP``                       — LMP product.
* ``version=1``                               — API version.
* ``market_run_id=DAM`` or ``RTM``            — DAM = hourly, RTM = 5-min.
* ``startdatetime=YYYYMMDDTHH:MM-0000``       — UTC.
* ``enddatetime=YYYYMMDDTHH:MM-0000``         — UTC.
* ``node=<pnode>``                            — e.g. ``TH_NP15_GEN-APND``.

Returned LMPs are in $/MWh; we divide by 1000 to get $/kWh.

This adapter does live HTTP via httpx with exponential-backoff retries on
HTTP 429. Tests must mock the transport — see :mod:`tests.test_feeds_caiso`.
"""

from __future__ import annotations

import asyncio
import csv
import io
import zipfile
from datetime import datetime, timezone
from typing import Any

import httpx

from .base import PriceFeed, PricePoint
from .cache import FeedCache

CAISO_URL = "https://oasis.caiso.com/oasisapi/SingleZip"


class CAISOLMPFeed(PriceFeed):
    """CAISO OASIS LMP feed (DAM or RTM)."""

    name = "caiso_lmp"
    timezone = "UTC"

    def __init__(
        self,
        node: str = "TH_NP15_GEN-APND",
        market_run_id: str = "DAM",
        client: httpx.AsyncClient | None = None,
        cache: FeedCache | None = None,
        max_retries: int = 3,
        backoff_seconds: float = 0.05,
    ) -> None:
        if market_run_id not in {"DAM", "RTM"}:
            raise ValueError("market_run_id must be 'DAM' or 'RTM'")
        self.node = node
        self.market_run_id = market_run_id
        self._client = client
        self._cache = cache
        self.max_retries = int(max_retries)
        self.backoff_seconds = float(backoff_seconds)

    @staticmethod
    def _fmt_oasis(dt: datetime) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H:%M-0000")

    async def fetch(self, start: datetime, end: datetime, **kwargs: Any) -> list[PricePoint]:
        if self._cache is not None:
            cached = self._cache.get(self.name + ":" + self.market_run_id, start, end)
            if cached is not None:
                return cached

        params = {
            "resultformat": "6",
            "queryname": "PRC_LMP",
            "version": "1",
            "market_run_id": self.market_run_id,
            "startdatetime": self._fmt_oasis(start),
            "enddatetime": self._fmt_oasis(end),
            "node": self.node,
        }
        body = await self._fetch_with_retry(params)
        points = self._parse_zip(body)
        if self._cache is not None:
            self._cache.set(self.name + ":" + self.market_run_id, start, end, points)
        return points

    async def _fetch_with_retry(self, params: dict[str, str]) -> bytes:
        client = self._client
        owns_client = False
        if client is None:
            client = httpx.AsyncClient(timeout=30.0)
            owns_client = True
        try:
            attempt = 0
            while True:
                resp = await client.get(CAISO_URL, params=params)
                if resp.status_code == 200:
                    return resp.content
                if resp.status_code == 429 and attempt < self.max_retries:
                    delay = self.backoff_seconds * (2**attempt)
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                resp.raise_for_status()
                return resp.content
        finally:
            if owns_client:
                await client.aclose()

    def _parse_zip(self, body: bytes) -> list[PricePoint]:
        out: list[PricePoint] = []
        with zipfile.ZipFile(io.BytesIO(body)) as zf:
            for name in zf.namelist():
                if not name.lower().endswith(".csv"):
                    continue
                with zf.open(name) as fh:
                    text = fh.read().decode("utf-8", errors="replace")
                reader = csv.DictReader(io.StringIO(text))
                for row in reader:
                    # OASIS columns vary slightly; use INTERVALSTARTTIME_GMT + LMP_PRC.
                    ts_raw = (
                        row.get("INTERVALSTARTTIME_GMT")
                        or row.get("INTERVAL_START_GMT")
                        or row.get("OPR_DT")
                    )
                    price_raw = row.get("LMP_PRC") or row.get("MW") or row.get("VALUE")
                    if not ts_raw or price_raw is None:
                        continue
                    # OASIS timestamps look like "2024-07-01T00:00:00-00:00".
                    try:
                        ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                    except ValueError:
                        continue
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    try:
                        mwh_price = float(price_raw)
                    except (TypeError, ValueError):
                        continue
                    out.append(
                        PricePoint(
                            timestamp=ts.astimezone(timezone.utc),
                            price_per_kwh=mwh_price / 1000.0,
                            feed=self.name,
                            metadata={
                                "market_run_id": self.market_run_id,
                                "node": self.node,
                                "lmp_per_mwh": mwh_price,
                            },
                        )
                    )
        out.sort(key=lambda p: p.timestamp)
        return out
