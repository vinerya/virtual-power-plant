"""Tests for the CAISO OASIS LMP feed adapter (M4)."""

from __future__ import annotations

import io
import zipfile
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from vpp.tariffs.feeds import CAISOLMPFeed


def _zip_csv(rows: list[dict]) -> bytes:
    """Return a zipped CSV byte-string in OASIS-like shape."""
    buf = io.StringIO()
    cols = ["INTERVALSTARTTIME_GMT", "LMP_PRC", "NODE", "MARKET_RUN_ID"]
    buf.write(",".join(cols) + "\n")
    for r in rows:
        buf.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("PRC_LMP.csv", buf.getvalue())
    return out.getvalue()


@pytest.mark.asyncio
async def test_caiso_dam_parses():
    """Mock OASIS DAM zip: assert PricePoints parsed correctly."""
    rows = [
        {
            "INTERVALSTARTTIME_GMT": "2024-07-01T00:00:00-00:00",
            "LMP_PRC": "45.0",
            "NODE": "TH_NP15_GEN-APND",
            "MARKET_RUN_ID": "DAM",
        },
        {
            "INTERVALSTARTTIME_GMT": "2024-07-01T01:00:00-00:00",
            "LMP_PRC": "60.0",
            "NODE": "TH_NP15_GEN-APND",
            "MARKET_RUN_ID": "DAM",
        },
    ]
    body = _zip_csv(rows)

    def handler(request: httpx.Request) -> httpx.Response:
        assert "PRC_LMP" in request.url.params.get("queryname", "")
        return httpx.Response(200, content=body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    feed = CAISOLMPFeed(market_run_id="DAM", client=client)

    start = datetime(2024, 7, 1, tzinfo=timezone.utc)
    end = start + timedelta(hours=2)
    points = await feed.fetch(start, end)
    await client.aclose()

    assert len(points) == 2
    assert points[0].price_per_kwh == pytest.approx(0.045)  # 45 $/MWh -> $0.045/kWh
    assert points[1].price_per_kwh == pytest.approx(0.060)
    assert points[0].metadata["market_run_id"] == "DAM"


@pytest.mark.asyncio
async def test_caiso_rtm_5min_resolution():
    """RTM market returns 5-minute intervals; ensure we accept and order them."""
    base = datetime(2024, 7, 1, tzinfo=timezone.utc)
    rows = []
    for k in range(12):  # one hour of 5-min steps
        ts = base + timedelta(minutes=5 * k)
        rows.append(
            {
                "INTERVALSTARTTIME_GMT": ts.isoformat(),
                "LMP_PRC": str(50.0 + k),
                "NODE": "TH_NP15_GEN-APND",
                "MARKET_RUN_ID": "RTM",
            }
        )
    body = _zip_csv(rows)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["market_run_id"] == "RTM"
        return httpx.Response(200, content=body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    feed = CAISOLMPFeed(market_run_id="RTM", client=client)
    points = await feed.fetch(base, base + timedelta(hours=1))
    await client.aclose()

    assert len(points) == 12
    deltas = {
        (points[i + 1].timestamp - points[i].timestamp).total_seconds()
        for i in range(len(points) - 1)
    }
    assert deltas == {300.0}  # all 5-minute spacings


@pytest.mark.asyncio
async def test_caiso_retry_on_429():
    """A 429 response triggers retry+backoff; success on the second call."""
    rows = [
        {
            "INTERVALSTARTTIME_GMT": "2024-07-01T00:00:00-00:00",
            "LMP_PRC": "30.0",
            "NODE": "X",
            "MARKET_RUN_ID": "DAM",
        }
    ]
    body = _zip_csv(rows)
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if state["calls"] == 1:
            return httpx.Response(429, content=b"rate limited")
        return httpx.Response(200, content=body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    feed = CAISOLMPFeed(client=client, max_retries=2, backoff_seconds=0.001)
    points = await feed.fetch(
        datetime(2024, 7, 1, tzinfo=timezone.utc),
        datetime(2024, 7, 1, 1, tzinfo=timezone.utc),
    )
    await client.aclose()
    assert state["calls"] == 2
    assert len(points) == 1
    assert points[0].price_per_kwh == pytest.approx(0.030)
