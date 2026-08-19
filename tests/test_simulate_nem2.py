"""NEM 2.0 TOU-period-aware bill simulation.

Before this, NEM2 export credit was a single blended retail-rate proxy
(total energy $ / total energy kWh) applied uniformly to all exported kWh
-- mis-crediting any customer whose TOU export mix differs from their
import mix. It also ignored URDB's ``energyratestructure[..].sell`` field
entirely, so a tariff that explicitly defines a different export rate per
TOU period had no way to express that.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from httpx import AsyncClient


def _two_period_urdb_with_sell() -> dict:
    """Off-peak (all hours except 16-20): rate=0.10, sell=0.08.
    Peak (hours 16-20): rate=0.30, sell=0.25.
    """
    weekday_row = [1 if 16 <= h < 20 else 0 for h in range(24)]
    return {
        "name": "Test TOU with explicit sell",
        "utility": "TestCo",
        "energyratestructure": [
            [{"rate": 0.10, "sell": 0.08}],
            [{"rate": 0.30, "sell": 0.25}],
        ],
        "energyweekdayschedule": [weekday_row] * 12,
        "energyweekendschedule": [weekday_row] * 12,
    }


def _tiered_only_urdb() -> dict:
    """A single period with two usage tiers -> TieredEnergyRate only, no
    TimeOfUseRate component. Used to exercise the fallback branch."""
    return {
        "name": "Test tiered-only",
        "utility": "TestCo",
        "energyratestructure": [
            [{"rate": 0.15, "max": 500}, {"rate": 0.25}],
        ],
        "energyweekdayschedule": [[0] * 24] * 12,
        "energyweekendschedule": [[0] * 24] * 12,
    }


@pytest.mark.asyncio
async def test_simulate_nem2_credits_per_tou_period_using_urdb_sell(
    client: AsyncClient, auth_headers: dict
):
    """Export credit must use each interval's own TOU period sell rate,
    not a single bill-wide blended average."""
    n = 24
    start = datetime(2024, 7, 1, tzinfo=timezone.utc)  # a Monday
    timestamps = [(start + timedelta(hours=i)).isoformat() for i in range(n)]
    import_kwh = [1.0] * n
    export_kwh = [0.0] * n
    export_kwh[10] = 10.0  # off-peak hour -> sell @ 0.08
    export_kwh[17] = 10.0  # peak hour (16-20) -> sell @ 0.25

    body = {
        "urdb_json": _two_period_urdb_with_sell(),
        "meter_trace": {
            "timestamps": timestamps,
            "import_kwh": import_kwh,
            "export_kwh": export_kwh,
            "interval_minutes": 60,
        },
        "billing_period_start": start.isoformat(),
        "billing_period_end": (start + timedelta(hours=24)).isoformat(),
        "nem": "nem2",
    }
    resp = await client.post("/api/v1/tariffs/simulate", json=body, headers=auth_headers)
    assert resp.status_code == 200, resp.text
    payload = resp.json()

    credits = [li for li in payload["line_items"] if li["kind"] == "credit"]
    assert len(credits) == 1
    # 10 kWh @ 0.08 (off-peak sell) + 10 kWh @ 0.25 (peak sell) = 3.30
    expected_credit = 10.0 * 0.08 + 10.0 * 0.25
    assert credits[0]["amount"] == pytest.approx(-expected_credit, abs=1e-3)


@pytest.mark.asyncio
async def test_simulate_nem2_defaults_to_import_rate_when_sell_absent(
    client: AsyncClient, auth_headers: dict
):
    """Without an explicit URDB `sell`, NEM2 still credits per-period at
    the *import* rate for that period -- not a blended bill-wide average.
    """
    weekday_row = [1 if 16 <= h < 20 else 0 for h in range(24)]
    urdb_json = {
        "name": "Test TOU without sell",
        "utility": "TestCo",
        "energyratestructure": [
            [{"rate": 0.10}],
            [{"rate": 0.30}],
        ],
        "energyweekdayschedule": [weekday_row] * 12,
        "energyweekendschedule": [weekday_row] * 12,
    }

    n = 24
    start = datetime(2024, 7, 1, tzinfo=timezone.utc)
    timestamps = [(start + timedelta(hours=i)).isoformat() for i in range(n)]
    import_kwh = [1.0] * n
    export_kwh = [0.0] * n
    export_kwh[10] = 10.0  # off-peak -> credited @ 0.10
    export_kwh[17] = 10.0  # peak -> credited @ 0.30

    body = {
        "urdb_json": urdb_json,
        "meter_trace": {
            "timestamps": timestamps,
            "import_kwh": import_kwh,
            "export_kwh": export_kwh,
            "interval_minutes": 60,
        },
        "billing_period_start": start.isoformat(),
        "billing_period_end": (start + timedelta(hours=24)).isoformat(),
        "nem": "nem2",
    }
    resp = await client.post("/api/v1/tariffs/simulate", json=body, headers=auth_headers)
    assert resp.status_code == 200, resp.text
    payload = resp.json()

    credits = [li for li in payload["line_items"] if li["kind"] == "credit"]
    assert len(credits) == 1
    expected_credit = 10.0 * 0.10 + 10.0 * 0.30
    assert credits[0]["amount"] == pytest.approx(-expected_credit, abs=1e-3)


@pytest.mark.asyncio
async def test_simulate_nem2_falls_back_to_blended_average_without_tou(
    client: AsyncClient, auth_headers: dict
):
    """A flat/tiered-only tariff has no TOU period to be accurate about --
    must still fall back to the blended retail-rate proxy rather than
    crediting $0 or erroring."""
    n = 20
    start = datetime(2024, 7, 1, tzinfo=timezone.utc)
    timestamps = [(start + timedelta(hours=i)).isoformat() for i in range(n)]
    import_kwh = [1.0] * n  # 20 kWh total, all within tier 1 (<=500)
    export_kwh = [0.0] * n
    export_kwh[5] = 5.0

    body = {
        "urdb_json": _tiered_only_urdb(),
        "meter_trace": {
            "timestamps": timestamps,
            "import_kwh": import_kwh,
            "export_kwh": export_kwh,
            "interval_minutes": 60,
        },
        "billing_period_start": start.isoformat(),
        "billing_period_end": (start + timedelta(hours=n)).isoformat(),
        "nem": "nem2",
    }
    resp = await client.post("/api/v1/tariffs/simulate", json=body, headers=auth_headers)
    assert resp.status_code == 200, resp.text
    payload = resp.json()

    credits = [li for li in payload["line_items"] if li["kind"] == "credit"]
    assert len(credits) == 1
    # All 20 kWh billed in tier 1 @ 0.15 -> blended avg rate 0.15.
    assert credits[0]["amount"] == pytest.approx(-(5.0 * 0.15), abs=1e-3)
