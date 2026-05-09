"""NEM3-aware bill simulation (M4)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from httpx import AsyncClient


PRESET = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "vpp"
    / "tariffs"
    / "presets"
    / "pge_etouc.json"
)


def _preset() -> dict:
    with open(PRESET) as f:
        return json.load(f)


@pytest.mark.asyncio
async def test_simulate_with_nem3_export_credit(
    client: AsyncClient, auth_headers: dict
):
    """A meter trace exporting at peak hours earns NEM3 export credit.

    Setup: 24 hours of 1 kWh/hr import + a 50 kWh export concentrated at hour
    18 (peak avoided-cost = $0.20/kWh -> $10 credit).
    """
    n = 24
    start = datetime(2024, 7, 1, tzinfo=timezone.utc)
    timestamps = [(start + timedelta(hours=i)).isoformat() for i in range(n)]
    import_kwh = [1.0] * n
    export_kwh = [0.0] * n
    export_kwh[18] = 50.0

    avoided_cost = [0.05] * 18 + [0.20] + [0.05] * 5  # 24-element vector

    body = {
        "urdb_json": _preset(),
        "meter_trace": {
            "timestamps": timestamps,
            "import_kwh": import_kwh,
            "export_kwh": export_kwh,
            "interval_minutes": 60,
        },
        "billing_period_start": start.isoformat(),
        "billing_period_end": (start + timedelta(hours=24)).isoformat(),
        "nem": "nem3",
        "nem3_avoided_cost": avoided_cost,
    }
    resp = await client.post("/api/v1/tariffs/simulate", json=body, headers=auth_headers)
    assert resp.status_code == 200, resp.text
    payload = resp.json()

    credits = [li for li in payload["line_items"] if li["kind"] == "credit"]
    assert len(credits) == 1
    assert credits[0]["amount"] == pytest.approx(-10.0, abs=1e-2)
    # Total includes the credit reduction.
    assert payload["total"] < sum(
        li["amount"] for li in payload["line_items"] if li["kind"] != "credit"
    )
