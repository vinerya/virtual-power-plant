"""Tests for /api/v1/tariffs endpoints (Milestone 3)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from vpp.auth.security import create_access_token, get_password_hash
from vpp.db.repositories import UserRepository
from vpp.tariffs import BillingPeriod, MeterTrace, load_urdb_json


PRESET = Path(__file__).resolve().parents[1] / "src" / "vpp" / "tariffs" / "presets" / "pge_etouc.json"


def _load_preset_dict() -> dict:
    with open(PRESET) as f:
        return json.load(f)


@pytest_asyncio.fixture
async def viewer_headers(db_session: AsyncSession) -> dict[str, str]:
    """Auth headers for a non-admin (viewer) user."""
    user = await UserRepository.get_by_username(db_session, "testviewer")
    if user is None:
        user = await UserRepository.create_user(
            db_session,
            username="testviewer",
            hashed_password=get_password_hash("viewerpass123"),
            role="viewer",
        )
        await db_session.commit()
    token = create_access_token({
        "sub": user.id,
        "username": user.username,
        "role": user.role,
    })
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_create_and_get_tariff(client: AsyncClient, auth_headers: dict):
    body = {
        "name": "PG&E E-TOU-C (test create)",
        "utility": "PG&E",
        "urdb_json": _load_preset_dict(),
        "effective_date": "2024-03-01",
    }
    resp = await client.post("/api/v1/tariffs", json=body, headers=auth_headers)
    assert resp.status_code == 201, resp.text
    created = resp.json()
    assert created["name"] == body["name"]
    assert created["utility"] == "PG&E"
    tid = created["id"]

    resp2 = await client.get(f"/api/v1/tariffs/{tid}", headers=auth_headers)
    assert resp2.status_code == 200
    assert resp2.json()["id"] == tid


@pytest.mark.asyncio
async def test_list_filter_by_utility(client: AsyncClient, auth_headers: dict):
    # Create one PG&E + one SCE.
    await client.post(
        "/api/v1/tariffs",
        json={"name": "PG list test", "utility": "PG&E-LF", "urdb_json": _load_preset_dict()},
        headers=auth_headers,
    )
    await client.post(
        "/api/v1/tariffs",
        json={"name": "SCE list test", "utility": "SCE-LF", "urdb_json": _load_preset_dict()},
        headers=auth_headers,
    )
    resp = await client.get("/api/v1/tariffs?utility=PG%26E-LF", headers=auth_headers)
    assert resp.status_code == 200
    items = resp.json()
    assert all(it["utility"] == "PG&E-LF" for it in items)
    assert any(it["name"] == "PG list test" for it in items)


def _build_synthetic_trace_request(tariff_id: str | None = None, urdb_json: dict | None = None):
    start = datetime(2024, 7, 1, 0, 0, tzinfo=timezone.utc)
    n = 720  # 30 days * 24 hours
    timestamps = [(start + timedelta(hours=i)).isoformat() for i in range(n)]
    body = {
        "meter_trace": {
            "timestamps": timestamps,
            "import_kwh": [1.0] * n,  # 1 kWh/hr -> 720 kWh
            "interval_minutes": 60,
        },
        "billing_period_start": start.isoformat(),
        "billing_period_end": (start + timedelta(days=30)).isoformat(),
    }
    if tariff_id is not None:
        body["tariff_id"] = tariff_id
    if urdb_json is not None:
        body["urdb_json"] = urdb_json
    return body, start


@pytest.mark.asyncio
async def test_simulate_bill(client: AsyncClient, auth_headers: dict):
    preset = _load_preset_dict()
    create_resp = await client.post(
        "/api/v1/tariffs",
        json={"name": "PG sim", "utility": "PG&E-S", "urdb_json": preset},
        headers=auth_headers,
    )
    assert create_resp.status_code == 201
    tariff_id = create_resp.json()["id"]

    body, start = _build_synthetic_trace_request(tariff_id=tariff_id)

    resp = await client.post(
        f"/api/v1/tariffs/{tariff_id}/simulate", json=body, headers=auth_headers
    )
    assert resp.status_code == 200, resp.text
    api_total = resp.json()["total"]

    # Compare to direct Tariff.bill().
    tariff = load_urdb_json(preset)
    n = len(body["meter_trace"]["timestamps"])
    timestamps = [datetime.fromisoformat(ts) for ts in body["meter_trace"]["timestamps"]]
    trace = MeterTrace(timestamps=timestamps, import_kwh=[1.0] * n, interval_minutes=60)
    period = BillingPeriod(start=start, end=start + timedelta(days=30))
    direct = tariff.bill(trace, period)
    assert api_total == pytest.approx(direct.total, abs=1e-2)


@pytest.mark.asyncio
async def test_simulate_with_inline_urdb(client: AsyncClient, auth_headers: dict):
    preset = _load_preset_dict()
    body, start = _build_synthetic_trace_request(urdb_json=preset)
    resp = await client.post("/api/v1/tariffs/simulate", json=body, headers=auth_headers)
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["total"] > 0
    assert payload["line_items"]


@pytest.mark.asyncio
async def test_import_urdb_unauthenticated_403(client: AsyncClient, viewer_headers: dict):
    """A non-admin user must not import URDB tariffs."""
    resp = await client.post(
        "/api/v1/tariffs/import-urdb",
        json={"urdb_label": "anything"},
        headers=viewer_headers,
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_import_urdb_with_mocked_openei(client: AsyncClient, auth_headers: dict, monkeypatch):
    """Mock httpx call to simulate a URDB record import."""
    monkeypatch.setenv("OPENEI_API_KEY", "test-key")
    preset = _load_preset_dict()

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {"items": [preset]}

    class _FakeClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, params=None):
            return fake_response

    with patch("vpp.api.routes.tariffs.httpx.AsyncClient", _FakeClient):
        resp = await client.post(
            "/api/v1/tariffs/import-urdb",
            json={"urdb_label": "test-record-123"},
            headers=auth_headers,
        )

    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["urdb_label"] == "test-record-123"
    assert body["name"]
