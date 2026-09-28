"""Tests for the customer portal (/api/v1/customer/*) and its admin side."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import pytest
import pytest_asyncio
from _portal_helpers import (
    create_customer,
    create_resource,
    create_site,
    create_tariff,
    isolated_client,
    uid,
    user_headers,
)

if TYPE_CHECKING:
    from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def client(app):
    """Per-test client IP so these API-heavy tests don't drain the shared rate limit."""
    async with isolated_client(app) as c:
        yield c

CHI = ZoneInfo("America/Chicago")


async def _ingest(client, headers, site_id, start, hours, kwh_per_hour, export=0.0):
    readings = [
        {
            "timestamp": (start + timedelta(hours=i)).isoformat(),
            "import_kwh": kwh_per_hour,
            "export_kwh": export,
        }
        for i in range(hours)
    ]
    resp = await client.post(
        f"/api/v1/sites/{site_id}/meter-readings",
        json={"interval_minutes": 60, "readings": readings},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# Profile / devices
# ---------------------------------------------------------------------------


async def test_me_returns_profile(client: AsyncClient, auth_headers: dict):
    tid = await create_tariff(client, auth_headers)
    cid, headers = await create_customer(
        client, auth_headers, email="ada@example.com", address="1 Engine St",
        tariff_id=tid, baseline_kwh_per_month=850,
    )
    site = await create_site(client, auth_headers, owner_id=cid)
    resp = await client.get("/api/v1/customer/me", headers=headers)
    assert resp.status_code == 200, resp.text
    me = resp.json()
    assert me == {
        "id": cid, "username": me["username"], "name": "Ada Lovelace",
        "email": "ada@example.com", "address": "1 Engine St", "tariff_id": tid,
        "baseline_kwh_per_month": 850.0, "site_ids": [site["id"]], "is_active": True,
    }


async def test_devices_are_scoped_and_stateful(client: AsyncClient, auth_headers: dict):
    cid, headers = await create_customer(client, auth_headers)
    bat = await create_resource(client, auth_headers, rated_power=5)
    pv = await create_resource(client, auth_headers, resource_type="solar", rated_power=6)
    idle = await create_resource(client, auth_headers)
    foreign = await create_resource(client, auth_headers)
    await create_site(client, auth_headers, owner_id=cid, resource_ids=[bat, pv, idle])
    await create_site(client, auth_headers, resource_ids=[foreign])

    for rid, sample in ((bat, {"power_kw": -2.4, "state_of_charge": 0.62}), (pv, {"power_kw": 3.1})):
        r = await client.post(f"/api/v1/resources/{rid}/telemetry", json={"samples": [sample]},
                              headers=auth_headers)
        assert r.status_code == 202

    resp = await client.get("/api/v1/customer/me/devices", headers=headers)
    assert resp.status_code == 200, resp.text
    devices = {d["id"]: d for d in resp.json()}
    assert set(devices) == {bat, pv, idle}
    assert devices[bat]["kind"] == "battery"
    assert devices[bat]["state"] == "discharging"
    assert devices[bat]["current_power"] == pytest.approx(-2.4)
    assert devices[bat]["state_of_charge"] == pytest.approx(0.62)
    assert devices[pv]["state"] == "generating"
    assert devices[idle]["state"] == "idle"
    assert "state_of_charge" not in devices[idle]  # unknown, not faked

    await client.put(f"/api/v1/resources/{idle}", json={"online": False}, headers=auth_headers)
    devices = {d["id"]: d for d in (await client.get("/api/v1/customer/me/devices", headers=headers)).json()}
    assert devices[idle]["state"] == "offline"


async def test_customer_without_site_has_no_devices(client: AsyncClient, auth_headers: dict):
    _cid, headers = await create_customer(client, auth_headers)
    resp = await client.get("/api/v1/customer/me/devices", headers=headers)
    assert resp.status_code == 200
    assert resp.json() == []


# ---------------------------------------------------------------------------
# Bill
# ---------------------------------------------------------------------------


async def test_bill_from_meter_data_and_tariff(client: AsyncClient, auth_headers: dict):
    tid = await create_tariff(client, auth_headers)
    cid, headers = await create_customer(client, auth_headers, tariff_id=tid, baseline_kwh_per_month=744)
    site = await create_site(client, auth_headers, owner_id=cid, timezone="America/Chicago")
    july = datetime(2026, 7, 1, tzinfo=CHI)
    june = datetime(2026, 6, 1, tzinfo=CHI)
    await _ingest(client, auth_headers, site["id"], july, 48, 1.0, export=0.5)  # 48 kWh
    await _ingest(client, auth_headers, site["id"], june, 10, 2.0)              # 20 kWh
    # A reading just before local midnight July 1 belongs to June, not July.
    await _ingest(client, auth_headers, site["id"], july - timedelta(hours=1), 1, 100.0)

    resp = await client.get("/api/v1/customer/me/bill", params={"month": "2026-07"}, headers=headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    bill = body["bill"]
    assert bill["currency"] == "USD"
    assert bill["tariff_id"] == tid
    assert bill["total"] == pytest.approx(10.0 + 48 * 0.2)
    kinds = {li["kind"]: li for li in bill["line_items"]}
    assert kinds["fixed"]["amount"] == pytest.approx(10.0)
    assert kinds["energy"]["amount"] == pytest.approx(9.6)
    assert all({"kind", "name", "amount"} <= set(li) for li in bill["line_items"])
    assert datetime.fromisoformat(bill["period"]["start"]) == july
    assert datetime.fromisoformat(bill["period"]["end"]) == datetime(2026, 8, 1, tzinfo=CHI)
    meta = bill["metadata"]
    assert meta["meter_intervals"] == 48
    assert meta["expected_intervals"] == 744
    assert meta["data_coverage"] == pytest.approx(48 / 744, abs=1e-4)
    assert meta["export_kwh"] == pytest.approx(24.0)
    assert meta["export_credit_applied"] is False
    assert meta["timezone"] == "America/Chicago"

    assert body["this_month_kwh"] == pytest.approx(48.0)
    assert body["last_month_kwh"] == pytest.approx(120.0)
    # Baseline: 744 kWh declared, same tariff.
    assert body["baseline"]["total"] == pytest.approx(10.0 + 744 * 0.2)
    assert body["baseline"]["metadata"]["baseline_method"] == "declared_monthly_kwh_flat_profile"
    assert body["savings"] == pytest.approx(body["baseline"]["total"] - bill["total"])


async def test_bill_without_meter_data_is_honest(client: AsyncClient, auth_headers: dict):
    tid = await create_tariff(client, auth_headers)
    _cid, headers = await create_customer(client, auth_headers, tariff_id=tid)
    resp = await client.get("/api/v1/customer/me/bill", params={"month": "2026-02"}, headers=headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["bill"]["total"] == pytest.approx(10.0)  # fixed charge only
    assert body["bill"]["metadata"]["data_coverage"] == 0.0
    assert body["bill"]["metadata"]["timezone"] == "UTC"
    assert body["baseline"] is None
    assert body["savings"] is None
    assert body["this_month_kwh"] == 0


async def test_bill_errors(client: AsyncClient, auth_headers: dict):
    _cid, headers = await create_customer(client, auth_headers)
    resp = await client.get("/api/v1/customer/me/bill", headers=headers)
    assert resp.status_code == 409  # no tariff assigned: never a made-up bill
    tid = await create_tariff(client, auth_headers)
    _cid2, headers2 = await create_customer(client, auth_headers, tariff_id=tid)
    for bad in ("2026-13", "2026-7", "July"):
        r = await client.get("/api/v1/customer/me/bill", params={"month": bad}, headers=headers2)
        assert r.status_code == 422, bad
    # Default month works.
    assert (await client.get("/api/v1/customer/me/bill", headers=headers2)).status_code == 200


async def test_bill_sums_multiple_sites(client: AsyncClient, auth_headers: dict):
    tid = await create_tariff(client, auth_headers)
    cid, headers = await create_customer(client, auth_headers, tariff_id=tid)
    s1 = await create_site(client, auth_headers, owner_id=cid)
    s2 = await create_site(client, auth_headers, owner_id=cid)
    start = datetime(2026, 5, 1, tzinfo=timezone.utc)
    await _ingest(client, auth_headers, s1["id"], start, 5, 1.0)
    # Second site reports at 15-minute resolution -> resampled to hourly.
    readings = [
        {"timestamp": (start + timedelta(minutes=15 * i)).isoformat(), "import_kwh": 0.25}
        for i in range(8)
    ]
    r = await client.post(f"/api/v1/sites/{s2['id']}/meter-readings",
                          json={"interval_minutes": 15, "readings": readings}, headers=auth_headers)
    assert r.status_code == 200
    body = (await client.get("/api/v1/customer/me/bill", params={"month": "2026-05"},
                             headers=headers)).json()
    assert body["this_month_kwh"] == pytest.approx(7.0)
    assert body["bill"]["metadata"]["interval_minutes"] == 60
    assert body["bill"]["metadata"]["meter_intervals"] == 5


# ---------------------------------------------------------------------------
# Programs / enrollment
# ---------------------------------------------------------------------------


async def _program(client, admin, **fields):
    body = {"name": uid("program"), "description": "Shift load at peak", "utility": "Test",
            "incentive_per_event": 25, **fields}
    resp = await client.post("/api/v1/programs", json=body, headers=admin)
    assert resp.status_code == 201, resp.text
    return resp.json()


async def test_program_enrollment_flow(client: AsyncClient, auth_headers: dict):
    cid, headers = await create_customer(client, auth_headers)
    rid = await create_resource(client, auth_headers)
    await create_site(client, auth_headers, owner_id=cid, resource_ids=[rid])
    p1 = await _program(client, auth_headers)
    p2 = await _program(client, auth_headers)
    inactive = await _program(client, auth_headers, active=False)

    listing = await client.get("/api/v1/customer/programs", headers=headers)
    assert listing.status_code == 200
    by_id = {p["id"]: p for p in listing.json()}
    assert p1["id"] in by_id and inactive["id"] not in by_id
    assert by_id[p1["id"]]["enrolled"] is False
    assert by_id[p1["id"]]["incentive_per_event"] == 25
    assert {"id", "name", "description"} <= set(by_id[p1["id"]])

    url = "/api/v1/customer/enrollments"
    no_ack = await client.post(url, json={"program_ids": [p1["id"]], "acknowledged": False}, headers=headers)
    assert no_ack.status_code == 422
    unknown = await client.post(url, json={"program_ids": ["nope"], "acknowledged": True}, headers=headers)
    assert unknown.status_code == 422
    closed = await client.post(url, json={"program_ids": [inactive["id"]], "acknowledged": True},
                               headers=headers)
    assert closed.status_code == 422
    empty = await client.post(url, json={"program_ids": [], "acknowledged": True}, headers=headers)
    assert empty.status_code == 422

    ok = await client.post(url, json={"program_ids": [p1["id"]], "acknowledged": True}, headers=headers)
    assert ok.status_code == 201, ok.text
    assert ok.json() == {"ok": True, "enrolled": [p1["id"]], "device_ids": [rid]}
    again = await client.post(url, json={"program_ids": [p1["id"], p2["id"]], "acknowledged": True},
                              headers=headers)
    assert sorted(again.json()["enrolled"]) == sorted([p1["id"], p2["id"]])

    listing = {p["id"]: p for p in (await client.get("/api/v1/customer/programs", headers=headers)).json()}
    assert listing[p1["id"]]["enrolled"] is True

    admin_view = {p["id"]: p for p in (await client.get("/api/v1/programs", headers=auth_headers)).json()}
    assert admin_view[p1["id"]]["enrolled_count"] == 1
    assert admin_view[inactive["id"]]["active"] is False

    assert (await client.delete(f"{url}/{p1['id']}", headers=headers)).status_code == 204
    assert (await client.delete(f"{url}/{p1['id']}", headers=headers)).status_code == 404


async def test_enrollment_requires_devices(client: AsyncClient, auth_headers: dict):
    _cid, headers = await create_customer(client, auth_headers)
    p = await _program(client, auth_headers)
    resp = await client.post("/api/v1/customer/enrollments",
                             json={"program_ids": [p["id"]], "acknowledged": True}, headers=headers)
    assert resp.status_code == 409


async def test_program_admin_permissions_and_updates(client: AsyncClient, auth_headers: dict, db_session):
    operator = await user_headers(db_session, "operator")
    assert (await client.post("/api/v1/programs", json={"name": "x"}, headers=operator)).status_code == 403
    assert (await client.get("/api/v1/programs", headers=operator)).status_code == 200
    p = await _program(client, auth_headers)
    dup = await client.post("/api/v1/programs", json={"name": p["name"]}, headers=auth_headers)
    assert dup.status_code == 409
    upd = await client.patch(f"/api/v1/programs/{p['id']}", json={"active": False}, headers=auth_headers)
    assert upd.status_code == 200 and upd.json()["active"] is False
    assert (await client.patch(f"/api/v1/programs/{p['id']}", json={"name": None},
                               headers=auth_headers)).status_code == 422
    assert (await client.patch("/api/v1/programs/nope", json={}, headers=auth_headers)).status_code == 404


# ---------------------------------------------------------------------------
# Customer administration
# ---------------------------------------------------------------------------


async def test_customer_admin_endpoints(client: AsyncClient, auth_headers: dict, db_session):
    operator = await user_headers(db_session, "operator")
    tid = await create_tariff(client, auth_headers)
    cid, cheaders = await create_customer(client, auth_headers)

    # Operators can read, only admins can write.
    assert (await client.get(f"/api/v1/customers/{cid}", headers=operator)).status_code == 200
    assert cid in {c["id"] for c in (await client.get("/api/v1/customers", headers=operator)).json()}
    assert (await client.patch(f"/api/v1/customers/{cid}", json={"tariff_id": tid},
                               headers=operator)).status_code == 403
    bad = await client.patch(f"/api/v1/customers/{cid}", json={"tariff_id": "missing"}, headers=auth_headers)
    assert bad.status_code == 422
    upd = await client.patch(f"/api/v1/customers/{cid}", json={"tariff_id": tid, "name": "Grace"},
                             headers=auth_headers)
    assert upd.status_code == 200
    assert upd.json()["tariff_id"] == tid and upd.json()["name"] == "Grace"

    # Operators see the same bill/devices the customer sees.
    op_bill = await client.get(f"/api/v1/customers/{cid}/bill", params={"month": "2026-01"}, headers=operator)
    me_bill = await client.get("/api/v1/customer/me/bill", params={"month": "2026-01"}, headers=cheaders)
    assert op_bill.status_code == me_bill.status_code == 200
    assert op_bill.json() == me_bill.json()
    assert (await client.get(f"/api/v1/customers/{cid}/devices", headers=operator)).json() == []

    # Non-customer ids are not customers.
    admin_id = (await client.get("/api/v1/auth/me", headers=auth_headers)).json()["id"]
    assert (await client.get(f"/api/v1/customers/{admin_id}", headers=operator)).status_code == 404

    # Onboarding validation.
    dup = await client.post("/api/v1/customers", json={
        "username": (await client.get("/api/v1/customer/me", headers=cheaders)).json()["username"],
        "password": "password123", "name": "Dup"}, headers=auth_headers)
    assert dup.status_code == 409
    assert (await client.post("/api/v1/customers", json={
        "username": uid("c"), "password": "password123", "name": "X", "tariff_id": "missing"},
        headers=auth_headers)).status_code == 422
    assert (await client.post("/api/v1/customers", json={
        "username": uid("c"), "password": "password123", "name": "X"},
        headers=operator)).status_code == 403


async def test_register_customer_without_profile(client: AsyncClient, auth_headers: dict):
    """Customers created through /auth/register fall back to the username."""
    username = uid("reg")
    resp = await client.post("/api/v1/auth/register",
                             json={"username": username, "password": "password123", "role": "customer"},
                             headers=auth_headers)
    assert resp.status_code == 201
    login = await client.post("/api/v1/auth/token", params={"username": username, "password": "password123"})
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    me = (await client.get("/api/v1/customer/me", headers=headers)).json()
    assert me["name"] == username
    assert me["tariff_id"] is None
