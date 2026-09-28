"""Tests for /api/v1/sites (map feed, ownership, membership, meter readings)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from _portal_helpers import (
    create_customer,
    create_resource,
    create_site,
    isolated_client,
    user_headers,
)

from vpp.db.models import AlertModel, BatteryStateModel, ResourceModel
from vpp.portal import sites as sites_service

if TYPE_CHECKING:
    from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def client(app):
    """Per-test client IP so these API-heavy tests don't drain the shared rate limit."""
    async with isolated_client(app) as c:
        yield c


async def _set_power(client, headers, rid, kw, soc=None):
    sample = {"power_kw": kw}
    if soc is not None:
        sample["state_of_charge"] = soc
    resp = await client.post(
        f"/api/v1/resources/{rid}/telemetry", json={"samples": [sample]}, headers=headers
    )
    assert resp.status_code == 202, resp.text


async def test_site_contract_matches_console(client: AsyncClient, auth_headers: dict, db_session):
    bat = await create_resource(client, auth_headers, rated_power=5.0)
    pv = await create_resource(client, auth_headers, resource_type="solar", rated_power=7.0)
    res = await db_session.get(ResourceModel, bat)
    res.nominal_energy_kwh = 13.5
    await db_session.commit()
    await _set_power(client, auth_headers, bat, -2.0)
    await _set_power(client, auth_headers, pv, 4.5)
    db_session.add(
        BatteryStateModel(
            resource_id=bat, soc=60.0, timestamp=datetime.now(timezone.utc) - timedelta(minutes=1)
        )
    )
    await db_session.commit()

    site = await create_site(
        client, auth_headers, region="ERCOT", resource_ids=[bat, pv], timezone="America/Chicago"
    )
    for key in (
        "id",
        "name",
        "lat",
        "lon",
        "region",
        "resource_ids",
        "total_resources",
        "online_count",
        "current_power",
        "rated_power",
        "active_alerts",
        "health",
    ):
        assert key in site, key
    assert sorted(site["resource_ids"]) == sorted([bat, pv])
    assert site["total_resources"] == 2
    assert site["online_count"] == 2
    assert site["rated_power"] == pytest.approx(12.0)
    assert site["current_power"] == pytest.approx(2.5)
    assert site["capacity_kwh"] == pytest.approx(13.5)
    assert site["state_of_charge"] == pytest.approx(0.6)
    assert site["health"] == "green"
    assert site["active_alerts"] == 0

    listing = await client.get("/api/v1/sites", headers=auth_headers)
    assert listing.status_code == 200
    assert site["id"] in {s["id"] for s in listing.json()}
    # Trailing-slash form works too (the console calls it without).
    assert (await client.get("/api/v1/sites/", headers=auth_headers)).status_code == 200
    filtered = await client.get("/api/v1/sites", params={"region": "ERCOT"}, headers=auth_headers)
    assert all(s["region"] == "ERCOT" for s in filtered.json())


async def test_latest_soc_prefers_newest_source(
    client: AsyncClient, auth_headers: dict, db_session
):
    bat = await create_resource(client, auth_headers)
    now = datetime.now(timezone.utc)
    db_session.add(
        BatteryStateModel(resource_id=bat, soc=90.0, timestamp=now - timedelta(hours=1))
    )
    await db_session.commit()
    await _set_power(client, auth_headers, bat, 1.0, soc=0.25)  # newer, generic telemetry
    site = await create_site(client, auth_headers, resource_ids=[bat])
    assert site["state_of_charge"] == pytest.approx(0.25)


async def test_health_reflects_offline_resources(client: AsyncClient, auth_headers: dict):
    rids = [await create_resource(client, auth_headers) for _ in range(3)]
    site = await create_site(client, auth_headers, resource_ids=rids)
    assert site["health"] == "green"
    await client.put(f"/api/v1/resources/{rids[0]}", json={"online": False}, headers=auth_headers)
    one = (await client.get(f"/api/v1/sites/{site['id']}", headers=auth_headers)).json()
    assert (one["online_count"], one["health"]) == (2, "yellow")
    await client.put(f"/api/v1/resources/{rids[1]}", json={"online": False}, headers=auth_headers)
    two = (await client.get(f"/api/v1/sites/{site['id']}", headers=auth_headers)).json()
    assert two["health"] == "red"


async def test_active_alerts_count_persisted_alerts(
    client: AsyncClient, auth_headers: dict, db_session
):
    """The app wires the AlertService's store into the site summary."""
    a = await create_resource(client, auth_headers)
    b = await create_resource(client, auth_headers, resource_type="solar")
    outsider = await create_resource(client, auth_headers)
    site = await create_site(client, auth_headers, resource_ids=[a, b])
    assert site["active_alerts"] == 0

    now = datetime.now(timezone.utc)

    def alert(source: str, status: str, snoozed_until=None) -> AlertModel:
        return AlertModel(
            rule_name="t",
            title="t",
            source=source,
            source_kind="resource",
            status=status,
            snoozed_until=snoozed_until,
            fired_at=now,
            last_fired_at=now,
        )

    db_session.add_all(
        [
            alert(a, "active"),
            alert(a, "acknowledged"),
            alert(b, "snoozed", now - timedelta(minutes=5)),  # expired -> counts
            alert(b, "snoozed", now + timedelta(hours=1)),  # muted -> not counted
            alert(b, "resolved"),
            alert(outsider, "active"),  # not a member of the site
        ]
    )
    await db_session.commit()

    got = (await client.get(f"/api/v1/sites/{site['id']}", headers=auth_headers)).json()
    assert got["active_alerts"] == 3
    assert got["health"] == "red"  # >= 3 active alerts
    listed = (await client.get("/api/v1/sites", headers=auth_headers)).json()
    assert next(s for s in listed if s["id"] == site["id"])["active_alerts"] == 3

    # Resolving via the API drops the count.
    open_ids = [
        r["id"]
        for r in (
            await client.get(f"/api/v1/alerts?source={a}&status=open", headers=auth_headers)
        ).json()
    ]
    for alert_id in open_ids:
        resp = await client.post(f"/api/v1/alerts/{alert_id}/resolve", headers=auth_headers)
        assert resp.status_code == 200
    got = (await client.get(f"/api/v1/sites/{site['id']}", headers=auth_headers)).json()
    assert got["active_alerts"] == 1
    assert got["health"] == "yellow"


async def test_alert_count_provider_feeds_site(client: AsyncClient, auth_headers: dict):
    rid = await create_resource(client, auth_headers)
    site = await create_site(client, auth_headers, resource_ids=[rid])

    async def provider(_session, ids):
        return {i: 2 for i in ids if i == rid}

    previous = sites_service.register_alert_count_provider(provider)
    try:
        got = (await client.get(f"/api/v1/sites/{site['id']}", headers=auth_headers)).json()
    finally:
        sites_service.register_alert_count_provider(previous)
    assert got["active_alerts"] == 2
    assert got["health"] == "yellow"


async def test_customer_sees_only_own_sites(client: AsyncClient, auth_headers: dict):
    cid, cheaders = await create_customer(client, auth_headers)
    _other, _oheaders = await create_customer(client, auth_headers)
    mine = await create_site(client, auth_headers, owner_id=cid)
    theirs = await create_site(client, auth_headers, owner_id=_other)
    unowned = await create_site(client, auth_headers)

    listing = await client.get("/api/v1/sites", headers=cheaders)
    assert listing.status_code == 200
    assert [s["id"] for s in listing.json()] == [mine["id"]]
    assert (await client.get(f"/api/v1/sites/{mine['id']}", headers=cheaders)).status_code == 200
    for other in (theirs, unowned):
        assert (
            await client.get(f"/api/v1/sites/{other['id']}", headers=cheaders)
        ).status_code == 404


async def test_site_write_permissions(client: AsyncClient, auth_headers: dict, db_session):
    viewer = await user_headers(db_session, "viewer")
    operator = await user_headers(db_session, "operator")
    _cid, customer = await create_customer(client, auth_headers)
    body = {"name": "perm-test", "lat": 1.0, "lon": 2.0}
    assert (await client.post("/api/v1/sites", json=body, headers=viewer)).status_code == 403
    assert (await client.post("/api/v1/sites", json=body, headers=customer)).status_code == 403
    created = await client.post("/api/v1/sites", json=body, headers=operator)
    assert created.status_code == 201
    sid = created.json()["id"]
    # Only admins delete.
    assert (await client.delete(f"/api/v1/sites/{sid}", headers=operator)).status_code == 403
    assert (await client.delete(f"/api/v1/sites/{sid}", headers=auth_headers)).status_code == 204
    assert (await client.get(f"/api/v1/sites/{sid}", headers=auth_headers)).status_code == 404
    assert (await client.get("/api/v1/sites")).status_code == 401


async def test_site_validation(client: AsyncClient, auth_headers: dict, db_session):

    viewer = await user_headers(db_session, "viewer")
    viewer_id = (await client.get("/api/v1/auth/me", headers=viewer)).json()["id"]
    bad = [
        {"name": "x", "lat": 91, "lon": 0},
        {"name": "x", "lat": 0, "lon": 181},
        {"name": "x", "lat": 0, "lon": 0, "timezone": "Mars/Olympus"},
        {"name": "x", "lat": 0, "lon": 0, "owner_id": viewer_id},  # owner must be a customer
        {"name": "x", "lat": 0, "lon": 0, "owner_id": "nope"},
        {"name": "x", "lat": 0, "lon": 0, "resource_ids": ["does-not-exist"]},
        {"name": "x", "lat": 0, "lon": 0, "unexpected": 1},
    ]
    for body in bad:
        resp = await client.post("/api/v1/sites", json=body, headers=auth_headers)
        assert resp.status_code == 422, (body, resp.text)


async def test_membership_is_exclusive_and_replaceable(client: AsyncClient, auth_headers: dict):
    a = await create_resource(client, auth_headers)
    b = await create_resource(client, auth_headers)
    s1 = await create_site(client, auth_headers, resource_ids=[a])
    conflict = await client.post(
        "/api/v1/sites",
        json={"name": "s2", "lat": 0, "lon": 0, "resource_ids": [a]},
        headers=auth_headers,
    )
    assert conflict.status_code == 409

    patched = await client.patch(
        f"/api/v1/sites/{s1['id']}",
        json={"resource_ids": [b], "name": "renamed"},
        headers=auth_headers,
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["resource_ids"] == [b]
    assert patched.json()["name"] == "renamed"
    # `a` is free again and can join another site.
    s2 = await create_site(client, auth_headers, resource_ids=[a])
    assert s2["resource_ids"] == [a]

    # Deleting a site unassigns (not deletes) its resources.
    assert (
        await client.delete(f"/api/v1/sites/{s1['id']}", headers=auth_headers)
    ).status_code == 204
    assert (await client.get(f"/api/v1/resources/{b}", headers=auth_headers)).status_code == 200
    s3 = await create_site(client, auth_headers, resource_ids=[b])
    assert s3["resource_ids"] == [b]


async def test_owner_assignment_and_unassignment(client: AsyncClient, auth_headers: dict):
    cid, cheaders = await create_customer(client, auth_headers)
    site = await create_site(client, auth_headers)
    resp = await client.patch(
        f"/api/v1/sites/{site['id']}", json={"owner_id": cid}, headers=auth_headers
    )
    assert resp.json()["owner_id"] == cid
    assert (await client.get(f"/api/v1/sites/{site['id']}", headers=cheaders)).status_code == 200
    resp = await client.patch(
        f"/api/v1/sites/{site['id']}", json={"owner_id": None}, headers=auth_headers
    )
    assert resp.json()["owner_id"] is None
    assert (await client.get(f"/api/v1/sites/{site['id']}", headers=cheaders)).status_code == 404
    bad = await client.patch(
        f"/api/v1/sites/{site['id']}", json={"lat": None}, headers=auth_headers
    )
    assert bad.status_code == 422
    assert (
        await client.patch("/api/v1/sites/nope", json={}, headers=auth_headers)
    ).status_code == 404


async def test_meter_readings_ingest_and_read(client: AsyncClient, auth_headers: dict):
    cid, cheaders = await create_customer(client, auth_headers)
    site = await create_site(client, auth_headers, owner_id=cid)
    url = f"/api/v1/sites/{site['id']}/meter-readings"
    t0 = datetime(2026, 3, 1, tzinfo=timezone.utc)
    readings = [
        {
            "timestamp": (t0 + timedelta(minutes=15 * i)).isoformat(),
            "import_kwh": 0.5,
            "export_kwh": 0.1,
        }
        for i in range(4)
    ]
    resp = await client.post(
        url, json={"interval_minutes": 15, "readings": readings}, headers=auth_headers
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"site_id": site["id"], "received": 4, "inserted": 4, "updated": 0}

    # Re-sending an interval corrects it (upsert).
    fix = [{"timestamp": t0.isoformat(), "import_kwh": 0.9}]
    resp = await client.post(
        url, json={"interval_minutes": 15, "readings": fix}, headers=auth_headers
    )
    assert resp.json()["updated"] == 1 and resp.json()["inserted"] == 0

    got = await client.get(
        url,
        params={"start": t0.isoformat(), "end": (t0 + timedelta(hours=1)).isoformat()},
        headers=cheaders,
    )
    assert got.status_code == 200
    rows = got.json()
    assert len(rows) == 4
    assert rows[0]["import_kwh"] == pytest.approx(0.9)
    assert rows[0]["export_kwh"] == pytest.approx(0.0)

    # Misaligned timestamp, mixed interval, negative energy, customer write.
    misaligned = [{"timestamp": (t0 + timedelta(minutes=7)).isoformat(), "import_kwh": 1}]
    assert (
        await client.post(
            url, json={"interval_minutes": 15, "readings": misaligned}, headers=auth_headers
        )
    ).status_code == 422
    hourly = [{"timestamp": t0.isoformat(), "import_kwh": 1}]
    assert (
        await client.post(
            url, json={"interval_minutes": 60, "readings": hourly}, headers=auth_headers
        )
    ).status_code == 409
    neg = [{"timestamp": t0.isoformat(), "import_kwh": -1}]
    assert (
        await client.post(
            url, json={"interval_minutes": 15, "readings": neg}, headers=auth_headers
        )
    ).status_code == 422
    assert (
        await client.post(url, json={"interval_minutes": 15, "readings": fix}, headers=cheaders)
    ).status_code == 403


async def test_customer_cannot_read_foreign_meter_data(client: AsyncClient, auth_headers: dict):
    _cid, cheaders = await create_customer(client, auth_headers)
    other = await create_site(client, auth_headers)
    resp = await client.get(f"/api/v1/sites/{other['id']}/meter-readings", headers=cheaders)
    assert resp.status_code == 404
