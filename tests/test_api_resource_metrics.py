"""Tests for GET /resources/{id}/metrics and POST /resources/{id}/telemetry."""

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

from vpp.db.models import BatteryStateModel
from vpp.portal.telemetry import choose_bucket_seconds

if TYPE_CHECKING:
    from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def client(app):
    """Per-test client IP so these API-heavy tests don't drain the shared rate limit."""
    async with isolated_client(app) as c:
        yield c


T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


async def _post_samples(client, headers, rid, samples):
    resp = await client.post(
        f"/api/v1/resources/{rid}/telemetry", json={"samples": samples}, headers=headers
    )
    assert resp.status_code == 202, resp.text
    return resp.json()


async def test_downsampled_history_custom_range(
    client: AsyncClient, auth_headers: dict, db_session
):
    rid = await create_resource(client, auth_headers)
    # Generic telemetry: 1 sample/minute for an hour, power = minute index.
    await _post_samples(
        client,
        auth_headers,
        rid,
        [
            {"timestamp": (T0 + timedelta(minutes=m)).isoformat(), "power_kw": float(m)}
            for m in range(60)
        ],
    )
    # MQTT-style battery states (SOC in percent) in the first bucket only.
    for m, soc in ((1, 40.0), (2, 60.0)):
        db_session.add(
            BatteryStateModel(
                resource_id=rid,
                soc=soc,
                power=0.0,
                timestamp=T0 + timedelta(minutes=m, seconds=30),
            )
        )
    await db_session.commit()

    resp = await client.get(
        f"/api/v1/resources/{rid}/metrics",
        params={
            "start": T0.isoformat(),
            "end": (T0 + timedelta(hours=1)).isoformat(),
            "bucket_seconds": 900,
        },
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["resource_id"] == rid
    assert body["window"] == "custom"
    assert body["bucket_seconds"] == 900
    pts = body["points"]
    assert len(pts) == 4
    assert [datetime.fromisoformat(p["timestamp"]) for p in pts] == [
        T0 + timedelta(minutes=15 * i) for i in range(4)
    ]
    # Bucket 0: 15 generic samples (0..14, mean 7) + 2 battery rows at 0 kW.
    assert pts[0]["power"] == pytest.approx((sum(range(15)) + 0 + 0) / 17)
    assert pts[0]["samples"] == 17
    assert pts[0]["state_of_charge"] == pytest.approx(0.5)  # percent normalised to fraction
    assert pts[1]["power"] == pytest.approx(sum(range(15, 30)) / 15)
    assert "state_of_charge" not in pts[1]


async def test_named_window_and_auto_bucket(client: AsyncClient, auth_headers: dict):
    rid = await create_resource(client, auth_headers)
    now = datetime.now(timezone.utc)
    await _post_samples(
        client,
        auth_headers,
        rid,
        [
            {
                "timestamp": (now - timedelta(minutes=m)).isoformat(),
                "power_kw": 1.0,
                "state_of_charge": 0.8,
            }
            for m in range(1, 30)
        ]
        + [{"timestamp": (now - timedelta(days=3)).isoformat(), "power_kw": 9.0}],
    )

    day = (
        await client.get(f"/api/v1/resources/{rid}/metrics?window=24h", headers=auth_headers)
    ).json()
    assert day["window"] == "24h"
    assert day["bucket_seconds"] == choose_bucket_seconds(timedelta(hours=24), 500) == 300
    assert day["points"]
    assert all(p["power"] == pytest.approx(1.0) for p in day["points"])
    assert all(p["state_of_charge"] == pytest.approx(0.8) for p in day["points"])

    week = (
        await client.get(
            f"/api/v1/resources/{rid}/metrics",
            params={"window": "7d", "max_points": 50},
            headers=auth_headers,
        )
    ).json()
    assert week["bucket_seconds"] == 21600
    assert any(p["power"] == pytest.approx(9.0) for p in week["points"])

    default = (await client.get(f"/api/v1/resources/{rid}/metrics", headers=auth_headers)).json()
    assert default["window"] == "24h"


async def test_metrics_validation(client: AsyncClient, auth_headers: dict):
    rid = await create_resource(client, auth_headers)
    url = f"/api/v1/resources/{rid}/metrics"
    for params in (
        {"window": "2y"},
        {"window": "1h", "start": T0.isoformat()},
        {"start": T0.isoformat(), "end": (T0 - timedelta(hours=1)).isoformat()},
        {"start": T0.isoformat(), "end": (T0 + timedelta(days=400)).isoformat()},
        {
            "start": T0.isoformat(),
            "end": (T0 + timedelta(days=30)).isoformat(),
            "bucket_seconds": 60,
        },
    ):
        resp = await client.get(url, params=params, headers=auth_headers)
        assert resp.status_code == 422, (params, resp.text)
    assert (
        await client.get("/api/v1/resources/nope/metrics", headers=auth_headers)
    ).status_code == 404
    assert (await client.get(url)).status_code == 401


async def test_metrics_rbac_for_customers(client: AsyncClient, auth_headers: dict, db_session):
    cid, cheaders = await create_customer(client, auth_headers)
    mine = await create_resource(client, auth_headers)
    theirs = await create_resource(client, auth_headers)
    unsited = await create_resource(client, auth_headers)
    await create_site(client, auth_headers, owner_id=cid, resource_ids=[mine])
    await create_site(client, auth_headers, resource_ids=[theirs])
    assert (
        await client.get(f"/api/v1/resources/{mine}/metrics", headers=cheaders)
    ).status_code == 200
    for rid in (theirs, unsited):
        assert (
            await client.get(f"/api/v1/resources/{rid}/metrics", headers=cheaders)
        ).status_code == 404
    viewer = await user_headers(db_session, "viewer")
    assert (
        await client.get(f"/api/v1/resources/{theirs}/metrics", headers=viewer)
    ).status_code == 200


async def test_telemetry_ingest_live_state_and_guards(
    client: AsyncClient, auth_headers: dict, db_session
):
    rid = await create_resource(client, auth_headers)
    now = datetime.now(timezone.utc)
    live = await _post_samples(
        client,
        auth_headers,
        rid,
        [
            {"timestamp": (now - timedelta(seconds=30)).isoformat(), "power_kw": 1.5},
            {"timestamp": (now - timedelta(seconds=5)).isoformat(), "power_kw": -3.0},
        ],
    )
    assert live == {"resource_id": rid, "accepted": 2, "current_power": -3.0}
    got = (await client.get(f"/api/v1/resources/{rid}", headers=auth_headers)).json()
    assert got["current_power"] == pytest.approx(-3.0)

    # Historical backfill is stored but leaves live state alone.
    backfill = await _post_samples(
        client,
        auth_headers,
        rid,
        [{"timestamp": (now - timedelta(days=2)).isoformat(), "power_kw": 42.0}],
    )
    assert backfill["current_power"] == pytest.approx(-3.0)

    url = f"/api/v1/resources/{rid}/telemetry"
    future = [{"timestamp": (now + timedelta(hours=1)).isoformat(), "power_kw": 1}]
    assert (
        await client.post(url, json={"samples": future}, headers=auth_headers)
    ).status_code == 422
    bad_soc = [{"power_kw": 1, "state_of_charge": 55}]
    assert (
        await client.post(url, json={"samples": bad_soc}, headers=auth_headers)
    ).status_code == 422
    assert (await client.post(url, json={"samples": []}, headers=auth_headers)).status_code == 422
    viewer = await user_headers(db_session, "viewer")
    ok = [{"power_kw": 1}]
    assert (await client.post(url, json={"samples": ok}, headers=viewer)).status_code == 403
    assert (
        await client.post(
            "/api/v1/resources/nope/telemetry", json={"samples": ok}, headers=auth_headers
        )
    ).status_code == 404


def test_choose_bucket_seconds_is_nice_and_bounded():
    assert choose_bucket_seconds(timedelta(hours=1), 500) == 10
    assert choose_bucket_seconds(timedelta(hours=1), 3600) == 1
    assert choose_bucket_seconds(timedelta(days=30), 500) == 7200
    assert choose_bucket_seconds(timedelta(days=366), 100) == 4 * 86400
    for span in (timedelta(minutes=5), timedelta(days=7), timedelta(days=200)):
        b = choose_bucket_seconds(span, 300)
        assert span.total_seconds() / b <= 300
