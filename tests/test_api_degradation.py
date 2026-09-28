"""Tests for the /api/v1/batteries/.../soh endpoints (M3)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from httpx import AsyncClient

from vpp.auth.security import create_access_token, get_password_hash
from vpp.db.repositories import (
    BatteryDegradationRepository,
    ResourceRepository,
    UserRepository,
)


@pytest_asyncio.fixture
async def viewer_headers(db_session):
    """Auth headers for a non-admin user."""
    user = await UserRepository.get_by_username(db_session, "soh-viewer")
    if user is None:
        user = await UserRepository.create_user(
            db_session,
            username="soh-viewer",
            hashed_password=get_password_hash("viewerpassword"),
            role="viewer",
        )
        await db_session.commit()
    token = create_access_token({"sub": user.id, "username": user.username, "role": user.role})
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture
async def seeded_battery(db_session):
    """Battery with a known SOH/throughput for deterministic API tests."""
    battery = await ResourceRepository.create(
        db_session,
        name=f"api-soh-{datetime.now().timestamp()}",
        resource_type="battery",
        rated_power=100.0,
    )
    battery.chemistry = "lfp"
    battery.state_of_health = 0.97
    battery.cumulative_throughput_kwh = 250.0
    await db_session.commit()

    ts = datetime.now(timezone.utc)
    await BatteryDegradationRepository.update_battery_soh(
        db_session,
        battery.id,
        soh=0.97,
        cum_throughput_kwh=250.0,
        ts=ts,
    )
    await db_session.commit()
    return battery


@pytest.mark.asyncio
async def test_get_soh_endpoint(client: AsyncClient, auth_headers: dict, seeded_battery):
    resp = await client.get(f"/api/v1/batteries/{seeded_battery.id}/soh", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["battery_id"] == seeded_battery.id
    assert body["state_of_health"] == pytest.approx(0.97)
    assert body["cumulative_throughput_kwh"] == pytest.approx(250.0)
    assert body["chemistry"] == "lfp"
    assert "daily_efc" in body
    assert "projected_eol_date" in body


@pytest.mark.asyncio
async def test_post_soh_update_requires_admin(
    client: AsyncClient, viewer_headers: dict, seeded_battery
):
    base = datetime.now(timezone.utc)
    resp = await client.post(
        f"/api/v1/batteries/{seeded_battery.id}/soh/update",
        headers=viewer_headers,
        json={
            "soc_trace": [0.5, 0.6, 0.5],
            "timestamps": [
                base.isoformat(),
                (base + timedelta(minutes=15)).isoformat(),
                (base + timedelta(minutes=30)).isoformat(),
            ],
        },
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_projected_eol_date_calculation(client: AsyncClient, auth_headers: dict, db_session):
    """A battery with a known daily_efc should yield a deterministic EOL date.

    Formula: days_left = (SOH - 0.8) / (loss_per_efc * daily_efc),
    where loss_per_efc = (1 - 0.8) / cycles_to_eol_preset.
    For LFP preset: cycles_to_eol = 6000, loss_per_efc = 0.2 / 6000.
    """
    from vpp.degradation.models import LFP_PRESET

    capacity = 100.0
    soh = 0.95
    # Construct cumulative_throughput so daily_efc == 1.0:
    # daily_efc = (cum_throughput / (2*capacity)) / age_days = 1.0
    # We can't easily set created_at, so we instead just compute against
    # whatever age_days the row reports.
    battery = await ResourceRepository.create(
        db_session,
        name=f"eol-{datetime.now().timestamp()}",
        resource_type="battery",
        rated_power=capacity,
    )
    battery.chemistry = "lfp"
    battery.state_of_health = soh
    # Force a meaningful cumulative throughput.
    battery.cumulative_throughput_kwh = 600.0  # 3 EFC total
    battery.last_degradation_update = datetime.now(timezone.utc)
    await db_session.commit()

    resp = await client.get(f"/api/v1/batteries/{battery.id}/soh", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()

    daily_efc = body["daily_efc"]
    assert daily_efc > 0
    cycles_to_eol = LFP_PRESET["throughput"]["cycles_to_eol"]
    loss_per_efc = (1.0 - 0.8) / cycles_to_eol
    days_left = (soh - 0.8) / (loss_per_efc * daily_efc)

    assert body["projected_eol_date"] is not None
    projected = datetime.fromisoformat(body["projected_eol_date"].replace("Z", "+00:00"))
    expected = datetime.now(timezone.utc) + timedelta(days=days_left)
    delta_seconds = abs((projected - expected).total_seconds())
    # Allow up to a few seconds of clock drift between server-side now()
    # calls within the request.
    assert delta_seconds < 60
