"""Tests for M3 battery degradation persistence and the telemetry updater."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from vpp.db.engine import get_session_factory
from vpp.db.repositories import (
    BatteryDegradationRepository,
    ResourceRepository,
)
from vpp.degradation.telemetry import DegradationUpdater, TelemetryWindow


@pytest.mark.asyncio
async def test_update_battery_soh_persists(db_session, app):
    """Writing SOH should round-trip through the DB."""
    battery = await ResourceRepository.create(
        db_session,
        name=f"persist-battery-{datetime.now().timestamp()}",
        resource_type="battery",
        rated_power=100.0,
    )
    await db_session.commit()

    ts = datetime.now(timezone.utc)
    await BatteryDegradationRepository.update_battery_soh(
        db_session, battery.id, soh=0.95, cum_throughput_kwh=42.5, ts=ts,
        loss_fraction=0.05,
    )
    await db_session.commit()

    fresh = await ResourceRepository.get_by_id(db_session, battery.id)
    assert fresh is not None
    assert fresh.state_of_health == pytest.approx(0.95)
    assert fresh.cumulative_throughput_kwh == pytest.approx(42.5)
    assert fresh.last_degradation_update is not None


@pytest.mark.asyncio
async def test_get_due_for_update_filters_by_staleness(db_session, app):
    """Only batteries with stale (or missing) updates should be returned."""
    suffix = datetime.now().timestamp()
    fresh_b = await ResourceRepository.create(
        db_session, name=f"fresh-{suffix}", resource_type="battery",
        rated_power=100.0,
    )
    stale_b = await ResourceRepository.create(
        db_session, name=f"stale-{suffix}", resource_type="battery",
        rated_power=100.0,
    )
    never_b = await ResourceRepository.create(
        db_session, name=f"never-{suffix}", resource_type="battery",
        rated_power=100.0,
    )
    await db_session.commit()

    now = datetime.now(timezone.utc)
    await BatteryDegradationRepository.update_battery_soh(
        db_session, fresh_b.id, soh=1.0, cum_throughput_kwh=0.0, ts=now,
        record_sample=False,
    )
    await BatteryDegradationRepository.update_battery_soh(
        db_session, stale_b.id, soh=1.0, cum_throughput_kwh=0.0,
        ts=now - timedelta(hours=5), record_sample=False,
    )
    await db_session.commit()

    due = await BatteryDegradationRepository.get_batteries_due_for_degradation_update(
        db_session, stale_after_minutes=60
    )
    due_ids = {b.id for b in due}
    assert never_b.id in due_ids
    assert stale_b.id in due_ids
    assert fresh_b.id not in due_ids


@pytest.mark.asyncio
async def test_apply_window_decreases_soh(db_session, app):
    """Applying a SOC window should reduce SOH by the predicted loss."""
    battery = await ResourceRepository.create(
        db_session,
        name=f"window-{datetime.now().timestamp()}",
        resource_type="battery",
        rated_power=100.0,
    )
    battery.chemistry = "lfp"
    battery.state_of_health = 1.0
    battery.cumulative_throughput_kwh = 0.0
    await db_session.commit()

    base = datetime.now(timezone.utc)
    soc_trace = [0.5, 0.9, 0.5, 0.9, 0.5]  # repeated 80%-DoD-ish cycles
    timestamps = [base + timedelta(minutes=15 * i) for i in range(len(soc_trace))]
    window = TelemetryWindow(
        battery_id=battery.id,
        soc_trace=soc_trace,
        timestamps=timestamps,
        temperatures_c=[25.0] * len(soc_trace),
    )

    updater = DegradationUpdater(session_factory=get_session_factory())
    update = await updater.apply_window(window)
    assert update is not None
    assert update.previous_soh == pytest.approx(1.0)
    assert update.new_soh < update.previous_soh
    assert update.loss_fraction > 0
    # Throughput must reflect total |dSOC| * capacity_kwh
    expected_throughput = sum(
        abs(soc_trace[i] - soc_trace[i - 1]) for i in range(1, len(soc_trace))
    ) * battery.rated_power
    assert update.cumulative_throughput_kwh == pytest.approx(expected_throughput)


@pytest.mark.asyncio
async def test_periodic_loop_processes_all_batteries(db_session, app):
    """The periodic loop should call apply_window once per battery per tick."""
    suffix = datetime.now().timestamp()
    b1 = await ResourceRepository.create(
        db_session, name=f"loop1-{suffix}", resource_type="battery",
        rated_power=100.0,
    )
    b2 = await ResourceRepository.create(
        db_session, name=f"loop2-{suffix}", resource_type="battery",
        rated_power=100.0,
    )
    await db_session.commit()

    base = datetime.now(timezone.utc)

    def fetch(bid: str) -> TelemetryWindow:
        return TelemetryWindow(
            battery_id=bid,
            soc_trace=[0.5, 0.6, 0.5],
            timestamps=[base, base + timedelta(minutes=15), base + timedelta(minutes=30)],
            temperatures_c=[25.0, 25.0, 25.0],
        )

    updater = DegradationUpdater(session_factory=get_session_factory())
    apply_mock = AsyncMock(wraps=updater.apply_window)
    updater.apply_window = apply_mock  # type: ignore[assignment]

    await updater.run_periodic(
        fetch_telemetry=fetch,
        battery_ids=[b1.id, b2.id],
        interval_minutes=0,
        max_ticks=2,
    )
    # 2 batteries * 2 ticks = 4 calls
    assert apply_mock.await_count == 4
