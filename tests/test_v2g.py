"""Tests for V2G models, scheduler, and aggregator."""

import time
import pytest

from vpp.v2g.models import (
    EVBattery,
    EVFleet,
    EVConnectionState,
    FlexibilityWindow,
    ChargingSession,
    ScheduleStatus,
)
from vpp.v2g.scheduler import V2GScheduler
from vpp.v2g.aggregator import V2GAggregator, DispatchSignal, GridService


# ---------------------------------------------------------------------------
# EV Battery tests
# ---------------------------------------------------------------------------

class TestEVBattery:
    def test_defaults(self):
        ev = EVBattery()
        assert ev.capacity_kwh == 60.0
        assert ev.current_soc == 0.5
        assert ev.v2g_capable is True

    def test_current_energy(self):
        ev = EVBattery(capacity_kwh=100.0, current_soc=0.6)
        assert ev.current_energy_kwh == 60.0

    def test_energy_needed(self):
        ev = EVBattery(capacity_kwh=100.0, current_soc=0.5, target_soc=0.8, charge_efficiency=1.0)
        assert ev.energy_needed_kwh == pytest.approx(30.0)

    def test_available_discharge(self):
        ev = EVBattery(capacity_kwh=100.0, current_soc=0.5, min_soc=0.2, v2g_capable=True)
        assert ev.available_discharge_kwh == pytest.approx(30.0)

    def test_available_discharge_not_v2g(self):
        ev = EVBattery(v2g_capable=False)
        assert ev.available_discharge_kwh == 0.0

    def test_charge(self):
        ev = EVBattery(capacity_kwh=100.0, current_soc=0.5, max_charge_kw=50.0, charge_efficiency=1.0)
        added = ev.charge(50.0, 0.5)
        assert added == pytest.approx(25.0)
        assert ev.current_soc == pytest.approx(0.75)

    def test_charge_clamped_at_full(self):
        ev = EVBattery(capacity_kwh=100.0, current_soc=0.95, max_charge_kw=50.0, charge_efficiency=1.0)
        added = ev.charge(50.0, 1.0)
        assert added == pytest.approx(5.0)
        assert ev.current_soc == pytest.approx(1.0)

    def test_discharge(self):
        ev = EVBattery(capacity_kwh=100.0, current_soc=0.5, min_soc=0.2, max_discharge_kw=50.0, discharge_efficiency=1.0, v2g_capable=True)
        delivered = ev.discharge(50.0, 0.5)
        assert delivered == pytest.approx(25.0)
        assert ev.current_soc == pytest.approx(0.25)

    def test_discharge_clamped_at_min(self):
        ev = EVBattery(capacity_kwh=100.0, current_soc=0.25, min_soc=0.2, max_discharge_kw=50.0, discharge_efficiency=1.0, v2g_capable=True)
        delivered = ev.discharge(50.0, 1.0)
        assert delivered == pytest.approx(5.0)
        assert ev.current_soc == pytest.approx(0.2)

    def test_has_flexibility_no_departure(self):
        ev = EVBattery(departure_time=None)
        assert ev.has_flexibility is True

    def test_has_flexibility_tight(self):
        ev = EVBattery(
            capacity_kwh=60.0,
            current_soc=0.1,
            target_soc=0.9,
            max_charge_kw=11.0,
            charge_efficiency=0.9,
            departure_time=time.time() + 3600,  # 1 hour from now
        )
        # Needs ~53kWh / 11kW = ~4.8h, but only 1h available
        assert ev.has_flexibility is False

    def test_to_dict(self):
        ev = EVBattery(capacity_kwh=60.0, current_soc=0.5)
        d = ev.to_dict()
        assert d["capacity_kwh"] == 60.0
        assert d["current_soc"] == 0.5
        assert "has_flexibility" in d


# ---------------------------------------------------------------------------
# Fleet tests
# ---------------------------------------------------------------------------

class TestEVFleet:
    def _make_fleet(self, n: int = 3) -> EVFleet:
        fleet = EVFleet(name="test_fleet")
        for i in range(n):
            ev = EVBattery(
                ev_id=f"ev-{i}",
                capacity_kwh=60.0,
                current_soc=0.5,
                v2g_capable=True,
                connection_state=EVConnectionState.CONNECTED_IDLE,
            )
            fleet.add_vehicle(ev)
        return fleet

    def test_add_remove(self):
        fleet = self._make_fleet(3)
        assert len(fleet.vehicles) == 3
        fleet.remove_vehicle("ev-0")
        assert len(fleet.vehicles) == 2

    def test_aggregated_metrics(self):
        fleet = self._make_fleet(3)
        assert fleet.total_capacity_kwh == pytest.approx(180.0)
        assert fleet.average_soc == pytest.approx(0.5)
        assert fleet.connected_count == 3

    def test_discharge_capacity(self):
        fleet = self._make_fleet(2)
        assert fleet.total_discharge_capacity_kw == pytest.approx(22.0)  # 11 + 11

    def test_flexibility_windows(self):
        fleet = self._make_fleet(2)
        windows = fleet.get_flexibility_windows()
        assert len(windows) == 2
        for w in windows:
            assert w.max_discharge_kw > 0

    def test_to_dict(self):
        fleet = self._make_fleet(2)
        d = fleet.to_dict()
        assert d["vehicle_count"] == 2
        assert d["connected_count"] == 2


# ---------------------------------------------------------------------------
# Scheduler tests
# ---------------------------------------------------------------------------

class TestV2GScheduler:
    def _make_fleet_with_prices(self):
        fleet = EVFleet(name="sched_fleet")
        ev = EVBattery(
            ev_id="ev-sched",
            capacity_kwh=60.0,
            current_soc=0.3,
            target_soc=0.8,
            min_soc=0.2,
            max_charge_kw=11.0,
            max_discharge_kw=11.0,
            charge_efficiency=1.0,
            discharge_efficiency=1.0,
            v2g_capable=True,
            connection_state=EVConnectionState.CONNECTED_IDLE,
            departure_time=time.time() + 24 * 3600,
        )
        fleet.add_vehicle(ev)
        # Price pattern: cheap at night, expensive during peak
        prices = (
            [0.05] * 24  # 6 hours cheap (at 15-min slots)
            + [0.15] * 48  # 12 hours mid
            + [0.30] * 24  # 6 hours expensive
        )
        return fleet, prices

    def test_rule_based_schedule(self):
        fleet, prices = self._make_fleet_with_prices()
        scheduler = V2GScheduler(slot_duration_minutes=15)
        result = scheduler.schedule_fleet(fleet, prices=prices, time_horizon_hours=24.0)

        assert result.method == "rule_based"
        assert len(result.schedule) > 0
        assert result.total_cost >= 0
        assert result.solve_time_ms > 0

    def test_optimised_schedule(self):
        fleet, prices = self._make_fleet_with_prices()
        scheduler = V2GScheduler(slot_duration_minutes=15)
        result = scheduler.schedule_fleet(
            fleet, prices=prices, time_horizon_hours=24.0, use_optimiser=True,
        )

        assert result.method == "optimised_lp"
        assert len(result.schedule) > 0

    def test_no_flexibility(self):
        fleet = EVFleet()
        scheduler = V2GScheduler()
        result = scheduler.schedule_fleet(fleet)
        assert result.method == "no_flexibility"
        assert len(result.schedule) == 0

    def test_single_ev_schedule(self):
        ev = EVBattery(
            current_soc=0.4, target_soc=0.8, capacity_kwh=60.0,
            connection_state=EVConnectionState.CONNECTED_IDLE,
            departure_time=time.time() + 12 * 3600,
        )
        scheduler = V2GScheduler()
        result = scheduler.schedule_single(ev, prices=[0.10] * 48)
        assert len(result.schedule) > 0


# ---------------------------------------------------------------------------
# Aggregator tests
# ---------------------------------------------------------------------------

class TestV2GAggregator:
    def _make_aggregator(self, n: int = 5) -> V2GAggregator:
        fleet = EVFleet(name="agg_fleet")
        for i in range(n):
            ev = EVBattery(
                ev_id=f"ev-{i}",
                capacity_kwh=60.0,
                current_soc=0.6,
                min_soc=0.2,
                max_charge_kw=11.0,
                max_discharge_kw=11.0,
                v2g_capable=True,
                connection_state=EVConnectionState.CONNECTED_IDLE,
                departure_time=time.time() + 12 * 3600,
            )
            fleet.add_vehicle(ev)
        return V2GAggregator(fleet)

    def test_assess_flexibility(self):
        agg = self._make_aggregator(5)
        flex = agg.assess_flexibility()
        assert flex["connected_evs"] == 5
        assert flex["v2g_capable_evs"] == 5
        assert flex["max_discharge_kw"] == pytest.approx(55.0)

    def test_dispatch_discharge(self):
        agg = self._make_aggregator(5)
        signal = DispatchSignal(target_power_kw=-30.0, duration_seconds=900)
        result = agg.dispatch(signal)

        assert result.achieved_power_kw < 0
        assert abs(result.achieved_power_kw) <= 55.0
        assert len(result.ev_allocations) > 0
        assert result.achievement_ratio > 0

    def test_dispatch_charge(self):
        agg = self._make_aggregator(3)
        signal = DispatchSignal(target_power_kw=20.0, duration_seconds=900)
        result = agg.dispatch(signal)

        assert result.achieved_power_kw > 0
        assert len(result.ev_allocations) > 0

    def test_dispatch_exceeds_capacity(self):
        agg = self._make_aggregator(2)  # 22 kW discharge capacity
        signal = DispatchSignal(target_power_kw=-100.0, duration_seconds=900)
        result = agg.dispatch(signal)

        assert result.shortfall_kw > 0
        assert result.achievement_ratio < 1.0

    def test_create_flexibility_bid(self):
        agg = self._make_aggregator(5)
        bid = agg.create_flexibility_bid(
            service=GridService.FREQUENCY_REGULATION,
            capacity_fraction=0.8,
            duration_hours=1.0,
            price_per_kw=0.05,
        )
        assert bid is not None
        assert bid.capacity_kw > 0
        assert bid.total_value > 0

    def test_create_bid_insufficient_flexibility(self):
        fleet = EVFleet()
        agg = V2GAggregator(fleet)
        bid = agg.create_flexibility_bid(service=GridService.PEAK_SHAVING)
        assert bid is None

    def test_metrics(self):
        agg = self._make_aggregator(3)
        agg.dispatch(DispatchSignal(target_power_kw=-10.0))
        metrics = agg.get_metrics()
        assert metrics["total_dispatches"] == 1
        assert "fleet" in metrics
        assert "flexibility" in metrics
