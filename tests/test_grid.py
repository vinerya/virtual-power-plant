"""Tests for grid-forming/following inverters and microgrid controller."""

import math
import pytest

from vpp.grid.inverter import (
    GridFollowingInverter,
    GridFormingInverter,
    DroopSettings,
    VSMSettings,
    InverterState,
)
from vpp.grid.microgrid import (
    MicrogridController,
    MicrogridState,
    LoadPriority,
)


# ---------------------------------------------------------------------------
# Grid-following inverter tests
# ---------------------------------------------------------------------------

class TestGridFollowingInverter:
    def test_create(self):
        inv = GridFollowingInverter("GFol-1", rated_power_kw=100.0)
        assert inv.rated_power_kw == 100.0
        assert inv.state.real_power_kw == 0.0

    def test_set_power_reference(self):
        inv = GridFollowingInverter("GFol-1", rated_power_kw=100.0)
        inv.set_power_reference(50.0, 10.0)
        state = inv.update(1.0, grid_voltage_pu=1.0, grid_frequency_hz=50.0)
        assert state.real_power_kw == pytest.approx(50.0, abs=1)
        assert state.reactive_power_kvar == pytest.approx(10.0)

    def test_ramp_rate(self):
        inv = GridFollowingInverter("GFol-1", rated_power_kw=100.0, ramp_rate_kw_per_s=10.0)
        inv.set_power_reference(100.0)
        state = inv.update(1.0, grid_voltage_pu=1.0, grid_frequency_hz=50.0)
        # Should ramp by 10 kW in 1 second
        assert state.real_power_kw == pytest.approx(10.0)

    def test_tracks_grid(self):
        inv = GridFollowingInverter("GFol-1", rated_power_kw=100.0)
        state = inv.update(0.1, grid_voltage_pu=0.98, grid_frequency_hz=49.9)
        assert state.voltage_pu == pytest.approx(0.98)
        assert state.frequency_hz == pytest.approx(49.9)

    def test_clamps_to_rated(self):
        inv = GridFollowingInverter("GFol-1", rated_power_kw=50.0)
        inv.set_power_reference(200.0)
        assert inv._p_ref == 50.0

    def test_offline(self):
        inv = GridFollowingInverter("GFol-1", rated_power_kw=100.0)
        inv.state.is_online = False
        inv.set_power_reference(50.0)
        state = inv.update(1.0, 1.0, 50.0)
        assert state.real_power_kw == 0.0  # stays at 0, doesn't ramp


# ---------------------------------------------------------------------------
# Grid-forming inverter tests
# ---------------------------------------------------------------------------

class TestGridFormingInverter:
    def test_create(self):
        inv = GridFormingInverter("GFM-1", rated_power_kw=200.0)
        assert inv.rated_power_kw == 200.0

    def test_droop_response_frequency_drop(self):
        droop = DroopSettings(p_droop=0.05, deadband_hz=0.01)
        inv = GridFormingInverter("GFM-1", rated_power_kw=100.0, droop=droop)
        inv.set_power_reference(0.0)

        # Frequency drops by 1 Hz (below 50 Hz) -> should increase power
        state = inv.update(0.1, grid_voltage_pu=1.0, grid_frequency_hz=49.0)
        assert state.real_power_kw > 0

    def test_droop_response_frequency_rise(self):
        droop = DroopSettings(p_droop=0.05, deadband_hz=0.01)
        inv = GridFormingInverter("GFM-1", rated_power_kw=100.0, droop=droop)
        inv.set_power_reference(0.0)

        # Frequency rises -> should decrease power (absorb)
        state = inv.update(0.1, grid_voltage_pu=1.0, grid_frequency_hz=51.0)
        assert state.real_power_kw < 0

    def test_droop_voltage_reactive_power(self):
        droop = DroopSettings(q_droop=0.05, deadband_pu=0.005)
        inv = GridFormingInverter("GFM-1", rated_power_kw=100.0, droop=droop)
        inv.set_power_reference(0.0)

        # Voltage drops -> should inject reactive power
        state = inv.update(0.1, grid_voltage_pu=0.95, grid_frequency_hz=50.0)
        assert state.reactive_power_kvar > 0

    def test_droop_deadband(self):
        droop = DroopSettings(p_droop=0.05, deadband_hz=0.05)
        inv = GridFormingInverter("GFM-1", rated_power_kw=100.0, droop=droop)
        inv.set_power_reference(0.0)

        # Small frequency deviation within deadband
        state = inv.update(0.1, grid_voltage_pu=1.0, grid_frequency_hz=49.98)
        assert state.real_power_kw == pytest.approx(0.0, abs=0.1)

    def test_vsm_mode(self):
        vsm = VSMSettings(inertia_constant_s=5.0, damping_coefficient=20.0)
        inv = GridFormingInverter("GFM-1", rated_power_kw=100.0, vsm=vsm)
        inv.enable_vsm(True)
        inv.set_power_reference(50.0)

        # Simulate several steps
        for _ in range(10):
            state = inv.update(0.01, grid_voltage_pu=1.0, grid_frequency_hz=50.0)

        # VSM should settle toward the reference
        assert abs(state.frequency_hz - 50.0) < 1.0

    def test_inertia_response(self):
        vsm = VSMSettings(inertia_constant_s=5.0)
        inv = GridFormingInverter("GFM-1", rated_power_kw=100.0, vsm=vsm)
        inv.state.frequency_hz = 50.0

        # Frequency event: sudden drop
        p_inertia = inv.inertia_response(49.5)
        assert p_inertia > 0  # should inject power to arrest drop

    def test_to_dict(self):
        inv = GridFormingInverter("GFM-1", rated_power_kw=100.0)
        d = inv.to_dict()
        assert d["inverter_id"] == "GFM-1"
        assert "droop" in d


# ---------------------------------------------------------------------------
# Microgrid controller tests
# ---------------------------------------------------------------------------

class TestMicrogridController:
    def _make_controller(self) -> MicrogridController:
        ctrl = MicrogridController(nominal_frequency_hz=50.0)
        # Add one grid-forming inverter
        gfm = GridFormingInverter("GFM-1", rated_power_kw=100.0)
        gfm.state.is_online = True
        gfm.state.real_power_kw = 80.0
        ctrl.add_inverter(gfm)
        # Add loads
        ctrl.add_load(LoadPriority(load_id="critical", power_kw=30.0, priority=1))
        ctrl.add_load(LoadPriority(load_id="important", power_kw=30.0, priority=3))
        ctrl.add_load(LoadPriority(load_id="deferrable", power_kw=30.0, priority=7))
        return ctrl

    def test_initial_state(self):
        ctrl = self._make_controller()
        assert ctrl.state == MicrogridState.GRID_CONNECTED
        assert not ctrl.is_islanded

    def test_island_detection(self):
        ctrl = MicrogridController(frequency_threshold_hz=0.5)
        assert ctrl.detect_island(grid_frequency_hz=50.0, grid_voltage_pu=1.0) is False
        assert ctrl.detect_island(grid_frequency_hz=49.0, grid_voltage_pu=1.0) is True
        assert ctrl.detect_island(grid_frequency_hz=50.0, grid_voltage_pu=0.8) is True

    def test_islanding_transition(self):
        ctrl = self._make_controller()
        ok = ctrl.initiate_islanding()
        assert ok
        assert ctrl.state == MicrogridState.ISLANDED
        assert ctrl.is_islanded
        assert ctrl.metrics.island_events == 1

    def test_islanding_no_gfm(self):
        ctrl = MicrogridController()
        gfol = GridFollowingInverter("GFol-1", rated_power_kw=50.0)
        ctrl.add_inverter(gfol)
        ok = ctrl.initiate_islanding()
        assert not ok
        assert ctrl.state == MicrogridState.FAULT

    def test_reconnection(self):
        ctrl = self._make_controller()
        ctrl.initiate_islanding()
        assert ctrl.state == MicrogridState.ISLANDED

        # Simulate grid recovery with matching conditions
        gfm = list(ctrl._inverters.values())[0]
        gfm.state.frequency_hz = 50.0
        gfm.state.voltage_pu = 1.0

        ok = ctrl.initiate_reconnection(
            grid_frequency_hz=50.0,
            grid_voltage_pu=1.0,
            phase_angle_deg=0.0,
        )
        assert ok
        assert ctrl.state == MicrogridState.GRID_CONNECTED
        assert ctrl.metrics.reconnection_events == 1

    def test_load_shedding(self):
        ctrl = MicrogridController(nominal_frequency_hz=50.0)
        # Small inverter that can't serve all loads
        gfm = GridFormingInverter("GFM-1", rated_power_kw=50.0)
        gfm.state.is_online = True
        gfm.state.real_power_kw = 0.0
        ctrl.add_inverter(gfm)

        ctrl.add_load(LoadPriority(load_id="critical", power_kw=20.0, priority=1))
        ctrl.add_load(LoadPriority(load_id="medium", power_kw=20.0, priority=5))
        ctrl.add_load(LoadPriority(load_id="low", power_kw=20.0, priority=9))

        ctrl.initiate_islanding()
        # Total demand = 60 kW, capacity = 50 * 0.8 = 40 kW
        # Should shed the lowest priority load(s)
        assert ctrl.get_shed_load_kw() > 0
        assert ctrl.metrics.load_shed_events > 0

    def test_manual_shed_restore(self):
        ctrl = self._make_controller()
        count = ctrl.shed_loads_below_priority(5)
        assert count == 1  # only the priority 7 load

        restored = ctrl.restore_loads_above_priority(10)
        assert restored == 1

    def test_to_dict(self):
        ctrl = self._make_controller()
        d = ctrl.to_dict()
        assert d["state"] == "grid_connected"
        assert d["inverter_count"] == 1
        assert d["load_count"] == 3
        assert "metrics" in d
