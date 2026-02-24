"""Tests for energy resource models."""

import pytest

from vpp.exceptions import ResourceError
from vpp.resources import Battery, Solar, WindTurbine


class TestBattery:
    def test_create_valid(self, battery):
        assert battery.capacity == 100.0
        assert battery.current_charge == 50.0
        assert battery.rated_power == 50.0
        assert battery.is_online

    def test_invalid_capacity(self):
        with pytest.raises(ResourceError):
            Battery(capacity=-10, current_charge=5, max_power=10, nominal_voltage=48)

    def test_charge_over_capacity(self):
        with pytest.raises(ResourceError):
            Battery(capacity=100, current_charge=150, max_power=50, nominal_voltage=48)

    def test_charge(self, battery):
        battery.charge(power=10.0, duration=1.0)
        assert battery.current_charge > 50.0

    def test_discharge(self, battery):
        battery.discharge(power=10.0, duration=1.0)
        assert battery.current_charge < 50.0

    def test_charge_exceeds_capacity(self, battery):
        with pytest.raises(ResourceError):
            battery.charge(power=50.0, duration=100.0)

    def test_discharge_exceeds_charge(self, battery):
        with pytest.raises(ResourceError):
            battery.discharge(power=50.0, duration=100.0)

    def test_metrics(self, battery):
        metrics = battery.get_metrics()
        assert "capacity" in metrics
        assert "state_of_charge" in metrics
        assert metrics["state_of_charge"] == pytest.approx(50.0)

    def test_offline_charge_fails(self, battery):
        battery._online = False
        with pytest.raises(ResourceError):
            battery.charge(10, 1)


class TestSolar:
    def test_create_valid(self, solar):
        assert solar.rated_power == 10.0
        assert solar.panel_area == 20.0

    def test_invalid_efficiency(self):
        with pytest.raises(ResourceError):
            Solar(peak_power=10, panel_area=20, efficiency=1.5)

    def test_update_conditions(self, solar):
        solar.update_conditions(irradiance=800, temperature=30)
        assert solar._current_power > 0

    def test_zero_irradiance(self, solar):
        solar.update_conditions(irradiance=0)
        assert solar._current_power == 0.0

    def test_negative_irradiance_raises(self, solar):
        with pytest.raises(ResourceError):
            solar.update_conditions(irradiance=-100)

    def test_metrics(self, solar):
        metrics = solar.get_metrics()
        assert "panel_area" in metrics
        assert "irradiance" in metrics


class TestWindTurbine:
    def test_create_valid(self, wind_turbine):
        assert wind_turbine.rated_power == 100.0
        assert wind_turbine.rotor_diameter == 20.0

    def test_invalid_speeds(self):
        with pytest.raises(ResourceError):
            WindTurbine(rated_power=100, rotor_diameter=20, hub_height=30,
                        cut_in_speed=15, cut_out_speed=10, rated_speed=12)

    def test_below_cut_in(self, wind_turbine):
        wind_turbine.update_wind(wind_speed=1.0)
        assert wind_turbine._current_power == 0.0

    def test_above_cut_out(self, wind_turbine):
        wind_turbine.update_wind(wind_speed=30.0)
        assert wind_turbine._current_power == 0.0

    def test_at_rated_speed(self, wind_turbine):
        wind_turbine.update_wind(wind_speed=12.0)
        assert wind_turbine._current_power == wind_turbine.rated_power

    def test_between_cut_in_and_rated(self, wind_turbine):
        wind_turbine.update_wind(wind_speed=8.0)
        assert 0 < wind_turbine._current_power < wind_turbine.rated_power

    def test_negative_wind_raises(self, wind_turbine):
        with pytest.raises(ResourceError):
            wind_turbine.update_wind(wind_speed=-5)

    def test_metrics(self, wind_turbine):
        metrics = wind_turbine.get_metrics()
        assert "rotor_diameter" in metrics
        assert "wind_speed" in metrics


class TestResourceBase:
    def test_set_power(self, battery):
        battery.set_power(25.0)
        assert battery._current_power == 25.0

    def test_set_power_over_rated(self, battery):
        with pytest.raises(ResourceError):
            battery.set_power(battery.rated_power + 1)

    def test_metadata(self, battery):
        battery.add_metadata("location", "Building A")
        assert battery.get_metadata()["location"] == "Building A"

    def test_state_history(self, battery):
        battery.set_power(10.0)
        battery.set_power(20.0)
        history = battery.get_state_history()
        assert len(history) >= 2
