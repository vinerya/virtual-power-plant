"""Tests for Pydantic v2 schema validation."""

import pytest
from pydantic import ValidationError

from vpp.schemas.resources import BatteryCreate, SolarCreate, WindTurbineCreate, ResourceType
from vpp.schemas.optimization import DispatchRequest, StochasticRequest
from vpp.schemas.trading import OrderCreate
from vpp.schemas.auth import UserCreate


class TestResourceSchemas:
    def test_battery_create_valid(self):
        b = BatteryCreate(
            name="bat-1", rated_power=50.0, capacity_kwh=100.0,
            current_charge_kwh=50.0, nominal_voltage=48.0,
        )
        assert b.resource_type == ResourceType.BATTERY

    def test_battery_charge_exceeds_capacity(self):
        with pytest.raises(ValidationError):
            BatteryCreate(
                name="bat-x", rated_power=50.0, capacity_kwh=100.0,
                current_charge_kwh=150.0, nominal_voltage=48.0,
            )

    def test_battery_negative_power(self):
        with pytest.raises(ValidationError):
            BatteryCreate(
                name="bat-x", rated_power=-10, capacity_kwh=100.0,
                current_charge_kwh=50.0, nominal_voltage=48.0,
            )

    def test_solar_create_valid(self):
        s = SolarCreate(name="sol-1", rated_power=10.0, panel_area_m2=20.0, panel_efficiency=0.2)
        assert s.resource_type == ResourceType.SOLAR

    def test_wind_turbine_speed_validation(self):
        with pytest.raises(ValidationError):
            WindTurbineCreate(
                name="wt-x", rated_power=100.0, rotor_diameter_m=20.0,
                hub_height_m=30.0, cut_in_speed_ms=15.0,
                cut_out_speed_ms=10.0, rated_speed_ms=12.0,
            )


class TestOptimizationSchemas:
    def test_dispatch_request(self):
        d = DispatchRequest(target_power_kw=100.0)
        assert d.timeout_ms == 5000

    def test_stochastic_request_bounds(self):
        with pytest.raises(ValidationError):
            StochasticRequest(num_scenarios=0)  # min 1

    def test_stochastic_defaults(self):
        s = StochasticRequest()
        assert s.num_scenarios == 50
        assert s.time_horizon_hours == 24


class TestTradingSchemas:
    def test_order_create_valid(self):
        o = OrderCreate(order_type="limit", market="day_ahead", side="buy", quantity=100.0, price=0.12)
        assert o.time_in_force == "GTC"

    def test_order_invalid_side(self):
        with pytest.raises(ValidationError):
            OrderCreate(order_type="market", market="rt", side="hold", quantity=10.0)


class TestAuthSchemas:
    def test_user_create_valid(self):
        u = UserCreate(username="testuser", password="securepass123")
        assert u.role.value == "viewer"

    def test_username_too_short(self):
        with pytest.raises(ValidationError):
            UserCreate(username="ab", password="securepass123")

    def test_password_too_short(self):
        with pytest.raises(ValidationError):
            UserCreate(username="validuser", password="short")
