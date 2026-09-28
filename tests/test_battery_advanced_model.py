"""Regression tests for AdvancedElectrochemicalModel power limits.

``get_max_charge_power`` / ``get_max_discharge_power`` used to call the
abstract ``BatteryModel`` implementation via ``super()``, which returns
``None`` and made every call (and therefore ``update``) raise ``TypeError``.
"""

from __future__ import annotations

import math

from vpp.config.vpp_config import ResourceConfig
from vpp.models.battery import (
    AdvancedElectrochemicalModel,
    BatteryParameters,
    SimpleEquivalentCircuitModel,
)


def _params() -> BatteryParameters:
    return BatteryParameters(
        nominal_capacity=100.0,
        nominal_voltage=400.0,
        max_voltage=450.0,
        min_voltage=320.0,
        max_current=100.0,
        internal_resistance=0.05,
    )


def _config() -> ResourceConfig:
    return ResourceConfig(name="b1", type="battery", parameters={"initial_soc": 0.5})


def test_power_limits_are_finite_and_bounded_by_equivalent_circuit_limits():
    adv = AdvancedElectrochemicalModel(_params(), _config())
    simple = SimpleEquivalentCircuitModel(_params(), _config())

    charge = adv.get_max_charge_power()
    discharge = adv.get_max_discharge_power()

    assert isinstance(charge, float)
    assert isinstance(discharge, float)
    assert 0.0 < charge <= simple.get_max_charge_power()
    assert 0.0 < discharge <= simple.get_max_discharge_power()


def test_update_runs():
    adv = AdvancedElectrochemicalModel(_params(), _config())
    state = adv.update(power_setpoint=5.0, dt=60.0)
    assert math.isfinite(state.soc)
    assert 0.0 <= state.soc <= 1.0
