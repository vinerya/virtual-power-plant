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


def test_string_numeric_parameters_from_yaml_are_coerced():
    """PyYAML (YAML 1.1) loads ``1e-14`` as the string ``"1e-14"``; the shipped
    ``configs/advanced_vpp_config.yaml`` hit this and ``update`` raised
    ``TypeError: unsupported operand type(s) for ** or pow(): 'str' and 'int'``."""
    import yaml

    loaded = yaml.safe_load("d: 1e-14\nr: 5e-6\nt: 100e-6\nsoc: 0.5\n")
    assert isinstance(loaded["d"], str)  # the pitfall this guards against
    config = ResourceConfig(
        name="b1",
        type="battery",
        parameters={
            "initial_soc": loaded["soc"],
            "diffusion_coefficient": loaded["d"],
            "particle_radius": loaded["r"],
            "electrode_thickness": loaded["t"],
        },
    )
    adv = AdvancedElectrochemicalModel(_params(), config)
    assert adv.diffusion_coefficient == 1e-14
    assert adv.particle_radius == 5e-6
    state = adv.update(power_setpoint=5.0, dt=60.0)
    assert math.isfinite(state.soc)
