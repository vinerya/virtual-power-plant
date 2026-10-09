"""Battery models own their aging state and keep bounded histories.

``AdvancedElectrochemicalModel`` used to age by multiplying
``parameters.internal_resistance`` in place, so models sharing one
``BatteryParameters`` object aged each other (and the caller's config), and
the power/SoC histories of that model grew without bound.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from vpp.config.vpp_config import ResourceConfig
from vpp.models.battery import (
    DEFAULT_MAX_HISTORY,
    AdvancedElectrochemicalModel,
    BatteryParameters,
    SimpleEquivalentCircuitModel,
    create_battery_model,
)
from vpp.resources import Battery


def _params(**overrides: float) -> BatteryParameters:
    values: dict[str, float] = {
        "nominal_capacity": 100.0,
        "nominal_voltage": 400.0,
        "max_voltage": 450.0,
        "min_voltage": 320.0,
        "max_current": 100.0,
        "internal_resistance": 0.05,
    }
    values.update(overrides)
    return BatteryParameters(**values)  # type: ignore[arg-type]


def _config(**parameters: object) -> ResourceConfig:
    return ResourceConfig(name="b1", type="battery", parameters={"initial_soc": 0.5, **parameters})


@pytest.mark.parametrize("model_cls", [AdvancedElectrochemicalModel, SimpleEquivalentCircuitModel])
def test_shared_parameters_are_not_mutated_and_models_age_independently(model_cls):
    # Exaggerated aging rates so a few steps produce a visible change.
    shared = _params(capacity_fade_rate=0.5, resistance_growth_rate=50.0)
    snapshot = dataclasses.asdict(shared)
    worked = model_cls(shared, _config())
    idle = model_cls(shared, _config())

    for i in range(50):
        worked.update(20.0 if i % 2 == 0 else -20.0, 60.0)

    assert dataclasses.asdict(shared) == snapshot
    assert worked.parameters is shared
    assert worked.internal_resistance > shared.internal_resistance
    assert worked.state.soh < 1.0
    assert idle.internal_resistance == shared.internal_resistance
    assert idle.state.soh == 1.0


def test_advanced_aging_matches_expected_fade_over_n_steps():
    p = _params(capacity_fade_rate=0.01, calendar_fade_rate=0.02, resistance_growth_rate=3.0)
    model = AdvancedElectrochemicalModel(p, _config())
    model.state.temperature = 25.0  # temp_factor == 1

    current, dt, n = 50.0, 60.0, 100  # below the lithium-plating threshold
    for _ in range(n):
        model._update_aging(current, dt)

    per_step = (
        p.calendar_fade_rate * dt / 86400
        + p.capacity_fade_rate * (current / p.max_current) ** 1.5 * dt / 3600
    )
    total = n * per_step
    assert model.state.soh == pytest.approx(1.0 - total, rel=1e-12)
    assert model.internal_resistance == pytest.approx(
        p.internal_resistance * (1 + p.resistance_growth_rate * total), rel=1e-12
    )
    assert p.internal_resistance == 0.05


def test_advanced_resistance_depends_on_total_aging_not_step_count():
    p = _params(capacity_fade_rate=0.05, resistance_growth_rate=40.0)
    fine = AdvancedElectrochemicalModel(p, _config())
    coarse = AdvancedElectrochemicalModel(p, _config())
    fine.state.temperature = coarse.state.temperature = 25.0

    for _ in range(600):
        fine._update_aging(50.0, 1.0)
    coarse._update_aging(50.0, 600.0)

    # Per-step re-multiplication (the old behaviour) would compound and make
    # the finely stepped model age faster.
    assert fine.internal_resistance == pytest.approx(coarse.internal_resistance, rel=1e-9)
    assert fine.state.soh == pytest.approx(coarse.state.soh, rel=1e-12)


def test_advanced_voltage_and_power_limits_use_aged_resistance():
    p = _params()
    fresh = AdvancedElectrochemicalModel(p, _config())
    aged = AdvancedElectrochemicalModel(p, _config())
    aged._cumulative_aging = 2.0 / p.resistance_growth_rate  # R = 3 * nominal
    assert aged.internal_resistance == pytest.approx(3 * p.internal_resistance)

    # Same starting state; only the resistance differs.
    fresh_state = fresh.update(10.0, 1.0)
    aged_state = aged.update(10.0, 1.0)
    assert aged_state.current == pytest.approx(fresh_state.current)
    assert aged_state.voltage - fresh_state.voltage == pytest.approx(
        fresh_state.current * 2 * p.internal_resistance, rel=1e-6
    )

    # Higher resistance means less power fits under the voltage limits.
    fresh2 = AdvancedElectrochemicalModel(p, _config())
    aged2 = AdvancedElectrochemicalModel(p, _config())
    aged2._cumulative_aging = 200.0 / p.resistance_growth_rate  # R = 201 * nominal
    assert aged2.get_max_charge_power() < fresh2.get_max_charge_power()
    assert aged2.get_max_discharge_power() < fresh2.get_max_discharge_power()


@pytest.mark.parametrize("model_type", ["advanced", "simple"])
def test_history_is_capped_and_accessors_return_lists(model_type):
    model = create_battery_model(model_type, _params(), _config(), max_history=5)
    for i in range(20):
        model.update(10.0 if i % 2 == 0 else -10.0, 60.0)

    soc_history = model.get_soc_history()
    power_history = model.get_power_history()
    assert isinstance(soc_history, list)
    assert isinstance(power_history, list)
    assert len(soc_history) == len(power_history) == 5
    assert soc_history[-1][1] == model.state.soc
    assert power_history[-1][1] == model.state.power
    assert all(math.isfinite(v) for _, v in soc_history)


def test_history_cap_from_config_and_default():
    from_config = AdvancedElectrochemicalModel(_params(), _config(max_history="3"))
    assert from_config.max_history == 3
    for _ in range(10):
        from_config.update(5.0, 60.0)
    assert len(from_config.get_soc_history()) == 3

    default = AdvancedElectrochemicalModel(_params(), _config())
    assert default.max_history == DEFAULT_MAX_HISTORY

    # Explicit argument wins over the config value.
    explicit = SimpleEquivalentCircuitModel(_params(), _config(max_history=3), max_history=7)
    assert explicit.max_history == 7


@pytest.mark.parametrize("bad", [0, -1, "many"])
def test_invalid_history_cap_is_rejected(bad):
    with pytest.raises(ValueError):
        AdvancedElectrochemicalModel(_params(), _config(max_history=bad))


def test_resource_state_history_is_capped():
    battery = Battery(capacity=100.0, current_charge=50.0, max_power=50.0, nominal_voltage=400.0)
    battery._state_history = type(battery._state_history)(maxlen=4)
    for power in range(1, 11):
        battery.set_power(float(power))
    history = battery.get_state_history()
    assert isinstance(history, list)
    assert [h["power"] for h in history] == [7.0, 8.0, 9.0, 10.0]
