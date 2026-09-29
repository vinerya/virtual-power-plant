"""SoC of the battery models must follow Coulomb counting.

``AdvancedElectrochemicalModel`` used to report a normalised "bulk
concentration" as SoC, driven by a dimensionally wrong flux
(``I / (F * Ah * 3600) * dt / (r / 3)``) and lagging the surface
concentration by the diffusion time constant. One hour at 100 kW into the
sample 2000 Ah / 400 V battery took SoC from 0.50 to 0.83 (Coulomb counting
gives ~0.62) even though the voltage limits only let ~30 kW in, and SoC kept
rising during the following 150 kW discharge, which the (per-cell) OCV
pinned at the pack's minimum voltage blocked entirely.

Both models also applied the efficiencies the wrong way round (charging
stored *more* than the power delivered, discharging drew *less* than the
power supplied), so a charge/discharge cycle created energy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vpp.config import ResourceConfig, VPPConfig
from vpp.models.battery import (
    AdvancedElectrochemicalModel,
    BatteryModel,
    BatteryParameters,
    SimpleEquivalentCircuitModel,
    create_battery_model,
)

SAMPLE = Path(__file__).resolve().parents[1] / "configs" / "advanced_vpp_config.yaml"
MODELS = ["simple", "advanced"]


def _sample_battery(model_type: str, monkeypatch, tmp_path) -> BatteryModel:
    """The battery from ``configs/advanced_vpp_config.yaml``, built like the demo."""
    monkeypatch.chdir(tmp_path)  # the sample config logs to a relative path
    config = VPPConfig.load_from_file(SAMPLE)
    assert isinstance(config, VPPConfig)
    resource = next(r for r in config.resources if r.type == "battery")
    p = resource.parameters
    params = BatteryParameters(
        nominal_capacity=p["nominal_capacity"],
        nominal_voltage=p["nominal_voltage"],
        max_voltage=p["max_voltage"],
        min_voltage=p["min_voltage"],
        max_current=p["max_current"],
        internal_resistance=p.get("internal_resistance", 0.01),
        charge_efficiency=p["charge_efficiency"],
        discharge_efficiency=p["discharge_efficiency"],
    )
    assert (params.nominal_capacity, params.nominal_voltage) == (2000.0, 400.0)
    return create_battery_model(model_type, params, resource)


def _run(model: BatteryModel, power_kw: float, minutes: int) -> float:
    """Apply ``power_kw`` in 1-minute steps; return the energy delivered (kWh)."""
    energy = 0.0
    for _ in range(minutes):
        state = model.update(power_kw, 60.0)
        energy += state.power / 60.0
        assert 0.0 <= state.soc <= 1.0
    return energy


@pytest.mark.parametrize("model_type", MODELS)
def test_one_hour_at_100kw_follows_coulomb_counting(model_type, monkeypatch, tmp_path):
    battery = _sample_battery(model_type, monkeypatch, tmp_path)
    eta_c = battery.parameters.charge_efficiency

    energy = _run(battery, 100.0, 60)

    # 0.25 C is well inside the sample battery's limits: all 100 kW goes in.
    assert energy == pytest.approx(100.0, rel=1e-9)
    # Ideal Coulomb counting: 100 kWh / (400 V * 2000 Ah) = 0.125 of capacity,
    # less the 5 % charge loss -> 0.5 + 0.95 * 0.125 = 0.61875.  The only
    # other effect is capacity fade (SoH ~ 0.9999 after an hour), worth
    # ~1e-5 of SoC, so 1e-3 is a tight bound that still leaves room for it.
    expected = 0.5 + eta_c * 100e3 / (400.0 * 2000.0)
    assert battery.state.soc == pytest.approx(expected, abs=1e-3)
    # ...and in the ballpark of the loss-free figure the issue quotes.
    assert battery.state.soc == pytest.approx(0.625, abs=0.01)


@pytest.mark.parametrize("model_type", MODELS)
def test_discharge_decreases_soc(model_type, monkeypatch, tmp_path):
    battery = _sample_battery(model_type, monkeypatch, tmp_path)
    _run(battery, 100.0, 60)
    soc_before = battery.state.soc
    eta_d = battery.parameters.discharge_efficiency

    previous = soc_before
    energy_out = 0.0
    for _ in range(30):
        state = battery.update(-150.0, 60.0)
        energy_out -= state.power / 60.0
        assert state.soc < previous  # strictly falls on every step
        previous = state.soc

    assert energy_out == pytest.approx(75.0, rel=1e-9)  # full 150 kW supplied
    expected = soc_before - 75e3 / (eta_d * 400.0 * 2000.0)
    assert battery.state.soc == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize("model_type", MODELS)
def test_soc_stays_within_limits(model_type, monkeypatch, tmp_path):
    battery = _sample_battery(model_type, monkeypatch, tmp_path)
    p = battery.parameters

    # The advanced model tapers power as the particle surface nears its
    # limit, so it approaches the SoC limits asymptotically: hence 1e-4 on
    # SoC and 10 W on the residual power rather than exact equality.
    _run(battery, 200.0, 8 * 60)  # far more than fits
    assert battery.state.soc == pytest.approx(p.max_soc, abs=1e-4)
    assert battery.state.soc <= p.max_soc
    assert battery.update(50.0, 60.0).power == pytest.approx(0.0, abs=1e-2)

    _run(battery, -200.0, 8 * 60)  # far more than is stored
    assert battery.state.soc == pytest.approx(p.min_soc, abs=1e-4)
    assert battery.state.soc >= p.min_soc
    assert battery.update(-50.0, 60.0).power == pytest.approx(0.0, abs=1e-2)


@pytest.mark.parametrize("model_type", MODELS)
def test_round_trip_does_not_create_energy(model_type, monkeypatch, tmp_path):
    battery = _sample_battery(model_type, monkeypatch, tmp_path)
    p = battery.parameters
    start = battery.state.soc

    energy_in = _run(battery, 100.0, 60)
    energy_out = 0.0
    for _ in range(3 * 360):  # 3 h cap; returning to ``start`` takes < 1 h
        if battery.state.soc <= start:
            break
        state = battery.update(-100.0, 10.0)
        energy_out -= state.power * 10.0 / 3600.0
    assert battery.state.soc <= start, "discharge never brought SoC back to its start"
    # Last step overshoots ``start`` by at most one 10 s step (~3e-4 of SoC);
    # credit that back so the comparison is at equal SoC.
    overshoot = (start - battery.state.soc) * p.nominal_capacity * p.nominal_voltage / 1000
    energy_out -= overshoot * p.discharge_efficiency

    efficiency = energy_out / energy_in
    assert efficiency < 1.0
    # Round trip = eta_c * eta_d; SoH fade over the cycle moves it by <0.1 %.
    assert efficiency == pytest.approx(p.charge_efficiency * p.discharge_efficiency, rel=5e-3)


@pytest.mark.parametrize("model_cls", [SimpleEquivalentCircuitModel, AdvancedElectrochemicalModel])
def test_initial_soc_is_respected(model_cls):
    params = BatteryParameters(
        nominal_capacity=100.0,
        nominal_voltage=400.0,
        max_voltage=450.0,
        min_voltage=320.0,
        max_current=100.0,
        internal_resistance=0.05,
    )
    config = ResourceConfig(name="b", type="battery", parameters={"initial_soc": 0.8})
    battery = model_cls(params, config)
    state = battery.update(0.0, 60.0)
    assert state.soc == pytest.approx(0.8, abs=1e-9)
