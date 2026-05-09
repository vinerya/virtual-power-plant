"""Tests for multi-resource fleet dispatch (M4)."""
from __future__ import annotations

import pytest

pyo = pytest.importorskip("pyomo.environ")

from vpp.optimization.formulations.fleet_dispatch import (
    FleetBattery,
    FleetCoupling,
    build_fleet_dispatch_model,
)
from vpp.optimization.solvers.pyomo_plugin import _try_import_pyomo


@pytest.fixture(scope="module")
def solver():
    pyo_mod, factory = _try_import_pyomo()
    if pyo_mod is None or factory is None:
        pytest.skip("Pyomo + HiGHS not available")
    return pyo_mod, factory


def _solve(model, factory, time_limit=10.0):
    s = factory(time_limit)
    s.solve(model)


def _identical(id_, soc=0.5):
    return FleetBattery(
        id=id_,
        capacity_kwh=100.0,
        max_charge_kw=50.0,
        max_discharge_kw=50.0,
        soc_init=soc,
        soc_min=0.05,
        soc_max=0.95,
        eta_charge=0.95,
        eta_discharge=0.95,
    )


def test_two_battery_balance_split(solver):
    pyo_mod, factory = solver
    # Sinusoidal-ish prices to encourage clear arbitrage
    prices = [10.0, 10.0, 10.0, 10.0, 100.0, 100.0, 100.0, 100.0]
    batts = [_identical("a", soc=0.5), _identical("b", soc=0.5)]
    m = build_fleet_dispatch_model(batts, len(prices), 1.0, prices)
    _solve(m, factory)

    total_chg_a = sum(float(pyo_mod.value(m.p_charge["a", t])) for t in range(len(prices)))
    total_chg_b = sum(float(pyo_mod.value(m.p_charge["b", t])) for t in range(len(prices)))
    total_dis_a = sum(float(pyo_mod.value(m.p_discharge["a", t])) for t in range(len(prices)))
    total_dis_b = sum(float(pyo_mod.value(m.p_discharge["b", t])) for t in range(len(prices)))

    # Both batteries should participate (roughly symmetric, allow generous tol)
    assert total_chg_a > 0 and total_chg_b > 0
    assert total_dis_a > 0 and total_dis_b > 0
    # Sums roughly equal (within 25%)
    if total_chg_a + total_chg_b > 0:
        ratio = abs(total_chg_a - total_chg_b) / (total_chg_a + total_chg_b)
        assert ratio < 0.25


def test_feeder_limit_binds(solver):
    pyo_mod, factory = solver
    prices = [10.0, 10.0, 10.0, 10.0, 100.0, 100.0, 100.0, 100.0]
    batts = [_identical("a", 0.9), _identical("b", 0.9)]  # high SOC -> want to discharge
    coupling = FleetCoupling(feeder_max_export_kw=30.0)
    m = build_fleet_dispatch_model(batts, len(prices), 1.0, prices, coupling=coupling)
    _solve(m, factory)

    for t in range(len(prices)):
        exp = float(pyo_mod.value(m.p_export[t]))
        assert exp <= 30.0 + 1e-4, f"export exceeded cap at t={t}: {exp}"


def test_reserve_capacity_held(solver):
    pyo_mod, factory = solver
    prices = [10.0, 10.0, 100.0, 100.0]
    batts = [_identical("a", 0.5), _identical("b", 0.5)]
    coupling = FleetCoupling(reserve_capacity_kw=80.0)
    m = build_fleet_dispatch_model(batts, len(prices), 1.0, prices, coupling=coupling)
    _solve(m, factory)

    # At each timestep, headroom across resources must >= 80 kW
    for t in range(len(prices)):
        hr = sum(
            (float(pyo_mod.value(m.p_chg_max[r])) - float(pyo_mod.value(m.p_charge[r, t])))
            + (float(pyo_mod.value(m.p_dis_max[r])) - float(pyo_mod.value(m.p_discharge[r, t])))
            for r in m.R
        )
        assert hr >= 80.0 - 1e-3, f"reserve headroom violated at t={t}: {hr}"


def test_aggregate_matches_per_battery_sum(solver):
    pyo_mod, factory = solver
    prices = [10.0, 50.0, 90.0, 50.0]
    batts = [_identical("a", 0.5), _identical("b", 0.5), _identical("c", 0.5)]
    m = build_fleet_dispatch_model(batts, len(prices), 1.0, prices)
    _solve(m, factory)

    for t in range(len(prices)):
        agg = float(pyo_mod.value(m.p_aggregate[t]))
        manual = sum(
            float(pyo_mod.value(m.p_discharge[r, t])) - float(pyo_mod.value(m.p_charge[r, t]))
            for r in m.R
        )
        assert abs(agg - manual) < 1e-6


def test_hook_api_per_resource(solver):
    """A per-battery throughput penalty hook reduces cycling."""
    pyo_mod, factory = solver
    prices = [10.0, 10.0, 100.0, 100.0, 10.0, 10.0, 100.0, 100.0]
    batts = [_identical("a", 0.5), _identical("b", 0.5)]

    # Baseline
    m1 = build_fleet_dispatch_model(batts, len(prices), 1.0, prices)
    _solve(m1, factory)
    base_throughput = {
        r: sum(
            float(pyo_mod.value(m1.p_charge[r, t])) + float(pyo_mod.value(m1.p_discharge[r, t]))
            for t in range(len(prices))
        )
        for r in m1.R
    }

    # With per-resource throughput penalty (lambda=10)
    def throughput_penalty(model, params):
        return 10.0 * sum(
            model.p_charge[r, t] + model.p_discharge[r, t]
            for r in model.R
            for t in model.T
        )

    m2 = build_fleet_dispatch_model(
        batts, len(prices), 1.0, prices, objective_terms=[throughput_penalty]
    )
    _solve(m2, factory)
    pen_throughput = {
        r: sum(
            float(pyo_mod.value(m2.p_charge[r, t])) + float(pyo_mod.value(m2.p_discharge[r, t]))
            for t in range(len(prices))
        )
        for r in m2.R
    }
    for r in base_throughput:
        assert pen_throughput[r] <= base_throughput[r] + 1e-3, (
            f"penalty did not reduce throughput for {r}: "
            f"base={base_throughput[r]} pen={pen_throughput[r]}"
        )
