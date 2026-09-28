"""Tests for ADMM fleet dispatch (M4)."""

from __future__ import annotations

import pytest

pyo = pytest.importorskip("pyomo.environ")

from vpp.optimization.formulations.admm import admm_fleet_solve
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


def _b(id_, soc=0.5):
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


def test_admm_converges_on_5_battery(solver):
    prices = [10.0, 10.0, 100.0, 100.0, 10.0, 10.0]
    batts = [_b(f"r{i}") for i in range(5)]
    # Use a non-binding feeder so consensus can match exactly.
    result = admm_fleet_solve(
        batts,
        len(prices),
        1.0,
        prices,
        coupling=FleetCoupling(feeder_max_export_kw=500.0, feeder_max_import_kw=500.0),
        rho=2.0,
        max_iters=40,
        tolerance=1.0,  # consensus tolerance in kW
    )
    assert result["converged"], (
        f"ADMM did not converge. iters={result['iterations']}, "
        f"primal={result['primal_residual']}, dual={result['dual_residual']}"
    )


def test_admm_matches_monolithic(solver):
    pyo_mod, factory = solver
    prices = [10.0, 10.0, 100.0, 100.0, 10.0, 10.0]
    batts = [_b(f"r{i}") for i in range(3)]

    # Monolithic
    m = build_fleet_dispatch_model(batts, len(prices), 1.0, prices)
    s = factory(10.0)
    s.solve(m)
    mono_obj = float(pyo_mod.value(m.cost))

    # ADMM
    result = admm_fleet_solve(
        batts, len(prices), 1.0, prices, rho=1.0, max_iters=60, tolerance=0.5
    )
    admm_obj = result["objective"]

    # Within 5% — L1 ADMM linearization has more slack than L2 ADMM
    if abs(mono_obj) > 1e-6:
        rel = abs(admm_obj - mono_obj) / max(abs(mono_obj), 1.0)
        assert rel < 0.05, f"ADMM obj {admm_obj} far from mono {mono_obj} (rel={rel})"
    else:
        assert abs(admm_obj - mono_obj) < 5.0


def test_admm_respects_feeder_limit(solver):
    prices = [10.0, 10.0, 100.0, 100.0]
    batts = [_b(f"r{i}", soc=0.9) for i in range(3)]
    coupling = FleetCoupling(feeder_max_export_kw=30.0, feeder_max_import_kw=200.0)
    result = admm_fleet_solve(
        batts,
        len(prices),
        1.0,
        prices,
        coupling=coupling,
        rho=2.0,
        max_iters=80,
        tolerance=0.5,
    )
    # With 1% slack for ADMM duality
    for t in range(len(prices)):
        assert result["p_export"][t] <= 30.0 * 1.01 + 0.5, (
            f"feeder export breached at t={t}: {result['p_export'][t]}"
        )
