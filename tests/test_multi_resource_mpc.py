"""Tests for MultiResourceMPCController (M4)."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

pyo = pytest.importorskip("pyomo.environ")

from vpp.optimization.mpc import (
    MultiResourceMPCConfig,
    MultiResourceMPCController,
    MultiResourceMPCStep,
)
from vpp.optimization.formulations.fleet_dispatch import FleetBattery, FleetCoupling
from vpp.optimization.solvers.pyomo_plugin import _try_import_pyomo


@pytest.fixture(scope="module")
def solver_available():
    pyo_mod, factory = _try_import_pyomo()
    if pyo_mod is None or factory is None:
        pytest.skip("Pyomo + HiGHS not available")
    return True


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


def test_multi_resource_mpc_basic(solver_available):
    cfg = MultiResourceMPCConfig(
        horizon_steps=8,
        interval_minutes=60,
        admm_threshold=10,
    )
    batts = [_b("a"), _b("b"), _b("c")]
    ctrl = MultiResourceMPCController(cfg, batts)

    base_prices = [10.0, 10.0, 50.0, 100.0, 100.0, 50.0, 10.0, 10.0]
    soc = {b.id: 0.5 for b in batts}
    t0 = datetime(2025, 1, 1)
    decisions = []
    for tick in range(24):
        # Rolling forecast: shift prices
        prices = base_prices[tick % len(base_prices):] + base_prices[: tick % len(base_prices)]
        prices = (prices * 2)[: cfg.horizon_steps]
        step = MultiResourceMPCStep(
            timestamp=t0 + timedelta(hours=tick),
            soc_init_per_resource=soc,
            forecast={"prices": prices},
        )
        d = ctrl.step(step)
        decisions.append(d)
        # Each decision should have entries for every battery
        assert set(d.per_resource.keys()) == {"a", "b", "c"}

    assert len(decisions) == 24
    # At least one decision should have non-trivial dispatch
    any_dispatch = any(
        any(pr["p_charge_kw"] > 0.1 or pr["p_discharge_kw"] > 0.1
            for pr in d.per_resource.values())
        for d in decisions
    )
    assert any_dispatch


def test_auto_routing_to_admm(solver_available):
    cfg = MultiResourceMPCConfig(
        horizon_steps=4,
        interval_minutes=60,
        admm_threshold=10,
        admm_max_iters=20,
        admm_tolerance=2.0,
    )
    batts = [_b(f"r{i}") for i in range(12)]
    ctrl = MultiResourceMPCController(cfg, batts)

    step = MultiResourceMPCStep(
        timestamp=datetime(2025, 1, 1),
        forecast={"prices": [10.0, 10.0, 100.0, 100.0]},
    )
    d = ctrl.step(step)
    assert d.method == "admm", f"expected admm, got {d.method}"
    assert "iterations" in d.metadata
