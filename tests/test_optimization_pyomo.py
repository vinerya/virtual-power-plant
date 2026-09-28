"""Tests for the M1 Pyomo + HiGHS battery dispatch plugin."""

from __future__ import annotations

import builtins
import importlib
import sys
from typing import Any

import pytest

from vpp.optimization import (
    OptimizationProblem,
    OptimizationStatus,
    solve_with_fallback,
)
from vpp.optimization.solvers import PyomoPlugin, SimpleBatteryDispatchRules


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _two_peak_prices() -> list[float]:
    """24-step price vector with a clear morning + evening peak."""
    base = 0.10
    p = [base] * 24
    # Cheap overnight (charge windows)
    for h in (0, 1, 2, 3, 4, 13, 14):
        p[h] = 0.04
    # Two peaks (discharge windows)
    for h in (8, 9):
        p[h] = 0.40
    for h in (18, 19, 20):
        p[h] = 0.45
    return p


def _problem_params(prices: list[float] | None = None) -> dict[str, Any]:
    return {
        "battery_capacity_kwh": 100.0,
        "max_charge_kw": 25.0,
        "max_discharge_kw": 25.0,
        "soc_init": 0.5,
        "soc_min": 0.1,
        "soc_max": 0.9,
        "eta_charge": 0.95,
        "eta_discharge": 0.95,
        "prices": prices or _two_peak_prices(),
        "dt_hours": 1.0,
    }


def _make_problem(prices: list[float] | None = None) -> OptimizationProblem:
    return OptimizationProblem(
        variables={},
        objectives=[],
        constraints=[],
        parameters=_problem_params(prices),
        time_horizon=24,
        time_step=1.0,
        metadata={"type": "battery_dispatch"},
    )


# ---------------------------------------------------------------------------
# 1. Plugin reports unavailable when pyomo missing
# ---------------------------------------------------------------------------
def test_plugin_unavailable_when_pyomo_missing(monkeypatch):
    """Force pyomo import failure and verify is_available() returns False."""
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("pyomo"):
            raise ImportError("simulated missing pyomo")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Drop any cached pyomo modules so the re-import path is exercised.
    for mod in list(sys.modules):
        if mod.startswith("pyomo"):
            monkeypatch.delitem(sys.modules, mod, raising=False)

    # Reload the plugin module so _try_import_pyomo runs under the patched env.
    import vpp.optimization.solvers.pyomo_plugin as pp

    pp = importlib.reload(pp)
    plugin = pp.PyomoPlugin()
    assert plugin.is_available() is False

    result = plugin.solve(_make_problem())
    assert result.status == OptimizationStatus.FAILED


# ---------------------------------------------------------------------------
# 2. Real solve produces arbitrage profit
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not PyomoPlugin().is_available(),
    reason="pyomo + HiGHS not installed",
)
def test_solve_simple_arbitrage():
    plugin = PyomoPlugin()
    problem = _make_problem()
    result = plugin.solve(problem, timeout_ms=10_000)

    assert result.status == OptimizationStatus.SUCCESS
    # Net revenue (cost is negative).
    assert result.objective_value < 0, f"expected net revenue, got cost={result.objective_value}"

    sol = result.solution
    assert "p_discharge" in sol and "p_charge" in sol and "soc" in sol
    p_dis = sol["p_discharge"]
    p_chg = sol["p_charge"]
    assert len(p_dis) == 24

    # Discharge happens at peak hours (8/9 morning + 18/19/20 evening)
    assert sum(p_dis[h] for h in (8, 9, 18, 19, 20)) > 0
    # Mutual exclusion: no simultaneous charge & discharge
    for c, d in zip(p_chg, p_dis, strict=True):
        assert c * d <= 1e-6, "charge and discharge simultaneously"


# ---------------------------------------------------------------------------
# 3. Pyomo beats rule-based by >=5%
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not PyomoPlugin().is_available(),
    reason="pyomo + HiGHS not installed",
)
def test_solve_beats_rulebased():
    problem = _make_problem()

    pyomo_result = PyomoPlugin().solve(problem, timeout_ms=10_000)
    rules_result = SimpleBatteryDispatchRules().solve(problem)

    assert pyomo_result.status == OptimizationStatus.SUCCESS
    assert rules_result.status == OptimizationStatus.SUCCESS

    # Lower (more negative) cost is better.
    pyomo_cost = pyomo_result.objective_value
    rules_cost = rules_result.objective_value

    # Both should produce some revenue or at least not lose much.
    # Improvement: pyomo is at least 5% better. Use absolute improvement
    # threshold relative to magnitude of rule-based cost.
    improvement = rules_cost - pyomo_cost  # >0 if pyomo better
    rel = improvement / max(1e-6, abs(rules_cost))
    assert improvement > 0, f"pyomo did not beat rules: pyomo={pyomo_cost}, rules={rules_cost}"
    assert rel >= 0.05, (
        f"pyomo improvement {rel:.2%} < 5% (pyomo={pyomo_cost}, rules={rules_cost})"
    )


# ---------------------------------------------------------------------------
# 4. solve_with_fallback returns valid result when plugin unavailable
# ---------------------------------------------------------------------------
def test_fallback_when_solver_unavailable(monkeypatch):
    """If PyomoPlugin reports unavailable, the rule-based fallback must run."""
    # Force PyomoPlugin.is_available -> False so solve_with_fallback won't
    # register it; the engine's registered fallback for 'battery_dispatch'
    # must take over.
    monkeypatch.setattr(PyomoPlugin, "is_available", lambda self: False)

    problem = _make_problem()
    result = solve_with_fallback(problem, timeout_ms=5_000)

    # Engine returns FALLBACK_USED status when fallback runs.
    assert result.status in (
        OptimizationStatus.FALLBACK_USED,
        OptimizationStatus.SUCCESS,
    )
    assert result.solution, "fallback should produce a solution dict"
    assert "p_charge" in result.solution
    assert "p_discharge" in result.solution
    # Sanity: fallback at least doesn't lose money on this two-peak profile.
    assert result.objective_value <= 0.0
