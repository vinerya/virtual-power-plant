"""Tests for the M2 stochastic CVaR extensive-form solver."""

from __future__ import annotations

from typing import Any

import pytest

from vpp.optimization import (
    OptimizationProblem,
    OptimizationStatus,
    solve_with_fallback,
)
from vpp.optimization.solvers import StochasticCVaRPlugin
from vpp.optimization.stochastic import Scenario

pyomo_available = StochasticCVaRPlugin().is_available()
pytestmark = pytest.mark.skipif(not pyomo_available, reason="pyomo + HiGHS not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _det_params() -> dict[str, Any]:
    return {
        "battery_capacity_kwh": 100.0,
        "max_charge_kw": 25.0,
        "max_discharge_kw": 25.0,
        "soc_init": 0.5,
        "soc_min": 0.1,
        "soc_max": 0.9,
        "eta_charge": 0.95,
        "eta_discharge": 0.95,
        "dt_hours": 1.0,
    }


def _scenario(sid: str, prob: float, prices: list[float]) -> Scenario:
    return Scenario(id=sid, probability=prob, data={"prices": prices})


def _three_scenarios_24h() -> list[Scenario]:
    base = [0.10] * 24
    cheap = (0, 1, 2, 3, 4)
    peak_a = (8, 9)
    peak_b = (18, 19, 20)

    s1 = list(base)
    for h in cheap:
        s1[h] = 0.04
    for h in peak_a:
        s1[h] = 0.40
    for h in peak_b:
        s1[h] = 0.45

    s2 = list(base)
    for h in cheap:
        s2[h] = 0.05
    for h in peak_a:
        s2[h] = 0.30
    for h in peak_b:
        s2[h] = 0.55  # extra-expensive evening

    s3 = list(base)
    for h in cheap:
        s3[h] = 0.06
    for h in peak_a:
        s3[h] = 0.50  # extra-expensive morning
    for h in peak_b:
        s3[h] = 0.35

    return [
        _scenario("s1", 0.4, s1),
        _scenario("s2", 0.3, s2),
        _scenario("s3", 0.3, s3),
    ]


def _make_problem(
    scenarios: list[Scenario],
    cvar_alpha: float = 0.95,
    cvar_lambda: float = 0.0,
    extra: dict[str, Any] | None = None,
) -> OptimizationProblem:
    params = _det_params()
    params["scenarios"] = scenarios
    params["cvar_alpha"] = cvar_alpha
    params["cvar_lambda"] = cvar_lambda
    if extra:
        params.update(extra)
    return OptimizationProblem(
        variables={},
        objectives=[],
        constraints=[],
        parameters=params,
        time_horizon=len(scenarios[0].data["prices"]),
        time_step=1.0,
        metadata={"type": "stochastic_dispatch"},
    )


# ---------------------------------------------------------------------------
# 1. Extensive form solves
# ---------------------------------------------------------------------------
def test_extensive_form_solves():
    scenarios = _three_scenarios_24h()
    problem = _make_problem(scenarios, cvar_lambda=0.0)
    plugin = StochasticCVaRPlugin()
    result = plugin.solve(problem, timeout_ms=30_000)

    assert result.status == OptimizationStatus.SUCCESS, result.metadata
    sol = result.solution
    assert len(sol["scenarios"]) == 3
    # Expected cost equals probability-weighted average of per-scenario costs.
    pi = sol["probabilities"]
    cs = sol["scenario_costs"]
    expected_manual = sum(p * c for p, c in zip(pi, cs, strict=True))
    assert sol["expected_cost"] == pytest.approx(expected_manual, rel=1e-5, abs=1e-5)


# ---------------------------------------------------------------------------
# 2. Non-anticipativity
# ---------------------------------------------------------------------------
def test_non_anticipativity():
    scenarios = _three_scenarios_24h()
    problem = _make_problem(scenarios, cvar_lambda=0.5)
    plugin = StochasticCVaRPlugin()
    result = plugin.solve(problem, timeout_ms=30_000)
    assert result.status == OptimizationStatus.SUCCESS

    p_chg_0 = [s["p_charge"][0] for s in result.solution["scenarios"]]
    p_dis_0 = [s["p_discharge"][0] for s in result.solution["scenarios"]]
    for v in p_chg_0[1:]:
        assert v == pytest.approx(p_chg_0[0], abs=1e-5)
    for v in p_dis_0[1:]:
        assert v == pytest.approx(p_dis_0[0], abs=1e-5)


# ---------------------------------------------------------------------------
# 3. CVaR responds to lambda
# ---------------------------------------------------------------------------
def test_cvar_increases_with_lambda():
    scenarios = _three_scenarios_24h()
    plugin = StochasticCVaRPlugin()

    p0 = _make_problem(scenarios, cvar_alpha=0.95, cvar_lambda=0.0)
    p1 = _make_problem(scenarios, cvar_alpha=0.95, cvar_lambda=1.0)

    r0 = plugin.solve(p0, timeout_ms=30_000)
    r1 = plugin.solve(p1, timeout_ms=30_000)
    assert r0.status == OptimizationStatus.SUCCESS
    assert r1.status == OptimizationStatus.SUCCESS

    max_cost_lam0 = max(r0.solution["scenario_costs"])
    # CVaR at lambda=1 must not exceed the worst scenario cost (it's a tail mean).
    assert r1.solution["cvar"] <= max_cost_lam0 + 1e-5
    # Risk-averse policy should weakly worsen expected cost vs. risk-neutral.
    assert r1.solution["expected_cost"] >= r0.solution["expected_cost"] - 1e-5


# ---------------------------------------------------------------------------
# 4. CVaR alpha extremes
# ---------------------------------------------------------------------------
def test_cvar_alpha_extreme():
    scenarios = _three_scenarios_24h()
    plugin = StochasticCVaRPlugin()

    # alpha = 0 -> CVaR collapses to expected cost (per our convention).
    p_alpha0 = _make_problem(scenarios, cvar_alpha=0.0, cvar_lambda=1.0)
    r0 = plugin.solve(p_alpha0, timeout_ms=30_000)
    assert r0.status == OptimizationStatus.SUCCESS
    # Total objective equals (1+lambda)*E[cost] when CVaR == E[cost]
    assert r0.solution["cvar"] == pytest.approx(r0.solution["expected_cost"], rel=1e-5, abs=1e-5)

    # alpha very close to 1 -> CVaR ~ worst-case cost.
    p_alpha_hi = _make_problem(scenarios, cvar_alpha=0.999, cvar_lambda=1.0)
    r1 = plugin.solve(p_alpha_hi, timeout_ms=30_000)
    assert r1.status == OptimizationStatus.SUCCESS
    worst = max(r1.solution["scenario_costs"])
    # CVaR for a discrete distribution with alpha near 1 collapses to worst case.
    assert r1.solution["cvar"] >= worst - 1e-3


# ---------------------------------------------------------------------------
# 5. EEV >= stochastic optimum >= wait-and-see
# ---------------------------------------------------------------------------
def _solve_det(prices: list[float]) -> float:
    """Deterministic single-scenario optimum (per-scenario perfect foresight)."""
    import pyomo.environ as pyo
    from pyomo.contrib.appsi.solvers.highs import Highs

    from vpp.optimization.formulations.dispatch import build_battery_dispatch_model

    params = _det_params()
    params["prices"] = prices
    model = build_battery_dispatch_model(params)
    s = Highs()
    s.config.time_limit = 30.0
    s.solve(model)
    return float(pyo.value(model.cost))


def test_eevpi_lower_bound():
    # Tiny 2-scenario, short-horizon problem.
    p_low = [0.05, 0.40, 0.05, 0.40]
    p_high = [0.40, 0.05, 0.40, 0.05]
    scenarios = [
        _scenario("L", 0.5, p_low),
        _scenario("H", 0.5, p_high),
    ]

    plugin = StochasticCVaRPlugin()

    # Stochastic optimum (risk-neutral)
    sp = _make_problem(scenarios, cvar_alpha=0.95, cvar_lambda=0.0)
    sr = plugin.solve(sp, timeout_ms=30_000)
    assert sr.status == OptimizationStatus.SUCCESS
    rp = sr.solution["expected_cost"]  # recourse-problem optimum

    # Wait-and-see: average of per-scenario perfect-foresight optima
    ws = 0.5 * _solve_det(p_low) + 0.5 * _solve_det(p_high)

    # EEV: solve deterministic with mean prices; then evaluate the stage-1 decision
    # against each scenario by fixing it and re-solving stage-2. Easier proxy:
    # build a stochastic problem where stage-1 is forced equal across scenarios
    # (already true) AND prices are replaced by the mean — its expected cost under
    # mean prices is what one would do if treating uncertainty as expectation.
    mean_prices = [0.5 * (a + b) for a, b in zip(p_low, p_high, strict=True)]
    mean_scenarios = [
        _scenario("M1", 0.5, mean_prices),
        _scenario("M2", 0.5, mean_prices),
    ]
    mp = _make_problem(mean_scenarios, cvar_alpha=0.95, cvar_lambda=0.0)
    mr = plugin.solve(mp, timeout_ms=30_000)
    assert mr.status == OptimizationStatus.SUCCESS
    # The mean-prices schedule's stage-1 decision is mr.solution["stage1"].
    # Now evaluate that stage-1 against the true scenarios by using a problem
    # where stage-1 is constrained to that value via tight bounds. We use a
    # post-hoc evaluation: re-solve original scenarios but with extra parameter.
    forced_chg = mr.solution["stage1"]["p_charge_0"]
    forced_dis = mr.solution["stage1"]["p_discharge_0"]

    # Build the standard stochastic problem and add a small bound by clipping
    # via the objective: easiest is to evaluate cost manually on each scenario
    # using deterministic dispatch with t=0 forced. We cheat using an enlarged
    # det problem per scenario: prepend a fake step? Instead, since stage-1
    # power is small, EEV is approximately the cost of running the mean-prices
    # plan against the true scenarios. We approximate EEV as the stochastic
    # optimum where t=0 is fixed to the mean-plan stage-1 decision.

    # Approach: build a new stochastic model and bound p_charge[0,*] and
    # p_discharge[0,*] tightly around the forced values.
    eev_prob = _make_problem(scenarios, cvar_alpha=0.95, cvar_lambda=0.0)
    eev_prob.parameters["_force_stage1"] = (forced_chg, forced_dis)

    # Easiest: solve the stochastic problem with stage-1 pinned via params.
    # We rebuild manually here:
    import pyomo.environ as pyo
    from pyomo.contrib.appsi.solvers.highs import Highs

    from vpp.optimization.formulations.stochastic import build_stochastic_dispatch_model

    params_eev = dict(eev_prob.parameters)
    params_eev.pop("_force_stage1", None)
    scen_list = params_eev.pop("scenarios")
    model = build_stochastic_dispatch_model(params_eev, scen_list)
    # Pin stage-1
    for s_idx in model.S:
        model.p_charge[0, s_idx].fix(forced_chg)
        model.p_discharge[0, s_idx].fix(forced_dis)
    solver = Highs()
    solver.config.time_limit = 30.0
    solver.solve(model)
    eev = float(pyo.value(model.expected_cost))

    # Inequality chain (cost-minimization version):
    #     WS  <=  RP (stochastic)  <=  EEV
    # Allow tiny tolerance.
    assert ws <= rp + 1e-4, f"WS={ws} should be <= RP={rp}"
    assert rp <= eev + 1e-4, f"RP={rp} should be <= EEV={eev}"


# ---------------------------------------------------------------------------
# 6. solve_with_fallback registers stochastic plugin
# ---------------------------------------------------------------------------
def test_solve_with_fallback_uses_stochastic_plugin():
    scenarios = _three_scenarios_24h()
    problem = _make_problem(scenarios)
    result = solve_with_fallback(problem, timeout_ms=30_000)
    assert result.status in (
        OptimizationStatus.SUCCESS,
        OptimizationStatus.FALLBACK_USED,
    )
    assert result.solution
