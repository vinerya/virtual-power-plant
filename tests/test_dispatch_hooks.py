"""Tests for the M2 dispatch.py hook API (objective_terms / constraint_builders)."""

from __future__ import annotations

from typing import Any

import pytest

from vpp.optimization.solvers import PyomoPlugin

pyomo_available = PyomoPlugin().is_available()
pytestmark = pytest.mark.skipif(not pyomo_available, reason="pyomo + HiGHS not installed")

if pyomo_available:
    import pyomo.environ as pyo

    from vpp.optimization.formulations.dispatch import build_battery_dispatch_model


def _params(prices: list[float]) -> dict[str, Any]:
    return {
        "battery_capacity_kwh": 100.0,
        "max_charge_kw": 25.0,
        "max_discharge_kw": 25.0,
        "soc_init": 0.5,
        "soc_min": 0.1,
        "soc_max": 0.9,
        "eta_charge": 0.95,
        "eta_discharge": 0.95,
        "prices": prices,
        "dt_hours": 1.0,
    }


def _two_peak_prices() -> list[float]:
    p = [0.10] * 24
    for h in (0, 1, 2, 3, 4, 13, 14):
        p[h] = 0.04
    for h in (8, 9):
        p[h] = 0.40
    for h in (18, 19, 20):
        p[h] = 0.45
    return p


def _solve(model):
    from pyomo.contrib.appsi.solvers.highs import Highs

    solver = Highs()
    solver.config.time_limit = 30.0
    return solver.solve(model)


def test_no_hooks_matches_baseline():
    """Calling with no hooks must reproduce the baseline objective."""
    params = _params(_two_peak_prices())
    m_base = build_battery_dispatch_model(params)
    m_hook = build_battery_dispatch_model(params, objective_terms=[], constraint_builders=[])
    _solve(m_base)
    _solve(m_hook)
    assert pyo.value(m_base.cost) == pytest.approx(pyo.value(m_hook.cost), rel=1e-6, abs=1e-6)


def test_objective_term_and_constraint_hook_swing_penalty():
    """Add an L1 power-swing penalty via constraint_builder + objective_term hooks."""
    params = _params(_two_peak_prices())

    def add_swing_aux(model, _params):
        # Auxiliary nonneg vars u[t] >= |delta[t]| where
        # delta[t] = (p_charge[t]-p_discharge[t]) - (p_charge[t-1]-p_discharge[t-1]).
        T_idx = list(model.T)
        model.swing_u = pyo.Var(model.T, domain=pyo.NonNegativeReals)

        def _ub_pos(mm, t):
            if t == T_idx[0]:
                return pyo.Constraint.Skip
            net_t = mm.p_charge[t] - mm.p_discharge[t]
            net_p = mm.p_charge[t - 1] - mm.p_discharge[t - 1]
            return mm.swing_u[t] >= net_t - net_p

        def _ub_neg(mm, t):
            if t == T_idx[0]:
                return pyo.Constraint.Skip
            net_t = mm.p_charge[t] - mm.p_discharge[t]
            net_p = mm.p_charge[t - 1] - mm.p_discharge[t - 1]
            return mm.swing_u[t] >= net_p - net_t

        model.swing_pos = pyo.Constraint(model.T, rule=_ub_pos)
        model.swing_neg = pyo.Constraint(model.T, rule=_ub_neg)

    SWING_W = 0.05  # penalty weight per kW swing-step

    def swing_penalty(model, _params):
        return SWING_W * sum(model.swing_u[t] for t in model.T)

    # Baseline with no hooks
    m_base = build_battery_dispatch_model(params)
    _solve(m_base)
    base_swing = sum(
        abs(
            (pyo.value(m_base.p_charge[t]) - pyo.value(m_base.p_discharge[t]))
            - (pyo.value(m_base.p_charge[t - 1]) - pyo.value(m_base.p_discharge[t - 1]))
        )
        for t in list(m_base.T)[1:]
    )

    # With penalty
    m_pen = build_battery_dispatch_model(
        params,
        objective_terms=[swing_penalty],
        constraint_builders=[add_swing_aux],
    )
    _solve(m_pen)
    assert hasattr(m_pen, "swing_u"), "constraint_builder must have added swing_u"
    pen_swing = sum(
        abs(
            (pyo.value(m_pen.p_charge[t]) - pyo.value(m_pen.p_discharge[t]))
            - (pyo.value(m_pen.p_charge[t - 1]) - pyo.value(m_pen.p_discharge[t - 1]))
        )
        for t in list(m_pen.T)[1:]
    )

    # Total swing under the penalty should be no greater than the unpenalized baseline,
    # and energy_cost should be no better (greater or equal, since we added a cost term).
    assert pen_swing <= base_swing + 1e-6
    assert pyo.value(m_pen.energy_cost) >= pyo.value(m_base.energy_cost) - 1e-4
