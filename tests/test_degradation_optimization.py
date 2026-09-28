"""Tests for degradation-aware dispatch optimization (Milestone 2)."""

from __future__ import annotations

from typing import Any

import pytest

from vpp.degradation import (
    LFP_PRESET,
    NMC_PRESET,
    RainflowDegradation,
    WearCost,
    add_calendar_aging_bias,
    add_dod_constraints,
    add_wear_cost_term,
)
from vpp.optimization.solvers import PyomoPlugin
from vpp.resources import Battery

pyomo_available = PyomoPlugin().is_available()
pytestmark = pytest.mark.skipif(not pyomo_available, reason="pyomo + HiGHS not installed")

if pyomo_available:
    import pyomo.environ as pyo

    from vpp.optimization.formulations.dispatch import build_battery_dispatch_model


CAP = 100.0
P_MAX = 25.0


def _params(prices: list[float]) -> dict[str, Any]:
    return {
        "battery_capacity_kwh": CAP,
        "max_charge_kw": P_MAX,
        "max_discharge_kw": P_MAX,
        "soc_init": 0.5,
        "soc_min": 0.1,
        "soc_max": 0.9,
        "eta_charge": 0.95,
        "eta_discharge": 0.95,
        "prices": prices,
        "dt_hours": 1.0,
    }


def _solve(model):
    from pyomo.contrib.appsi.solvers.highs import Highs

    solver = Highs()
    solver.config.time_limit = 30.0
    return solver.solve(model)


def _two_peak_prices() -> list[float]:
    p = [0.10] * 24
    for h in (0, 1, 2, 3, 4, 13, 14):
        p[h] = 0.04
    for h in (8, 9):
        p[h] = 0.40
    for h in (18, 19, 20):
        p[h] = 0.45
    return p


def _step_prices() -> list[float]:
    """Single peak per day -- one cheap window, one expensive window."""
    p = [0.10] * 24
    for h in range(0, 6):
        p[h] = 0.02
    for h in (16, 17, 18, 19):
        p[h] = 0.50
    return p


def _soc_trace(model) -> list[float]:
    cap = pyo.value(model.cap)
    soc0 = pyo.value(model.soc_0) / cap
    return [soc0] + [pyo.value(model.soc[t]) / cap for t in model.T]


def _cycling(soc_trace: list[float]) -> float:
    return sum(abs(soc_trace[i] - soc_trace[i - 1]) for i in range(1, len(soc_trace)))


# ---------------------------------------------------------------------------
# A. Throughput wear reduces cycling
# ---------------------------------------------------------------------------


def test_throughput_wear_reduces_cycling():
    params = _params(_two_peak_prices())
    # High-cost wear so the trade-off is unambiguous.
    # Wear cost large enough to bite vs. arbitrage spread (~$0.41/kWh).
    wear = WearCost(
        throughput_cost_per_kwh=0.15,
        cycle_cost_curve={0.2: 100.0, 1.0: 20.0},
    )
    m_base = build_battery_dispatch_model(params)
    m_wear = build_battery_dispatch_model(
        params, objective_terms=[add_wear_cost_term(wear, mode="throughput")]
    )
    _solve(m_base)
    _solve(m_wear)

    cyc_base = _cycling(_soc_trace(m_base))
    cyc_wear = _cycling(_soc_trace(m_wear))

    assert cyc_wear < cyc_base - 1e-4, (
        f"wear cost should reduce cycling: base={cyc_base}, wear={cyc_wear}"
    )
    # Energy revenue (= -energy_cost) is higher in baseline.
    assert pyo.value(m_base.energy_cost) <= pyo.value(m_wear.energy_cost) + 1e-6


# ---------------------------------------------------------------------------
# B. Wear cost equals throughput * lambda for fixed dispatch
# ---------------------------------------------------------------------------


def test_wear_cost_proportional_to_throughput():
    params = _params(_two_peak_prices())
    m_base = build_battery_dispatch_model(params)
    _solve(m_base)

    wear = WearCost(throughput_cost_per_kwh=0.03, cycle_cost_curve={1.0: 1.0})
    dt = pyo.value(m_base.dt)
    realized = wear.throughput_cost_per_kwh * sum(
        (pyo.value(m_base.p_charge[t]) + pyo.value(m_base.p_discharge[t])) * dt for t in m_base.T
    )

    # Now build a fresh model with the wear term, fix p_charge/p_discharge
    # to baseline values, and read off the wear-term contribution.
    m2 = build_battery_dispatch_model(
        params, objective_terms=[add_wear_cost_term(wear, mode="throughput")]
    )
    for t in m2.T:
        m2.p_charge[t].fix(pyo.value(m_base.p_charge[t]))
        m2.p_discharge[t].fix(pyo.value(m_base.p_discharge[t]))
    _solve(m2)
    # Total objective = energy_cost + wear-cost.
    wear_contrib = pyo.value(m2.cost) - pyo.value(m2.energy_cost)
    assert wear_contrib == pytest.approx(realized, rel=1e-4, abs=1e-6)


# ---------------------------------------------------------------------------
# C. dod_pwl prefers shallow cycles
# ---------------------------------------------------------------------------


def test_dod_pwl_prefers_shallow_cycles():
    params = _params(_step_prices())
    # Use a strongly-convex curve: cheap shallow, expensive deep.
    wear = WearCost(
        throughput_cost_per_kwh=0.0,
        cycle_cost_curve={
            0.1: 0.5,
            0.3: 2.0,
            0.6: 10.0,
            1.0: 50.0,
        },
    )
    # throughput-mode comparison
    wear_thru = WearCost(throughput_cost_per_kwh=0.001, cycle_cost_curve=wear.cycle_cost_curve)
    m_thru = build_battery_dispatch_model(
        params, objective_terms=[add_wear_cost_term(wear_thru, mode="throughput")]
    )
    _solve(m_thru)

    m_pwl = build_battery_dispatch_model(
        params,
        constraint_builders=[add_dod_constraints(wear, num_bins=4)],
        objective_terms=[add_wear_cost_term(wear, mode="dod_pwl")],
    )
    _solve(m_pwl)

    soc_thru = _soc_trace(m_thru)
    soc_pwl = _soc_trace(m_pwl)

    # Max single-step |dSOC| is smaller in pwl mode (deep bins discouraged).
    max_step_thru = max(abs(soc_thru[i] - soc_thru[i - 1]) for i in range(1, len(soc_thru)))
    max_step_pwl = max(abs(soc_pwl[i] - soc_pwl[i - 1]) for i in range(1, len(soc_pwl)))
    assert max_step_pwl <= max_step_thru + 1e-6, (
        f"dod_pwl should produce shallower per-step swings: "
        f"thru={max_step_thru}, pwl={max_step_pwl}"
    )


# ---------------------------------------------------------------------------
# C2. Telemetry-consistent wear cost hooks (dod_pwl + calendar, not throughput)
# ---------------------------------------------------------------------------


def test_wear_cost_hooks_for_telemetry_consistency_builds_and_solves():
    """DegradationUpdater.apply_window() (telemetry.py) persists SOH using
    Rainflow (cycle-depth) + Calendar aging -- never ThroughputDegradation.
    Wiring add_wear_cost_term's default mode="throughput" into live dispatch
    would optimize against a different physical model than the one actually
    tracked as SOH. This helper returns the matching (dod_pwl + calendar)
    hook pair so the two stay consistent, and must build/solve without the
    "requires add_dod_constraints()" RuntimeError."""
    from vpp.degradation import wear_cost_hooks_for_telemetry_consistency

    wear = WearCost(
        throughput_cost_per_kwh=0.001,
        cycle_cost_curve={0.1: 0.5, 0.3: 2.0, 0.6: 10.0, 1.0: 50.0},
    )
    objective_terms, constraint_builders = wear_cost_hooks_for_telemetry_consistency(
        wear,
        calendar_weight=0.01,
    )
    params = _params(_two_peak_prices())
    model = build_battery_dispatch_model(
        params,
        constraint_builders=constraint_builders,
        objective_terms=objective_terms,
    )
    result = _solve(model)

    assert str(result.termination_condition) == "TerminationCondition.optimal"
    # Sanity: the model actually has a usable dispatch plan (soc trace
    # includes the initial SOC point, so 24 steps -> 25 values).
    soc = _soc_trace(model)
    assert len(soc) == 25


# ---------------------------------------------------------------------------
# D. Calendar aging biases idle SOC toward 0.5
# ---------------------------------------------------------------------------


def test_calendar_bias_lowers_idle_soc():
    # Flat prices => no arbitrage. Set soc_init high so the optimizer
    # has the option to either stay high or drift to 0.5.
    flat = [0.10] * 24
    params = _params(flat)
    params["soc_init"] = 0.8
    params["terminal_soc"] = 0.1  # allow drift down
    params["soc_min"] = 0.1
    params["soc_max"] = 0.9

    # Without bias: terminal constraint forces some discharge but
    # SOC during the day is unconstrained (any feasible path is optimal).
    # With bias: penalty drives SOC toward 0.5.
    m_bias = build_battery_dispatch_model(
        params, objective_terms=[add_calendar_aging_bias(weight=0.01, soc_neutral=0.5)]
    )
    _solve(m_bias)

    soc = _soc_trace(m_bias)
    cap = pyo.value(m_bias.cap)
    avg_soc = sum(soc) / len(soc)
    # Mean SOC under bias should be much closer to 0.5 than to 0.8.
    assert abs(avg_soc - 0.5) < abs(avg_soc - 0.8), (
        f"calendar bias should pull SOC toward 0.5; avg={avg_soc}"
    )


# ---------------------------------------------------------------------------
# E. SOH updates from realized dispatch
# ---------------------------------------------------------------------------


def test_soh_updates_from_dispatch():
    battery = Battery(
        capacity=100.0,
        current_charge=50.0,
        max_power=25.0,
        nominal_voltage=400.0,
    )
    assert battery.state_of_health == 1.0
    assert battery.cumulative_throughput_kwh == 0.0

    # 4 full cycles between 0.2 and 0.8: triangle wave.
    trace = []
    for _ in range(4):
        trace.extend([0.2, 0.5, 0.8, 0.5])
    trace.append(0.2)
    dt_hours = 1.0

    rf = RainflowDegradation(**LFP_PRESET["rainflow"])
    expected_loss = rf.predict_capacity_loss(trace, dt_hours)

    applied = battery.apply_realized_dispatch(trace, dt_hours, degradation_model=rf)
    assert applied == pytest.approx(expected_loss, rel=1e-9)
    assert battery.state_of_health == pytest.approx(1.0 - expected_loss, rel=1e-9)
    assert battery.cumulative_throughput_kwh > 0


# ---------------------------------------------------------------------------
# F. WearCost.from_preset sanity / NMC > LFP wear cost
# ---------------------------------------------------------------------------


def test_wear_cost_from_preset():
    lfp = WearCost.from_preset(LFP_PRESET, capacity_kwh=100.0, replacement_cost_dollars=20000.0)
    nmc = WearCost.from_preset(NMC_PRESET, capacity_kwh=100.0, replacement_cost_dollars=20000.0)

    # LFP: 6000 cycles * 2 * 100 = 1.2e6 kWh => $20000 / 1.2e6 ~= $0.0167/kWh
    assert lfp.throughput_cost_per_kwh == pytest.approx(20000.0 / 1_200_000.0, rel=1e-9)
    # NMC: 3000 cycles * 2 * 100 = 6e5 kWh => $20000 / 6e5 ~= $0.0333/kWh
    assert nmc.throughput_cost_per_kwh == pytest.approx(20000.0 / 600_000.0, rel=1e-9)
    # NMC has fewer cycles => higher wear cost.
    assert nmc.throughput_cost_per_kwh > lfp.throughput_cost_per_kwh

    # cycle_cost_curve sanity: deeper DoD => higher $/cycle.
    lfp_curve = sorted(lfp.cycle_cost_curve.items())
    for i in range(1, len(lfp_curve)):
        assert lfp_curve[i][1] >= lfp_curve[i - 1][1] - 1e-12
