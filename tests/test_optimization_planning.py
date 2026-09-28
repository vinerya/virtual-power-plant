"""Unit tests for the API-facing planning layer (vpp.optimization.planning)
and the single-interval power-allocation LP."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from vpp.optimization import OptimizationProblem, solve_with_fallback
from vpp.optimization.planning import (
    FleetAsset,
    allocate_power,
    explain_schedule,
    make_forecast_fn,
    plan_schedule,
    run_closed_loop_backtest,
    soh_adjusted_wear_cost,
)
from vpp.optimization.solvers import PowerAllocationPlugin, ProportionalAllocationRules

PRICES = [0.1] * 6 + [0.3] * 6 + [0.1] * 6 + [0.4] * 6


def _batt(rid="b", soc=0.5, soh=1.0, rated=50.0, cap=200.0, chem=None) -> FleetAsset:
    return FleetAsset(
        id=rid,
        name=rid,
        resource_type="battery",
        rated_power_kw=rated,
        capacity_kwh=cap,
        soc=soc,
        soh=soh,
        chemistry=chem,
    )


# ---------------------------------------------------------------------------
# Allocation LP
# ---------------------------------------------------------------------------


def _alloc_problem(resources, target, dt=1.0):
    return OptimizationProblem(
        variables={},
        objectives=[],
        constraints=[],
        parameters={"resources": resources, "target_kw": target, "dt_hours": dt},
        metadata={"type": "power_allocation"},
    )


def test_allocation_lp_merit_order_and_ties():
    res = [
        {"id": "cheap", "lo_kw": 0, "hi_kw": 10, "cost_up": 0.0, "cost_down": 0.0},
        {"id": "a", "lo_kw": -50, "hi_kw": 50, "cost_up": 0.05, "cost_down": 0.05},
        {"id": "b", "lo_kw": -25, "hi_kw": 25, "cost_up": 0.05, "cost_down": 0.05},
    ]
    r = PowerAllocationPlugin().solve(_alloc_problem(res, 40.0))
    alloc = r.solution["allocations"]
    assert alloc["cheap"] == pytest.approx(10.0)
    # Equal-cost resources share the remainder in proportion to headroom.
    assert alloc["a"] == pytest.approx(20.0, abs=1e-6)
    assert alloc["b"] == pytest.approx(10.0, abs=1e-6)
    assert r.solution["shortfall_kw"] == pytest.approx(0.0, abs=1e-6)


def test_allocation_lp_reports_shortfall_instead_of_infeasible():
    res = [{"id": "a", "lo_kw": -5, "hi_kw": 5, "cost_up": 0.0, "cost_down": 0.0}]
    r = PowerAllocationPlugin().solve(_alloc_problem(res, 12.0))
    assert r.solution["allocations"]["a"] == pytest.approx(5.0)
    assert r.solution["shortfall_kw"] == pytest.approx(7.0)


def test_proportional_rules_and_solve_with_fallback_routing():
    res = [
        {"id": "a", "lo_kw": -30, "hi_kw": 30, "cost_up": 1.0, "cost_down": 1.0},
        {"id": "b", "lo_kw": -10, "hi_kw": 10, "cost_up": 0.0, "cost_down": 0.0},
    ]
    rules = ProportionalAllocationRules().solve(_alloc_problem(res, -20.0))
    assert rules.solution["allocations"] == pytest.approx({"a": -15.0, "b": -5.0})

    lp = solve_with_fallback(_alloc_problem(res, 20.0))
    assert lp.fallback_used is False
    assert lp.solution["method"] == "pyomo_highs_allocation"
    forced = solve_with_fallback(_alloc_problem(res, 20.0), force_fallback=True)
    assert forced.fallback_used is True
    assert forced.solution["method"] == "proportional_allocation_rules"


def test_allocate_power_with_no_assets():
    out = allocate_power([], 10.0)
    assert out["status"] == "failed"
    assert out["shortfall_kw"] == 10.0


def test_allocate_power_prefers_high_soc_battery():
    full, empty = _batt("full", soc=0.9), _batt("empty", soc=0.3)
    out = allocate_power([full, empty], 30.0)
    assert out["allocations"]["full"] > out["allocations"]["empty"]
    assert sum(out["allocations"].values()) == pytest.approx(30.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Degradation-aware wear cost
# ---------------------------------------------------------------------------


def test_soh_adjusted_wear_cost_increases_as_pack_ages():
    new = soh_adjusted_wear_cost(_batt(soh=1.0))
    mid = soh_adjusted_wear_cost(_batt(soh=0.9))
    old = soh_adjusted_wear_cost(_batt(soh=0.8))
    assert new.throughput_cost_per_kwh < mid.throughput_cost_per_kwh < old.throughput_cost_per_kwh
    # Past EOL the cost is large but finite (5 % remaining-life floor).
    assert old.throughput_cost_per_kwh == pytest.approx(new.throughput_cost_per_kwh / (0.05 * 0.8))
    # NMC wears faster than LFP.
    nmc = soh_adjusted_wear_cost(_batt(chem="nmc"))
    assert nmc.throughput_cost_per_kwh > new.throughput_cost_per_kwh


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


def test_plan_schedule_value_policy_beats_rules_on_adjusted_cost():
    b = _batt(soc=0.8)
    plan = plan_schedule([b], prices=PRICES, interval_minutes=60)
    assert plan["method"] == "milp_highs"
    exp = explain_schedule(
        {"prices": PRICES, "interval_minutes": 60, "batteries": [b.summary()]}, plan
    )
    costs = {c["name"]: c["total_cost"] for c in exp["counterfactuals"]}
    assert exp["actual"]["total_cost"] <= costs["price_naive"] + 1e-6
    assert exp["actual"]["total_cost"] < costs["no_action"]


def test_plan_schedule_rejects_bad_input():
    with pytest.raises(ValueError):
        plan_schedule([], prices=PRICES, interval_minutes=60)
    with pytest.raises(ValueError):
        plan_schedule([_batt()], prices=PRICES, interval_minutes=60, terminal_soc_policy="bogus")


def test_plan_schedule_large_fleet_uses_admm_with_hold_policy():
    fleet = [_batt(f"b{i}", rated=5.0, cap=20.0) for i in range(11)]
    plan = plan_schedule(fleet, prices=PRICES[:8], interval_minutes=60, timeout_ms=5000)
    assert plan["terminal_soc_policy"] == "hold"
    assert plan["method"] in ("admm_highs", "rule_based_threshold")
    assert len(plan["per_resource"]) == 11


def test_explain_schedule_without_schedule_returns_none():
    assert explain_schedule({"prices": [1.0]}, {"allocations": {"a": 1.0}}) is None
    assert explain_schedule({}, {"charge": [1.0], "discharge": [0.0]}) is None


# ---------------------------------------------------------------------------
# Forecasts and backtest
# ---------------------------------------------------------------------------


def test_persistence_forecast_never_uses_future_prices():
    prices = list(range(48))  # strictly increasing: any peek at the future shows up
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    fn = make_forecast_fn(prices, [], [], start=start, interval_minutes=60, mode="persistence")
    for k in (0, 5, 23, 24, 30, 47):
        fc = fn(start + timedelta(hours=k), 12)["prices"]
        assert len(fc) == 12
        assert max(fc) <= prices[k]
    # One day in, the forecast for the next hour is yesterday's same hour.
    assert fn(start + timedelta(hours=30), 3)["prices"] == [30, 7, 8]


def test_noisy_forecast_is_seeded():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    a = make_forecast_fn(PRICES, [], [], start=start, interval_minutes=60, mode="noisy", seed=3)
    b = make_forecast_fn(PRICES, [], [], start=start, interval_minutes=60, mode="noisy", seed=3)
    assert a(start, 6)["prices"] == b(start, 6)["prices"]
    with pytest.raises(ValueError):
        make_forecast_fn(PRICES, [], [], start=start, interval_minutes=60, mode="oracle")(start, 2)


@pytest.mark.parametrize("policy", ["value", "hold"])
def test_backtest_regret_is_non_negative(policy):
    res = run_closed_loop_backtest(
        _batt(soc=0.5),
        prices=PRICES,
        interval_minutes=60,
        horizon_steps=8,
        forecast_mode="perfect",
        terminal_soc_policy=policy,
    )
    assert res["ticks"] == 24
    assert res["perfect_foresight_status"] == "success"
    assert res["regret"] >= -1e-3
    assert res["perfect_foresight_cost_adjusted"] <= res["rules_cost_adjusted"] + 1e-6
