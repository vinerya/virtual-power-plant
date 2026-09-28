"""API tests for DB-backed dispatch, schedule/MPC, backtest, run history,
explainer and optimization events (/api/v1/optimization, /api/v1/dispatches)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from vpp.db.models import ResourceModel
from vpp.events import EventType, get_event_bus

if TYPE_CHECKING:
    from httpx import AsyncClient

PRESET = Path(__file__).resolve().parents[1] / "src" / "vpp" / "tariffs" / "presets" / "pge_etouc.json"
# Two cheap/expensive cycles: charge at 0.10, discharge at 0.30 / 0.40.
ARB_PRICES = [0.1] * 6 + [0.3] * 6 + [0.1] * 6 + [0.4] * 6


async def _make_resource(client: AsyncClient, headers: dict, rtype: str, rated: float, **meta) -> str:
    resp = await client.post(
        "/api/v1/resources/",
        json={
            "name": f"opt-{rtype}-{uuid.uuid4().hex[:10]}",
            "resource_type": rtype,
            "rated_power": rated,
            "metadata": meta,
        },
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


@pytest.fixture
def captured_events():
    """Record OPTIMIZATION_* events published on the global EventBus."""
    bus = get_event_bus()
    seen: list = []

    async def _cb(event):
        seen.append(event)

    sub = bus.subscribe(
        _cb,
        event_types={
            EventType.OPTIMIZATION_STARTED,
            EventType.OPTIMIZATION_COMPLETED,
            EventType.OPTIMIZATION_FAILED,
        },
    )
    yield seen
    bus.unsubscribe(sub)


def _events_for(events, run_id):
    return [e for e in events if e.data.get("run_id") == run_id]


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_uses_db_resources_and_solver(client, auth_headers, captured_events):
    b1 = await _make_resource(client, auth_headers, "battery", 50.0, capacity_kwh=200, soc=0.8)
    b2 = await _make_resource(client, auth_headers, "battery", 50.0, capacity_kwh=200, soc=0.8)
    s1 = await _make_resource(client, auth_headers, "solar", 20.0)

    resp = await client.post(
        "/api/v1/optimization/dispatch",
        json={"target_power_kw": 60.0, "resource_ids": [b1, b2, s1]},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert body["method"] == "pyomo_highs_allocation"
    assert body["fallback_used"] is False
    assert body["actual_power_kw"] == pytest.approx(60.0, abs=1e-4)
    alloc = {a["resource_id"]: a for a in body["allocations"]}
    assert set(alloc) == {b1, b2, s1}
    # Zero-marginal-cost solar is dispatched first, the remainder is split
    # evenly between the two identical batteries.
    assert alloc[s1]["allocated_power_kw"] == pytest.approx(20.0, abs=1e-4)
    assert alloc[b1]["allocated_power_kw"] == pytest.approx(20.0, abs=0.5)
    assert alloc[b2]["allocated_power_kw"] == pytest.approx(20.0, abs=0.5)
    assert alloc[b1]["soc_source"] == "config"
    assert alloc[s1]["availability_basis"] == "nameplate"

    # Persisted as a run and announced on the EventBus.
    run = (await client.get(f"/api/v1/optimization/runs/{body['run_id']}", headers=auth_headers)).json()
    assert run["problem_type"] == "dispatch"
    assert run["status"] == "success"
    assert run["solver"] == "pyomo_highs_allocation"
    assert run["finished_at"] is not None
    assert set(run["inputs"]["resource_ids"]) == {b1, b2, s1}
    kinds = [e.event_type for e in _events_for(captured_events, body["run_id"])]
    assert kinds == [EventType.OPTIMIZATION_STARTED, EventType.OPTIMIZATION_COMPLETED]
    done = _events_for(captured_events, body["run_id"])[-1]
    assert done.data["method"] == "pyomo_highs_allocation"
    assert done.data["duration_s"] >= 0
    assert done.data["problem_type"] == "dispatch"
    assert done.data["status"] == "success"


@pytest.mark.asyncio
async def test_dispatch_respects_soc_energy_limits(client, auth_headers):
    # 100 kWh at 10 % SOC (min 5 %) holds only ~5 kWh: over a 15 min
    # interval that caps discharge at ~19 kW regardless of the 50 kW rating.
    low = await _make_resource(client, auth_headers, "battery", 50.0, capacity_kwh=100, soc=0.10)
    high = await _make_resource(client, auth_headers, "battery", 50.0, capacity_kwh=100, soc=0.90)
    resp = await client.post(
        "/api/v1/optimization/dispatch",
        json={"target_power_kw": 60.0, "resource_ids": [low, high], "interval_minutes": 15},
        headers=auth_headers,
    )
    body = resp.json()
    alloc = {a["resource_id"]: a for a in body["allocations"]}
    assert alloc[low]["available_power_kw"] == pytest.approx(0.05 * 100 * 0.95 / 0.25, rel=1e-6)
    assert alloc[low]["allocated_power_kw"] <= alloc[low]["available_power_kw"] + 1e-6
    assert alloc[high]["allocated_power_kw"] > alloc[low]["allocated_power_kw"]
    assert body["success"] is True


@pytest.mark.asyncio
async def test_dispatch_negative_target_charges_batteries(client, auth_headers):
    b = await _make_resource(client, auth_headers, "battery", 40.0, capacity_kwh=400, soc=0.5)
    s = await _make_resource(client, auth_headers, "solar", 30.0)
    resp = await client.post(
        "/api/v1/optimization/dispatch",
        json={"target_power_kw": -25.0, "resource_ids": [b, s]},
        headers=auth_headers,
    )
    body = resp.json()
    alloc = {a["resource_id"]: a["allocated_power_kw"] for a in body["allocations"]}
    assert body["success"] is True
    assert alloc[b] == pytest.approx(-25.0, abs=1e-4)
    assert alloc[s] == pytest.approx(0.0, abs=1e-6)  # renewables cannot absorb


@pytest.mark.asyncio
async def test_dispatch_forced_fallback_is_proportional(client, auth_headers):
    b1 = await _make_resource(client, auth_headers, "battery", 30.0, capacity_kwh=400, soc=0.5)
    b2 = await _make_resource(client, auth_headers, "battery", 60.0, capacity_kwh=400, soc=0.5)
    resp = await client.post(
        "/api/v1/optimization/dispatch",
        json={"target_power_kw": 45.0, "resource_ids": [b1, b2], "force_fallback": True},
        headers=auth_headers,
    )
    body = resp.json()
    assert body["fallback_used"] is True
    assert body["method"] == "proportional_allocation_rules"
    alloc = {a["resource_id"]: a["allocated_power_kw"] for a in body["allocations"]}
    assert alloc[b1] == pytest.approx(15.0)
    assert alloc[b2] == pytest.approx(30.0)
    run = (await client.get(f"/api/v1/optimization/runs/{body['run_id']}", headers=auth_headers)).json()
    assert run["status"] == "fallback_used" and run["fallback_used"] is True


@pytest.mark.asyncio
async def test_dispatch_falls_back_when_solver_unavailable(client, auth_headers, monkeypatch):
    from vpp.optimization.solvers.allocation_plugin import PowerAllocationPlugin

    monkeypatch.setattr(PowerAllocationPlugin, "is_available", lambda self: False)
    b = await _make_resource(client, auth_headers, "battery", 30.0, capacity_kwh=400, soc=0.5)
    resp = await client.post(
        "/api/v1/optimization/dispatch",
        json={"target_power_kw": 10.0, "resource_ids": [b]},
        headers=auth_headers,
    )
    body = resp.json()
    assert body["success"] is True
    assert body["fallback_used"] is True
    assert body["method"] == "proportional_allocation_rules"


@pytest.mark.asyncio
async def test_dispatch_target_beyond_capacity_reports_shortfall(client, auth_headers):
    b = await _make_resource(client, auth_headers, "battery", 10.0, capacity_kwh=400, soc=0.5)
    resp = await client.post(
        "/api/v1/optimization/dispatch",
        json={"target_power_kw": 25.0, "resource_ids": [b]},
        headers=auth_headers,
    )
    body = resp.json()
    assert body["success"] is False
    assert body["status"] == "shortfall"
    assert body["actual_power_kw"] == pytest.approx(10.0, abs=1e-4)
    assert body["shortfall_kw"] == pytest.approx(15.0, abs=1e-4)
    assert "shortfall" in body["message"]


@pytest.mark.asyncio
async def test_dispatch_constraints_exclude_and_cap(client, auth_headers):
    b1 = await _make_resource(client, auth_headers, "battery", 50.0, capacity_kwh=400, soc=0.5)
    b2 = await _make_resource(client, auth_headers, "battery", 50.0, capacity_kwh=400, soc=0.5)
    b3 = await _make_resource(client, auth_headers, "battery", 50.0, capacity_kwh=400, soc=0.5)
    resp = await client.post(
        "/api/v1/optimization/dispatch",
        json={
            "target_power_kw": 40.0,
            "resource_ids": [b1, b2, b3],
            "resource_constraints": {b1: {"exclude": True}, b2: {"max_kw": 5.0}},
        },
        headers=auth_headers,
    )
    body = resp.json()
    alloc = {a["resource_id"]: a["allocated_power_kw"] for a in body["allocations"]}
    assert b1 not in alloc
    assert alloc[b2] <= 5.0 + 1e-6
    assert alloc[b2] + alloc[b3] == pytest.approx(40.0, abs=1e-4)


@pytest.mark.asyncio
async def test_dispatch_skips_offline_and_404s_unknown(client, auth_headers, db_session):
    b = await _make_resource(client, auth_headers, "battery", 50.0, capacity_kwh=400, soc=0.5)
    row = await db_session.get(ResourceModel, b)
    row.online = False
    await db_session.commit()
    resp = await client.post(
        "/api/v1/optimization/dispatch",
        json={"target_power_kw": 10.0, "resource_ids": [b]},
        headers=auth_headers,
    )
    body = resp.json()
    assert body["success"] is False and body["allocations"] == []

    resp = await client.post(
        "/api/v1/optimization/dispatch",
        json={"target_power_kw": 10.0, "resource_ids": ["does-not-exist"]},
        headers=auth_headers,
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_dispatch_solver_crash_marks_run_failed(client, auth_headers, captured_events, monkeypatch):
    from vpp.api.routes import optimization as opt_routes

    def _boom(*_a, **_k):
        raise RuntimeError("solver exploded")

    monkeypatch.setattr(opt_routes, "allocate_power", _boom)
    b = await _make_resource(client, auth_headers, "battery", 50.0, capacity_kwh=400, soc=0.5)
    resp = await client.post(
        "/api/v1/optimization/dispatch",
        json={"target_power_kw": 10.0, "resource_ids": [b]},
        headers=auth_headers,
    )
    assert resp.status_code == 500
    failed = [e for e in captured_events if e.event_type == EventType.OPTIMIZATION_FAILED]
    assert failed and "solver exploded" in failed[-1].data["error"]
    run_id = failed[-1].data["run_id"]
    run = (await client.get(f"/api/v1/optimization/runs/{run_id}", headers=auth_headers)).json()
    assert run["status"] == "failed"
    assert "solver exploded" in run["metadata"]["error"]


# ---------------------------------------------------------------------------
# Stochastic (CVaR)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stochastic_runs_cvar_plugin(client, auth_headers):
    resp = await client.post(
        "/api/v1/optimization/stochastic",
        json={
            "num_scenarios": 5,
            "time_horizon_hours": 12,
            "base_prices": [0.1] * 6 + [0.4] * 6,
            "risk_level": 0.1,
            "risk_weight": 0.5,
            "seed": 7,
        },
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["fallback_used"] is False
    assert body["status"] == "success"
    assert body["solver"] == "pyomo_highs_stochastic_cvar"
    sol = body["solution"]
    assert sol["alpha"] == pytest.approx(0.9)
    assert sol["cvar"] >= sol["expected_cost"] - 1e-6
    assert len(sol["charge"]) == 12 and len(sol["discharge"]) == 12
    # Cheap first half, expensive second half: charge early, discharge late.
    assert sum(sol["charge"][:6]) > 0 and sum(sol["discharge"][6:]) > 0
    run = (await client.get(f"/api/v1/optimization/runs/{body['run_id']}", headers=auth_headers)).json()
    assert run["problem_type"] == "stochastic"
    assert run["solver"] == "pyomo_highs_stochastic_cvar"


@pytest.mark.asyncio
async def test_stochastic_is_reproducible_with_seed(client, auth_headers):
    payload = {"num_scenarios": 4, "time_horizon_hours": 6, "seed": 42, "volatility": 0.3}
    r1 = (await client.post("/api/v1/optimization/stochastic", json=payload, headers=auth_headers)).json()
    r2 = (await client.post("/api/v1/optimization/stochastic", json=payload, headers=auth_headers)).json()
    assert r1["solution"]["scenario_costs"] == pytest.approx(r2["solution"]["scenario_costs"])


@pytest.mark.asyncio
async def test_stochastic_forced_fallback_and_validation(client, auth_headers):
    resp = await client.post(
        "/api/v1/optimization/stochastic",
        json={"num_scenarios": 3, "time_horizon_hours": 6, "force_fallback": True},
        headers=auth_headers,
    )
    body = resp.json()
    assert body["fallback_used"] is True
    assert body["status"] == "fallback_used"

    too_big = await client.post(
        "/api/v1/optimization/stochastic",
        json={"num_scenarios": 1000, "time_horizon_hours": 168},
        headers=auth_headers,
    )
    assert too_big.status_code == 422
    bad_len = await client.post(
        "/api/v1/optimization/stochastic",
        json={"time_horizon_hours": 6, "base_prices": [1.0, 2.0]},
        headers=auth_headers,
    )
    assert bad_len.status_code == 422


# ---------------------------------------------------------------------------
# Schedule (MPC)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_schedule_single_battery_arbitrage_and_explainer(client, auth_headers, captured_events):
    b = await _make_resource(client, auth_headers, "battery", 50.0, capacity_kwh=200, soc=0.5)
    resp = await client.post(
        "/api/v1/optimization/schedule",
        json={"resource_ids": [b], "prices": ARB_PRICES, "interval_minutes": 60},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["method"] == "milp_highs"
    assert body["fallback_used"] is False
    assert len(body["charge"]) == len(body["discharge"]) == len(body["power"]) == 24
    assert body["energy_cost"] < 0  # arbitrage revenue
    assert sum(body["charge"][:6]) > 0 and sum(body["discharge"][18:]) > 0
    assert body["power"][0] == pytest.approx(body["charge"][0] - body["discharge"][0])
    soc = body["per_resource"][b]["soc"]
    assert all(0.05 - 1e-6 <= s <= 0.95 + 1e-6 for s in soc)
    kinds = [e.event_type for e in _events_for(captured_events, body["run_id"])]
    assert kinds == [EventType.OPTIMIZATION_STARTED, EventType.OPTIMIZATION_COMPLETED]

    # Explainer via the UI's /dispatches alias.
    exp = await client.get(f"/api/v1/dispatches/{body['run_id']}/explain", headers=auth_headers)
    assert exp.status_code == 200, exp.text
    e = exp.json()
    assert e["run_id"] == body["run_id"]
    names = [c["name"] for c in e["counterfactuals"]]
    assert names == ["no_action", "price_naive"]
    assert len(e["actual"]["per_step"]) == 24
    no_action = e["counterfactuals"][0]["total_cost"]
    assert e["actual"]["total_cost"] < no_action
    # The schedule optimises exactly the adjusted cost the explainer reports,
    # so the rule baseline can't beat it.
    assert e["actual"]["total_cost"] <= e["counterfactuals"][1]["total_cost"] + 1e-6
    assert e["binding_constraints"], "full charge / discharge blocks should bind"
    assert "saves" in e["rationale"]
    alias = await client.get(f"/api/v1/optimization/explain/{body['run_id']}", headers=auth_headers)
    assert alias.json() == e


@pytest.mark.asyncio
async def test_schedule_hold_policy_returns_to_initial_soc(client, auth_headers):
    b = await _make_resource(client, auth_headers, "battery", 50.0, capacity_kwh=200, soc=0.5)
    body = (await client.post(
        "/api/v1/optimization/schedule",
        json={"resource_ids": [b], "prices": ARB_PRICES, "terminal_soc_policy": "hold"},
        headers=auth_headers,
    )).json()
    assert body["terminal_soc_policy"] == "hold"
    assert body["per_resource"][b]["soc"][-1] >= 0.5 - 1e-4


@pytest.mark.asyncio
async def test_schedule_fleet_and_mpc_alias(client, auth_headers):
    b1 = await _make_resource(client, auth_headers, "battery", 50.0, capacity_kwh=200, soc=0.5)
    b2 = await _make_resource(client, auth_headers, "battery", 25.0, capacity_kwh=100, soc=0.3)
    resp = await client.post(
        "/api/v1/optimization/mpc",
        json={"resource_ids": [b1, b2], "prices": ARB_PRICES, "feeder_max_export_kw": 60.0},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["method"] == "milp_highs"
    assert set(body["per_resource"]) == {b1, b2}
    assert max(body["discharge"]) <= 60.0 + 1e-4  # feeder export cap honoured
    for t in range(24):
        assert body["charge"][t] == pytest.approx(
            body["per_resource"][b1]["charge"][t] + body["per_resource"][b2]["charge"][t], abs=1e-3
        )


@pytest.mark.asyncio
async def test_schedule_falls_back_without_solver(client, auth_headers, monkeypatch):
    from vpp.optimization import mpc

    monkeypatch.setattr(mpc, "_try_import_pyomo", lambda: (None, None))
    b = await _make_resource(client, auth_headers, "battery", 50.0, capacity_kwh=200, soc=0.5)
    body = (await client.post(
        "/api/v1/optimization/schedule",
        json={"resource_ids": [b], "prices": ARB_PRICES},
        headers=auth_headers,
    )).json()
    assert body["fallback_used"] is True
    assert body["method"] == "rule_based_threshold"
    assert body["fallback_reason"] == "no_pyomo"
    assert body["status"] == "fallback_used"


@pytest.mark.asyncio
async def test_schedule_with_tariff_id(client, auth_headers):
    with open(PRESET) as f:
        urdb = json.load(f)
    t = await client.post(
        "/api/v1/tariffs",
        json={"name": f"opt-tariff-{uuid.uuid4().hex[:6]}", "utility": "PG&E-opt", "urdb_json": urdb},
        headers=auth_headers,
    )
    assert t.status_code == 201, t.text
    tariff_id = t.json()["id"]
    b = await _make_resource(client, auth_headers, "battery", 5.0, capacity_kwh=13.5, soc=0.5)
    resp = await client.post(
        "/api/v1/optimization/schedule",
        json={
            "resource_ids": [b],
            "tariff_id": tariff_id,
            "horizon_start": "2024-07-15T00:00:00Z",
            "horizon_hours": 24,
            "interval_minutes": 60,
            "load_kw": [1.5] * 24,
        },
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["tariff_id"] == tariff_id
    assert body["method"] == "milp_highs"
    assert len(body["prices"]) == 24
    assert max(body["prices"]) > min(body["prices"])  # TOU peak vs off-peak
    run = (await client.get(f"/api/v1/optimization/runs/{body['run_id']}", headers=auth_headers)).json()
    assert run["inputs"]["tariff_id"] == tariff_id

    missing = await client.post(
        "/api/v1/optimization/schedule",
        json={"resource_ids": [b], "tariff_id": "nope"},
        headers=auth_headers,
    )
    assert missing.status_code == 404


@pytest.mark.asyncio
async def test_schedule_degradation_aware_uses_db_soh(client, auth_headers, db_session):
    healthy = await _make_resource(client, auth_headers, "battery", 50.0, capacity_kwh=200, soc=0.5)
    worn = await _make_resource(client, auth_headers, "battery", 50.0, capacity_kwh=200, soc=0.5)
    row = await db_session.get(ResourceModel, worn)
    row.state_of_health = 0.82
    row.chemistry = "nmc"
    await db_session.commit()

    async def _run(rid):
        r = await client.post(
            "/api/v1/optimization/schedule",
            json={"resource_ids": [rid], "prices": ARB_PRICES, "degradation_aware": True},
            headers=auth_headers,
        )
        assert r.status_code == 200, r.text
        return r.json()

    h, w = await _run(healthy), await _run(worn)
    assert h["wear_cost"] > 0
    run_h = (await client.get(f"/api/v1/optimization/runs/{h['run_id']}", headers=auth_headers)).json()
    run_w = (await client.get(f"/api/v1/optimization/runs/{w['run_id']}", headers=auth_headers)).json()
    assert run_w["metadata"]["wear_cost_per_kwh"][worn] > run_h["metadata"]["wear_cost_per_kwh"][healthy]
    assert w["resources"][0]["soh"] == pytest.approx(0.82)
    # Capacity fade: the worn pack schedules against 82 % of nameplate.
    assert w["resources"][0]["capacity_kwh"] == pytest.approx(200 * 0.82)
    # A costlier, smaller pack cycles no more energy than the healthy one.
    assert sum(w["charge"]) <= sum(h["charge"]) + 1e-6


@pytest.mark.asyncio
async def test_schedule_validation(client, auth_headers):
    url = "/api/v1/optimization/schedule"
    both = await client.post(url, json={"prices": [1.0], "tariff_id": "x"}, headers=auth_headers)
    neither = await client.post(url, json={}, headers=auth_headers)
    too_long = await client.post(url, json={"prices": [1.0] * 289}, headers=auth_headers)
    bad_interval = await client.post(url, json={"prices": [1.0], "interval_minutes": 7}, headers=auth_headers)
    bad_load = await client.post(url, json={"prices": [1.0, 2.0], "load_kw": [1.0]}, headers=auth_headers)
    for r in (both, neither, too_long, bad_interval, bad_load):
        assert r.status_code == 422, r.text


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backtest_inline_battery(client, auth_headers):
    resp = await client.post(
        "/api/v1/optimization/backtest",
        json={
            "battery": {"capacity_kwh": 200, "max_power_kw": 50, "soc_init": 0.5},
            "prices": ARB_PRICES,
            "interval_minutes": 60,
            "horizon_steps": 12,
            "forecast_mode": "perfect",
        },
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ticks"] == 24
    assert body["fallback_count"] == 0
    assert body["perfect_foresight_status"] == "success"
    # The offline benchmark is a lower bound on the adjusted cost.
    assert body["regret"] >= -1e-3
    assert body["realized_cost_adjusted"] < body["no_action_cost"]
    assert len(body["soc"]) == 24
    # Explainable like any other schedule run.
    exp = await client.get(f"/api/v1/optimization/runs/{body['run_id']}/explain", headers=auth_headers)
    assert exp.status_code == 200
    assert len(exp.json()["actual"]["per_step"]) == 24


@pytest.mark.asyncio
async def test_backtest_with_resource_and_persistence(client, auth_headers):
    b = await _make_resource(client, auth_headers, "battery", 50.0, capacity_kwh=200, soc=0.5)
    resp = await client.post(
        "/api/v1/optimization/backtest",
        json={
            "resource_id": b,
            "prices": ARB_PRICES * 2,
            "interval_minutes": 60,
            "horizon_steps": 12,
            "forecast_mode": "persistence",
            "compare_offline": False,
        },
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ticks"] == 48
    assert body["perfect_foresight_status"] == "skipped"
    assert body["regret"] is None
    runs = (await client.get(
        "/api/v1/optimization/runs", params={"resource_id": b}, headers=auth_headers
    )).json()
    assert [r["id"] for r in runs] == [body["run_id"]]
    assert runs[0]["resource_id"] == b


@pytest.mark.asyncio
async def test_backtest_validation(client, auth_headers):
    url = "/api/v1/optimization/backtest"
    battery = {"capacity_kwh": 10, "max_power_kw": 5}
    both = await client.post(url, json={"resource_id": "x", "battery": battery, "prices": [1, 2]}, headers=auth_headers)
    neither = await client.post(url, json={"prices": [1, 2]}, headers=auth_headers)
    too_much = await client.post(
        url, json={"battery": battery, "prices": [1.0] * 672, "horizon_steps": 96}, headers=auth_headers
    )
    for r in (both, neither, too_much):
        assert r.status_code == 422, r.text
    missing = await client.post(url, json={"resource_id": "nope", "prices": [1, 2]}, headers=auth_headers)
    assert missing.status_code == 404


# ---------------------------------------------------------------------------
# Run history
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_history_filters_and_aliases(client, auth_headers):
    b = await _make_resource(client, auth_headers, "battery", 50.0, capacity_kwh=200, soc=0.5)
    sched = (await client.post(
        "/api/v1/optimization/schedule",
        json={"resource_ids": [b], "prices": ARB_PRICES[:6]},
        headers=auth_headers,
    )).json()
    disp = (await client.post(
        "/api/v1/optimization/dispatch",
        json={"target_power_kw": 5.0, "resource_ids": [b]},
        headers=auth_headers,
    )).json()

    by_res = (await client.get(
        "/api/v1/optimization/history", params={"resource_id": b, "limit": 10}, headers=auth_headers
    )).json()
    assert [r["id"] for r in by_res] == [disp["run_id"], sched["run_id"]]  # newest first
    first = by_res[0]
    for key in ("id", "problem_type", "status", "created_at", "solver", "inputs", "solution", "metadata"):
        assert key in first

    typed = (await client.get(
        "/api/v1/optimization/runs",
        params={"resource_id": b, "problem_type": "schedule"},
        headers=auth_headers,
    )).json()
    assert [r["id"] for r in typed] == [sched["run_id"]]

    lean = (await client.get(
        "/api/v1/optimization/runs",
        params={"resource_id": b, "include_details": "false"},
        headers=auth_headers,
    )).json()
    assert lean[0].get("solution") is None

    paged = (await client.get(
        "/api/v1/optimization/history", params={"resource_id": b, "limit": 1, "offset": 1}, headers=auth_headers
    )).json()
    assert [r["id"] for r in paged] == [sched["run_id"]]

    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    assert (await client.get(
        "/api/v1/optimization/history", params={"resource_id": b, "start": future}, headers=auth_headers
    )).json() == []
    assert len((await client.get(
        "/api/v1/optimization/history", params={"resource_id": b, "start": past}, headers=auth_headers
    )).json()) == 2
    assert (await client.get(
        "/api/v1/optimization/history", params={"resource_id": b, "end": past}, headers=auth_headers
    )).json() == []

    alias = (await client.get("/api/v1/dispatches", params={"resource_id": b}, headers=auth_headers)).json()
    assert [r["id"] for r in alias] == [disp["run_id"], sched["run_id"]]
    one = await client.get(f"/api/v1/dispatches/{disp['run_id']}", headers=auth_headers)
    assert one.status_code == 200 and one.json()["problem_type"] == "dispatch"

    # Single-interval dispatch has no per-step schedule to explain.
    no_expl = await client.get(f"/api/v1/dispatches/{disp['run_id']}/explain", headers=auth_headers)
    assert no_expl.status_code == 204


@pytest.mark.asyncio
async def test_unknown_run_404s(client, auth_headers):
    for url in (
        "/api/v1/optimization/runs/nope",
        "/api/v1/optimization/runs/nope/explain",
        "/api/v1/dispatches/nope",
        "/api/v1/dispatches/nope/explain",
    ):
        assert (await client.get(url, headers=auth_headers)).status_code == 404, url


@pytest.mark.asyncio
async def test_run_endpoints_require_auth(client):
    for url in ("/api/v1/optimization/runs", "/api/v1/dispatches"):
        assert (await client.get(url)).status_code in (401, 403)
    resp = await client.post("/api/v1/optimization/schedule", json={"prices": [1.0]})
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_realtime_and_distributed_are_persisted(client, auth_headers, captured_events):
    rt = (await client.post(
        "/api/v1/optimization/realtime",
        json={"grid_frequency_hz": 49.9, "active_power_demand_kw": 50.0},
        headers=auth_headers,
    )).json()
    assert rt["run_id"]
    dist = (await client.post(
        "/api/v1/optimization/distributed",
        json={"sites": [{"site_id": "a"}, {"site_id": "b"}], "target_power_kw": 10.0},
        headers=auth_headers,
    )).json()
    for run_id, ptype in ((rt["run_id"], "realtime"), (dist["run_id"], "distributed")):
        run = (await client.get(f"/api/v1/optimization/runs/{run_id}", headers=auth_headers)).json()
        assert run["problem_type"] == ptype
        assert run["status"] != "running"
        assert len(_events_for(captured_events, run_id)) == 2
