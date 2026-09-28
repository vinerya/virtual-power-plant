"""Optimization, scheduling and dispatch-history routes.

Every solve endpoint:

* reads the fleet from the database (the source of truth — never from the
  in-memory ``VirtualPowerPlant`` singleton),
* runs the Pyomo/HiGHS solver path through the plugin framework, falling
  back to rule-based heuristics when the solver is unavailable or fails,
  and reports which method ran,
* persists the run in ``optimization_runs`` and publishes
  ``OPTIMIZATION_STARTED`` / ``OPTIMIZATION_COMPLETED`` /
  ``OPTIMIZATION_FAILED`` on the EventBus (forwarded to the
  ``optimization_events`` WebSocket channel).

Solves run in a worker thread so a long MILP never blocks the event loop.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: TC002 - resolved at runtime by FastAPI
from starlette.concurrency import run_in_threadpool

from vpp.api.optimization_support import (
    finish_run,
    load_fleet_assets,
    run_to_dict,
    start_run,
    unpack_solution,
)
from vpp.auth.security import get_current_user
from vpp.db.engine import get_db
from vpp.db.models import UserModel  # noqa: TC001 - resolved at runtime by FastAPI
from vpp.db.repositories import OptimizationRepository, TariffRepository
from vpp.optimization.planning import (
    FleetAsset,
    allocate_power,
    explain_schedule,
    plan_schedule,
    run_closed_loop_backtest,
)
from vpp.schemas.optimization import (
    BacktestRequest,
    BacktestResponse,
    DispatchRequest,
    DispatchResponse,
    DistributedRequest,
    ExplainerResponse,
    OptimizationResponse,
    OptimizationRunRead,
    RealTimeRequest,
    ResourceAllocation,
    ScheduleRequest,
    ScheduleResponse,
    StochasticRequest,
)

if TYPE_CHECKING:
    from collections.abc import Callable

router = APIRouter(prefix="/api/v1/optimization", tags=["Optimization"])
# Dispatch-history aliases consumed by the operator UI (web/lib/api/*.ts).
dispatches_router = APIRouter(prefix="/api/v1/dispatches", tags=["Optimization"])

# Upper bound on resource-steps (batteries x horizon steps) for one schedule
# MILP; larger requests are rejected rather than left to time out.
MAX_SCHEDULE_RESOURCE_STEPS = 20 * 288
_SHORTFALL_TOL_KW = 1e-3
_MAX_PERSISTED_SCENARIOS = 20


def _status_str(s: Any) -> str:
    return s.value if hasattr(s, "value") else str(s)


def _finite_or_zero(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f if math.isfinite(f) else 0.0


async def _run_tracked(
    session: AsyncSession,
    problem_type: str,
    inputs: dict[str, Any],
    fn: Callable[[], Any],
) -> tuple[str, Any, float]:
    """Persist a run, execute ``fn`` in a worker thread and time it.

    On exception the run is marked failed (and the failure published) before
    a 500 is raised. Returns ``(run_id, result, solve_ms)``; the caller
    finishes the run with :func:`finish_run`.
    """
    run = await start_run(session, problem_type, inputs)
    t0 = time.perf_counter()
    try:
        result = await run_in_threadpool(fn)
    except Exception as exc:
        solve_ms = (time.perf_counter() - t0) * 1000
        await finish_run(
            session,
            run.id,
            problem_type=problem_type,
            status="failed",
            solve_time_ms=solve_ms,
            metadata={"error": f"{type(exc).__name__}: {exc}"},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"optimization failed: {exc}",
        ) from exc
    return run.id, result, (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------------------
# Single-interval dispatch
# ---------------------------------------------------------------------------


@router.post("/dispatch", response_model=DispatchResponse)
async def dispatch(
    body: DispatchRequest,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """Split a site power target across the online resources in the database.

    Solved as an LP (Pyomo + HiGHS) that respects each battery's power and
    energy limits over the interval, prices battery throughput with an
    SOH-aware wear cost, and prefers renewables; falls back to a
    headroom-proportional split when the solver is unavailable or fails.
    """
    assets, missing = await load_fleet_assets(session, body.resource_ids)
    if missing:
        raise HTTPException(
            status_code=404, detail={"message": "unknown resource ids", "missing": missing}
        )

    dt_hours = body.interval_minutes / 60.0
    inputs = {
        "target_power_kw": body.target_power_kw,
        "interval_minutes": body.interval_minutes,
        "resource_ids": [a.id for a in assets],
        "resource_constraints": body.resource_constraints,
        "wear_cost": body.wear_cost,
        "replacement_cost_per_kwh": body.replacement_cost_per_kwh,
        "force_fallback": body.force_fallback,
        "resources": [a.summary() for a in assets],
    }
    run_id, result, solve_ms = await _run_tracked(
        session,
        "dispatch",
        inputs,
        lambda: allocate_power(
            assets,
            body.target_power_kw,
            dt_hours=dt_hours,
            constraints=body.resource_constraints,
            timeout_ms=body.timeout_ms,
            force_fallback=body.force_fallback,
            wear_cost=body.wear_cost,
            replacement_cost_per_kwh=body.replacement_cost_per_kwh,
        ),
    )

    by_id = {a.id: a for a in assets}
    allocations: list[ResourceAllocation] = []
    for rid, b in result["bounds"].items():
        a: FleetAsset = by_id[rid]
        allocations.append(
            ResourceAllocation(
                resource_id=a.id,
                resource_name=a.name,
                resource_type=a.resource_type,
                allocated_power_kw=round(float(result["allocations"].get(rid, 0.0)), 6),
                max_power_kw=a.rated_power_kw,
                min_power_kw=b["lo_kw"],
                available_power_kw=b["hi_kw"],
                marginal_cost_per_kwh=b["cost_up"]
                if body.target_power_kw >= 0
                else b["cost_down"],
                state_of_charge=a.soc if a.is_battery else None,
                soc_source=a.soc_source if a.is_battery else None,
                availability_basis=None if a.is_battery else a.availability_basis,
            )
        )

    shortfall = float(result["shortfall_kw"])
    solved = result["status"] in ("success", "fallback_used")
    success = solved and abs(shortfall) <= _SHORTFALL_TOL_KW
    if not solved:
        run_status = "failed"
    elif not success:
        run_status = "shortfall"
    else:
        run_status = result["status"]
    message = result.get("message") or ""
    if solved and not success:
        message = (
            f"target not reachable with the available resources; "
            f"delivered {result['delivered_kw']:.3f} kW, shortfall {shortfall:.3f} kW"
        )

    await finish_run(
        session,
        run_id,
        problem_type="dispatch",
        status=run_status,
        objective_value=result["objective_value"],
        solve_time_ms=solve_ms,
        solver=result["method"],
        fallback_used=result["fallback_used"],
        solution={
            "allocations": result["allocations"],
            "delivered_kw": result["delivered_kw"],
            "shortfall_kw": shortfall,
            "bounds": result["bounds"],
            "method": result["method"],
        },
        metadata={"message": message, "total_cost": result["objective_value"]},
    )
    return DispatchResponse(
        success=success,
        target_power_kw=body.target_power_kw,
        actual_power_kw=round(float(result["delivered_kw"]), 6),
        allocations=allocations,
        solve_time_ms=round(solve_ms, 3),
        fallback_used=result["fallback_used"],
        message=message,
        run_id=run_id,
        status=run_status,
        method=result["method"],
        shortfall_kw=round(shortfall, 6),
        objective_value=result["objective_value"],
    )


# ---------------------------------------------------------------------------
# Stochastic (CVaR) dispatch
# ---------------------------------------------------------------------------


def _expected_schedule(solution: dict[str, Any], cap_kwh: float, T: int) -> dict[str, list[float]]:
    """Probability-weighted schedule from a CVaR or rule-based solution."""
    scen = solution.get("scenarios")
    if isinstance(scen, list) and scen and isinstance(scen[0], dict) and "p_charge" in scen[0]:
        chg = [sum(s["probability"] * s["p_charge"][t] for s in scen) for t in range(T)]
        dis = [sum(s["probability"] * s["p_discharge"][t] for s in scen) for t in range(T)]
        soc = [sum(s["probability"] * s["soc"][t] for s in scen) / cap_kwh for t in range(T)]
    else:
        power = [float(p) for p in (solution.get("battery_power") or [0.0] * T)][:T]
        chg = [max(0.0, p) for p in power]
        dis = [max(0.0, -p) for p in power]
        soc = [float(s) for s in (solution.get("battery_soc") or [])][1 : T + 1]
    r = [round(x, 4) for x in chg], [round(x, 4) for x in dis]
    return {
        "charge": r[0],
        "discharge": r[1],
        "power": [round(c - d, 4) for c, d in zip(*r, strict=False)],
        "soc": [round(x, 4) for x in soc],
    }


@router.post("/stochastic", response_model=OptimizationResponse)
async def stochastic(
    body: StochasticRequest,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """Two-stage stochastic battery dispatch with a CVaR risk term.

    Price scenarios are sampled log-normally around ``base_prices``; the
    extensive-form MILP (``StochasticCVaRPlugin``) minimises
    ``E[cost] + risk_weight * CVaR_{1-risk_level}[cost]``, falling back to
    conservative percentile rules when the solver is unavailable.
    """
    import numpy as np

    from vpp.optimization import OptimizationProblem, solve_with_fallback

    T = body.time_horizon_hours
    battery: FleetAsset | None = None
    if body.resource_id:
        assets, missing = await load_fleet_assets(session, [body.resource_id], batteries_only=True)
        if missing or not assets:
            raise HTTPException(status_code=404, detail="battery resource not found or offline")
        battery = assets[0]
        bp = battery.battery_params()
    else:
        bp = {
            "battery_capacity_kwh": body.battery_capacity_kwh,
            "max_charge_kw": body.max_power_kw,
            "max_discharge_kw": body.max_power_kw,
            "soc_init": body.soc_init,
            "soc_min": body.soc_min,
            "soc_max": body.soc_max,
            "eta_charge": body.efficiency,
            "eta_discharge": body.efficiency,
        }
    base_prices = [float(p) for p in (body.base_prices or [50.0] * T)]
    base_load = [float(x) for x in (body.base_load or [100.0] * T)]
    rng = np.random.default_rng(body.seed)
    scenarios = []
    for _ in range(body.num_scenarios):
        mult = rng.lognormal(mean=0.0, sigma=body.volatility, size=T)
        scenarios.append(
            {
                "probability": 1.0 / body.num_scenarios,
                "prices": [float(p * m) for p, m in zip(base_prices, mult, strict=False)],
                "load": base_load,
            }
        )
    alpha = 1.0 - body.risk_level
    problem = OptimizationProblem(
        variables={},
        objectives=[{"type": "minimize_cost_cvar"}],
        constraints=[],
        parameters={
            **bp,
            "prices": base_prices,
            "dt_hours": 1.0,
            "scenarios": scenarios,
            "cvar_alpha": alpha,
            "cvar_lambda": body.risk_weight,
            # Keys read by the SimpleStochasticRules fallback.
            "battery_capacity": bp["battery_capacity_kwh"],
            "max_power": bp["max_discharge_kw"],
            "efficiency": bp["eta_discharge"],
        },
        time_horizon=T,
        time_step=1.0,
        # Must match the StochasticCVaRPlugin registration in
        # solve_with_fallback ("stochastic" never reaches the MILP).
        metadata={"type": "stochastic_dispatch", "num_scenarios": body.num_scenarios},
    )
    battery_summary = (
        battery.summary()
        if battery is not None
        else {
            "id": "inline",
            "name": "inline battery",
            "resource_type": "battery",
            "capacity_kwh": bp["battery_capacity_kwh"],
            "max_charge_kw": bp["max_charge_kw"],
            "max_discharge_kw": bp["max_discharge_kw"],
            "soc_init": bp["soc_init"],
            "soc_min": bp["soc_min"],
            "soc_max": bp["soc_max"],
            "eta_charge": bp["eta_charge"],
            "eta_discharge": bp["eta_discharge"],
        }
    )
    inputs = {
        "num_scenarios": body.num_scenarios,
        "time_horizon_hours": T,
        "risk_level": body.risk_level,
        "risk_weight": body.risk_weight,
        "volatility": body.volatility,
        "seed": body.seed,
        "base_prices": base_prices,
        "prices": base_prices,
        "interval_minutes": 60,
        "resource_ids": [battery.id] if battery is not None else [],
        "batteries": [battery_summary],
    }
    run_id, result, solve_ms = await _run_tracked(
        session,
        "stochastic",
        inputs,
        lambda: solve_with_fallback(
            problem, timeout_ms=body.timeout_ms, force_fallback=body.force_fallback
        ),
    )
    status_s = _status_str(result.status)
    solution = dict(result.solution or {})
    solver = str(solution.get("method") or (result.solver_info or {}).get("name") or "")
    if not solver and result.fallback_used:
        solver = "simple_stochastic_rules"
    if solution:
        solution.update(_expected_schedule(solution, bp["battery_capacity_kwh"], T))
    if (
        isinstance(solution.get("scenarios"), list)
        and len(solution["scenarios"]) > _MAX_PERSISTED_SCENARIOS
    ):
        solution["scenarios"] = solution["scenarios"][:_MAX_PERSISTED_SCENARIOS]
        solution["scenarios_truncated"] = True
    metadata = {
        **(result.metadata or {}),
        "total_cost": solution.get("expected_cost"),
        "cvar_alpha": alpha,
        "note": "scenario load is used only by the rule fallback; the CVaR MILP prices battery exchange",
    }
    await finish_run(
        session,
        run_id,
        problem_type="stochastic",
        status=status_s,
        objective_value=result.objective_value,
        solve_time_ms=solve_ms,
        solver=solver,
        fallback_used=result.fallback_used,
        solution=solution,
        metadata=metadata,
        iterations=result.iterations or None,
        gap=result.gap,
    )
    return OptimizationResponse(
        status=status_s,
        objective_value=_finite_or_zero(result.objective_value),
        solution=solution,
        solve_time_ms=round(solve_ms, 3),
        fallback_used=result.fallback_used,
        solver=solver,
        metadata={k: v for k, v in metadata.items() if k != "note"},
        run_id=run_id,
    )


# ---------------------------------------------------------------------------
# Real-time / distributed (framework problems)
# ---------------------------------------------------------------------------


async def _solve_framework_problem(
    session: AsyncSession,
    problem_type: str,
    inputs: dict[str, Any],
    problem: Any,
    timeout_ms: int,
    force_fallback: bool,
) -> OptimizationResponse:
    from vpp.optimization import solve_with_fallback

    run_id, result, solve_ms = await _run_tracked(
        session,
        problem_type,
        inputs,
        lambda: solve_with_fallback(problem, timeout_ms=timeout_ms, force_fallback=force_fallback),
    )
    status_s = _status_str(result.status)
    solver = str(
        (result.metadata or {}).get("method") or (result.solver_info or {}).get("name") or ""
    )
    await finish_run(
        session,
        run_id,
        problem_type=problem_type,
        status=status_s,
        objective_value=result.objective_value,
        solve_time_ms=solve_ms,
        solver=solver,
        fallback_used=result.fallback_used,
        solution=result.solution,
        metadata=result.metadata,
    )
    return OptimizationResponse(
        status=status_s,
        objective_value=_finite_or_zero(result.objective_value),
        solution=result.solution,
        solve_time_ms=round(solve_ms, 3),
        fallback_used=result.fallback_used,
        solver=solver,
        run_id=run_id,
    )


@router.post("/realtime", response_model=OptimizationResponse)
async def realtime(
    body: RealTimeRequest,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """Run real-time fast-dispatch optimization."""
    from vpp.optimization import create_realtime_problem

    problem = create_realtime_problem(
        current_state={
            "grid_frequency": body.grid_frequency_hz,
            "grid_voltage": body.grid_voltage_pu,
            "active_power_demand": body.active_power_demand_kw,
            "reactive_power_demand": body.reactive_power_demand_kvar,
        },
        forecasts=body.forecasts,
    )
    return await _solve_framework_problem(
        session,
        "realtime",
        body.model_dump(exclude={"forecasts"}),
        problem,
        body.timeout_ms,
        body.force_fallback,
    )


@router.post("/distributed", response_model=OptimizationResponse)
async def distributed(
    body: DistributedRequest,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """Run distributed multi-site optimization."""
    from vpp.optimization import create_distributed_problem

    sites_data = [s.model_dump() for s in body.sites]
    problem = create_distributed_problem(
        sites_data=sites_data,
        coordination_targets={
            "target_power": body.target_power_kw,
            "target_reserve": body.target_reserve_kw,
            "mode": body.coordination_mode,
        },
    )
    inputs = {
        "sites": [s.site_id for s in body.sites],
        "target_power_kw": body.target_power_kw,
        "target_reserve_kw": body.target_reserve_kw,
        "coordination_mode": body.coordination_mode,
    }
    return await _solve_framework_problem(
        session,
        "distributed",
        inputs,
        problem,
        body.timeout_ms,
        body.force_fallback,
    )


# ---------------------------------------------------------------------------
# Horizon schedule (MPC)
# ---------------------------------------------------------------------------


async def _tariff_prices(
    session: AsyncSession, body: ScheduleRequest
) -> tuple[Any, Any, list[float], datetime]:
    from vpp.tariffs import load_urdb_json
    from vpp.tariffs.optimization import load_nem3_avoided_cost_2024, tariff_to_opt_params

    row = await TariffRepository.get(session, body.tariff_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Tariff not found")
    try:
        tariff = load_urdb_json(json.loads(row.urdb_json) if row.urdb_json else {})
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid stored tariff: {exc}") from exc
    start = body.horizon_start
    if start is None:
        now = datetime.now(timezone.utc)
        start = now.replace(second=0, microsecond=0) - timedelta(
            minutes=now.minute % body.interval_minutes
        )
    elif start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    opt = tariff_to_opt_params(
        tariff,
        horizon_start=start,
        horizon_hours=body.horizon_hours or 24,
        interval_minutes=body.interval_minutes,
        nem=body.nem,
        nem3_avoided_cost=load_nem3_avoided_cost_2024() if body.nem == "nem3" else None,
    )
    return tariff, opt, list(opt.energy_buy_per_kwh), start


@router.post("/schedule", response_model=ScheduleResponse)
@router.post("/mpc", response_model=ScheduleResponse, include_in_schema=False)
async def schedule(
    body: ScheduleRequest,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """Optimise battery charge/discharge over a horizon (one MPC solve).

    Uses the M3/M4 MPC controllers: a single battery gets the full tariff
    (TOU, export regime, demand charges) and rainflow-consistent wear-cost
    hooks; fleets are solved jointly (monolithic MILP, or ADMM above ten
    batteries) with feeder limits and a per-battery SOH-aware wear cost.
    Prices come from ``prices`` or are derived from ``tariff_id``.
    """
    batteries, missing = await load_fleet_assets(session, body.resource_ids, batteries_only=True)
    if missing:
        raise HTTPException(
            status_code=404, detail={"message": "unknown resource ids", "missing": missing}
        )
    if not batteries:
        raise HTTPException(status_code=422, detail="no online battery resources to schedule")
    steps = body.steps
    if len(batteries) * steps > MAX_SCHEDULE_RESOURCE_STEPS:
        raise HTTPException(
            status_code=422,
            detail=f"batteries x steps must be <= {MAX_SCHEDULE_RESOURCE_STEPS} "
            f"(got {len(batteries)} x {steps}); narrow resource_ids or the horizon",
        )

    tariff = tariff_opt = None
    horizon_start: datetime | None = None
    if body.tariff_id is not None:
        tariff, tariff_opt, prices, horizon_start = await _tariff_prices(session, body)
    else:
        prices = [float(p) for p in body.prices or []]

    summaries = [b.summary() for b in batteries]
    inputs = {
        "prices": prices,
        "interval_minutes": body.interval_minutes,
        "resource_ids": [b.id for b in batteries],
        "batteries": summaries,
        "tariff_id": body.tariff_id,
        "nem": body.nem if body.tariff_id else None,
        "horizon_start": horizon_start,
        "load_kw": body.load_kw,
        "solar_kw": body.solar_kw,
        "degradation_aware": body.degradation_aware,
        "replacement_cost_per_kwh": body.replacement_cost_per_kwh,
        "feeder_max_import_kw": body.feeder_max_import_kw,
        "feeder_max_export_kw": body.feeder_max_export_kw,
        "terminal_soc_policy": body.terminal_soc_policy,
        "force_fallback": body.force_fallback,
    }
    run_id, plan, solve_ms = await _run_tracked(
        session,
        "schedule",
        inputs,
        lambda: plan_schedule(
            batteries,
            prices=prices,
            interval_minutes=body.interval_minutes,
            load_kw=body.load_kw,
            solar_kw=body.solar_kw,
            tariff_opt=tariff_opt,
            tariff=tariff,
            degradation_aware=body.degradation_aware,
            replacement_cost_per_kwh=body.replacement_cost_per_kwh,
            feeder_max_import_kw=body.feeder_max_import_kw,
            feeder_max_export_kw=body.feeder_max_export_kw,
            timeout_ms=body.timeout_ms,
            force_fallback=body.force_fallback,
            terminal_soc_policy=body.terminal_soc_policy,
            now=horizon_start,
        ),
    )
    run_status = "fallback_used" if plan["fallback_used"] else "success"
    solution = {
        k: plan[k]
        for k in (
            "method",
            "objective",
            "energy_cost",
            "wear_cost",
            "charge",
            "discharge",
            "power",
            "per_resource",
        )
    }
    await finish_run(
        session,
        run_id,
        problem_type="schedule",
        status=run_status,
        objective_value=plan["objective"],
        solve_time_ms=solve_ms,
        solver=plan["method"],
        fallback_used=plan["fallback_used"],
        solution=solution,
        metadata={
            "total_cost": plan["energy_cost"],
            "fallback_reason": plan["fallback_reason"],
            "notes": plan["notes"],
            "terminal_soc_policy": plan["terminal_soc_policy"],
            "wear_cost_per_kwh": plan["wear_cost_per_kwh"],
            **plan["solver_meta"],
        },
        iterations=plan["solver_meta"].get("solver_iterations")
        or plan["solver_meta"].get("iterations"),
    )
    return ScheduleResponse(
        run_id=run_id,
        status=run_status,
        method=plan["method"],
        fallback_used=plan["fallback_used"],
        fallback_reason=plan["fallback_reason"],
        solve_time_ms=round(solve_ms, 3),
        objective_value=plan["objective"],
        energy_cost=plan["energy_cost"],
        wear_cost=plan["wear_cost"],
        interval_minutes=body.interval_minutes,
        prices=prices,
        charge=plan["charge"],
        discharge=plan["discharge"],
        power=plan["power"],
        per_resource=plan["per_resource"],
        resources=summaries,
        tariff_id=body.tariff_id,
        terminal_soc_policy=plan["terminal_soc_policy"],
        notes=plan["notes"],
    )


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------


@router.post("/backtest", response_model=BacktestResponse)
async def backtest(
    body: BacktestRequest,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """Closed-loop MPC backtest over a historical price series.

    Each tick the controller sees a forecast (``perfect``, day-ahead
    ``persistence``, or seeded ``noisy``), commits only its first-step
    decision, and the battery is simulated against the true prices. Results
    are compared with idling, the rule-based dispatcher and the
    perfect-foresight offline optimum (``regret``).
    """
    if body.resource_id is not None:
        assets, missing = await load_fleet_assets(session, [body.resource_id], batteries_only=True)
        if missing or not assets:
            raise HTTPException(status_code=404, detail="battery resource not found or offline")
        battery = assets[0]
    else:
        spec = body.battery
        battery = FleetAsset(
            id="inline",
            name="inline battery",
            resource_type="battery",
            rated_power_kw=spec.max_power_kw,
            capacity_kwh=spec.capacity_kwh,
            capacity_source="request",
            soc=spec.soc_init,
            soc_source="request",
            soh=spec.state_of_health,
            eta_charge=spec.eta_charge,
            eta_discharge=spec.eta_discharge,
            soc_min=spec.soc_min,
            soc_max=spec.soc_max,
        )
    inputs = {
        "prices": body.prices,
        "interval_minutes": body.interval_minutes,
        "horizon_steps": body.horizon_steps,
        "forecast_mode": body.forecast_mode,
        "noise_sigma": body.noise_sigma,
        "seed": body.seed,
        "terminal_soc_policy": body.terminal_soc_policy,
        "resource_ids": [battery.id] if body.resource_id else [],
        "batteries": [battery.summary()],
        "load_kw": body.load_kw,
        "solar_kw": body.solar_kw,
    }
    run_id, res, solve_ms = await _run_tracked(
        session,
        "backtest",
        inputs,
        lambda: run_closed_loop_backtest(
            battery,
            prices=body.prices,
            interval_minutes=body.interval_minutes,
            horizon_steps=body.horizon_steps,
            load_kw=body.load_kw,
            solar_kw=body.solar_kw,
            forecast_mode=body.forecast_mode,
            noise_sigma=body.noise_sigma,
            seed=body.seed,
            solver_timeout_ms=body.solver_timeout_ms,
            start=body.start,
            compare_offline=body.compare_offline,
            terminal_soc_policy=body.terminal_soc_policy,
        ),
    )
    notes = [
        f"ticks within {body.horizon_steps} steps of the window end see forecasts "
        "padded with the last price; expect end-of-window effects in regret",
        "costs count only the battery's net energy exchange (idle = 0)",
    ]
    fallback_used = res["fallback_count"] > 0
    run_status = "fallback_used" if fallback_used else "success"
    solution = {
        "charge": res["charge"],
        "discharge": res["discharge"],
        "power": res["power"],
        "soc": res["soc"],
        "per_resource": {
            battery.id: {"charge": res["charge"], "discharge": res["discharge"], "soc": res["soc"]}
        },
    }
    summary = {k: v for k, v in res.items() if not isinstance(v, list)}
    await finish_run(
        session,
        run_id,
        problem_type="backtest",
        status=run_status,
        objective_value=res["realized_cost_adjusted"],
        solve_time_ms=solve_ms,
        solver="mpc_milp_highs",
        fallback_used=fallback_used,
        solution=solution,
        metadata={**summary, "total_cost": res["realized_cost"], "notes": notes},
        iterations=res["cumulative_solver_iterations"] or None,
    )
    return BacktestResponse(
        run_id=run_id,
        status=run_status,
        forecast_mode=body.forecast_mode,
        interval_minutes=body.interval_minutes,
        horizon_steps=body.horizon_steps,
        notes=notes,
        **{k: v for k, v in res.items() if k != "realized_price"},
    )


# ---------------------------------------------------------------------------
# Run history
# ---------------------------------------------------------------------------


def _utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


async def _list_runs(
    session: AsyncSession,
    *,
    skip: int,
    limit: int,
    problem_type: str | None,
    start: datetime | None,
    end: datetime | None,
    resource_ids: list[str] | None,
    include_details: bool,
) -> list[dict[str, Any]]:
    runs = await OptimizationRepository.list_runs(
        session,
        skip=skip,
        limit=limit,
        problem_type=problem_type,
        start=_utc(start),
        end=_utc(end),
        resource_ids=resource_ids or None,
    )
    return [run_to_dict(r, include_details=include_details) for r in runs]


async def _get_run_or_404(session: AsyncSession, run_id: str):
    row = await OptimizationRepository.get_run(session, run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Optimization run not found")
    return row


@router.get("/runs", response_model=list[OptimizationRunRead])
@router.get("/history", response_model=list[OptimizationRunRead])
async def list_runs(
    skip: int = Query(0, ge=0),
    offset: int | None = Query(None, ge=0, description="Alias of skip"),
    limit: int = Query(50, ge=1, le=200),
    problem_type: str | None = None,
    start: datetime | None = Query(None, description="Only runs created at or after this time"),
    end: datetime | None = Query(None, description="Only runs created at or before this time"),
    resource_id: list[str] | None = Query(
        None, description="Only runs touching these resources (repeatable)"
    ),
    include_details: bool = Query(True, description="Include inputs / solution / metadata"),
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """List persisted optimization runs, newest first."""
    return await _list_runs(
        session,
        skip=offset if offset is not None else skip,
        limit=limit,
        problem_type=problem_type,
        start=start,
        end=end,
        resource_ids=resource_id,
        include_details=include_details,
    )


@router.get("/runs/{run_id}", response_model=OptimizationRunRead)
async def get_run(
    run_id: str,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """Fetch one optimization run with its inputs and solution."""
    return run_to_dict(await _get_run_or_404(session, run_id))


async def _explain(session: AsyncSession, run_id: str):
    row = await _get_run_or_404(session, run_id)
    inputs = json.loads(row.parameters_json) if row.parameters_json else {}
    solution = unpack_solution(row)["solution"]
    try:
        explained = explain_schedule(inputs if isinstance(inputs, dict) else {}, solution)
    except Exception:
        explained = None
    if explained is None:
        # No per-step schedule (e.g. single-interval dispatch, running or
        # failed run): 204 lets the UI show its "not available" state.
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    return {"run_id": row.id, **explained}


@router.get(
    "/runs/{run_id}/explain",
    response_model=ExplainerResponse,
    responses={204: {"description": "Run has no per-step schedule to explain"}},
)
@router.get("/explain/{run_id}", response_model=ExplainerResponse, include_in_schema=False)
async def explain_run(
    run_id: str,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """Counterfactual explainer: the run vs. idling and the rule-based
    dispatcher on the same prices, plus binding SOC / power constraints."""
    return await _explain(session, run_id)


@router.get("/stats")
async def stats(
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """Aggregate optimization performance statistics."""
    return await OptimizationRepository.get_stats(session)


# ---------------------------------------------------------------------------
# /api/v1/dispatches aliases (operator UI contract)
# ---------------------------------------------------------------------------


@dispatches_router.get("", response_model=list[OptimizationRunRead])
@dispatches_router.get("/", response_model=list[OptimizationRunRead], include_in_schema=False)
async def list_dispatches(
    skip: int = Query(0, ge=0),
    offset: int | None = Query(None, ge=0),
    limit: int = Query(50, ge=1, le=200),
    problem_type: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    resource_id: list[str] | None = Query(None),
    include_details: bool = True,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """Alias of ``GET /api/v1/optimization/runs``."""
    return await _list_runs(
        session,
        skip=offset if offset is not None else skip,
        limit=limit,
        problem_type=problem_type,
        start=start,
        end=end,
        resource_ids=resource_id,
        include_details=include_details,
    )


@dispatches_router.get("/{run_id}", response_model=OptimizationRunRead)
async def get_dispatch(
    run_id: str,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """Alias of ``GET /api/v1/optimization/runs/{run_id}``."""
    return run_to_dict(await _get_run_or_404(session, run_id))


@dispatches_router.get(
    "/{run_id}/explain",
    response_model=ExplainerResponse,
    responses={204: {"description": "Run has no per-step schedule to explain"}},
)
async def explain_dispatch(
    run_id: str,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """Alias of ``GET /api/v1/optimization/runs/{run_id}/explain``."""
    return await _explain(session, run_id)
