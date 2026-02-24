"""Optimization dispatch and solve routes."""

from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from vpp.api.deps import get_vpp
from vpp.auth.security import get_current_user
from vpp.core import VirtualPowerPlant
from vpp.db.engine import get_db
from vpp.db.models import UserModel
from vpp.db.repositories import OptimizationRepository
from vpp.schemas.optimization import (
    DispatchRequest,
    DispatchResponse,
    OptimizationResponse,
    ResourceAllocation,
    StochasticRequest,
    RealTimeRequest,
    DistributedRequest,
)

router = APIRouter(prefix="/api/v1/optimization", tags=["Optimization"])


@router.post("/dispatch", response_model=DispatchResponse)
async def dispatch(
    body: DispatchRequest,
    session: AsyncSession = Depends(get_db),
    vpp: VirtualPowerPlant = Depends(get_vpp),
    _user: UserModel = Depends(get_current_user),
):
    """Run power dispatch across registered resources."""
    t0 = time.perf_counter()
    success = vpp.optimize_dispatch(body.target_power_kw)
    solve_ms = (time.perf_counter() - t0) * 1000

    allocations = []
    actual = 0.0
    for res in vpp.resources:
        alloc = res._current_power
        actual += alloc
        allocations.append(ResourceAllocation(
            resource_id=res.name,
            resource_name=res.name,
            allocated_power_kw=alloc,
            max_power_kw=res.rated_power,
        ))

    await OptimizationRepository.record_run(
        session,
        problem_type="dispatch",
        status="success" if success else "failed",
        objective_value=actual,
        solve_time_ms=solve_ms,
    )

    return DispatchResponse(
        success=success,
        target_power_kw=body.target_power_kw,
        actual_power_kw=actual,
        allocations=allocations,
        solve_time_ms=round(solve_ms, 3),
    )


@router.post("/stochastic", response_model=OptimizationResponse)
async def stochastic(
    body: StochasticRequest,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """Run stochastic (CVaR) optimization."""
    from vpp.optimization import create_stochastic_problem, solve_with_fallback

    t0 = time.perf_counter()
    problem = create_stochastic_problem(
        base_data={
            "base_prices": body.base_prices or [50.0] * body.time_horizon_hours,
            "base_load": body.base_load or [100.0] * body.time_horizon_hours,
        },
        num_scenarios=body.num_scenarios,
        uncertainty_config={"price_volatility": body.volatility},
    )
    result = solve_with_fallback(problem, timeout_ms=body.timeout_ms)
    solve_ms = (time.perf_counter() - t0) * 1000

    await OptimizationRepository.record_run(
        session,
        problem_type="stochastic",
        status=result.status.value if hasattr(result.status, "value") else str(result.status),
        objective_value=result.objective_value,
        solve_time_ms=solve_ms,
        fallback_used=result.fallback_used,
        solution=result.solution,
    )

    return OptimizationResponse(
        status=result.status.value if hasattr(result.status, "value") else str(result.status),
        objective_value=result.objective_value,
        solution=result.solution,
        solve_time_ms=round(solve_ms, 3),
        fallback_used=result.fallback_used,
    )


@router.post("/realtime", response_model=OptimizationResponse)
async def realtime(
    body: RealTimeRequest,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """Run real-time fast-dispatch optimization."""
    from vpp.optimization import create_realtime_problem, solve_with_fallback

    t0 = time.perf_counter()
    problem = create_realtime_problem(
        current_state={
            "grid_frequency": body.grid_frequency_hz,
            "grid_voltage": body.grid_voltage_pu,
            "active_power_demand": body.active_power_demand_kw,
            "reactive_power_demand": body.reactive_power_demand_kvar,
        },
        forecasts=body.forecasts,
    )
    result = solve_with_fallback(problem, timeout_ms=body.timeout_ms)
    solve_ms = (time.perf_counter() - t0) * 1000

    await OptimizationRepository.record_run(
        session,
        problem_type="realtime",
        status=result.status.value if hasattr(result.status, "value") else str(result.status),
        objective_value=result.objective_value,
        solve_time_ms=solve_ms,
        fallback_used=result.fallback_used,
        solution=result.solution,
    )

    return OptimizationResponse(
        status=result.status.value if hasattr(result.status, "value") else str(result.status),
        objective_value=result.objective_value,
        solution=result.solution,
        solve_time_ms=round(solve_ms, 3),
        fallback_used=result.fallback_used,
    )


@router.post("/distributed", response_model=OptimizationResponse)
async def distributed(
    body: DistributedRequest,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """Run distributed multi-site optimization."""
    from vpp.optimization import create_distributed_problem, solve_with_fallback

    t0 = time.perf_counter()
    sites_data = [s.model_dump() for s in body.sites]
    problem = create_distributed_problem(
        sites_data=sites_data,
        coordination_targets={
            "target_power": body.target_power_kw,
            "target_reserve": body.target_reserve_kw,
            "mode": body.coordination_mode,
        },
    )
    result = solve_with_fallback(problem, timeout_ms=body.timeout_ms)
    solve_ms = (time.perf_counter() - t0) * 1000

    await OptimizationRepository.record_run(
        session,
        problem_type="distributed",
        status=result.status.value if hasattr(result.status, "value") else str(result.status),
        objective_value=result.objective_value,
        solve_time_ms=solve_ms,
        fallback_used=result.fallback_used,
        solution=result.solution,
    )

    return OptimizationResponse(
        status=result.status.value if hasattr(result.status, "value") else str(result.status),
        objective_value=result.objective_value,
        solution=result.solution,
        solve_time_ms=round(solve_ms, 3),
        fallback_used=result.fallback_used,
    )


@router.get("/history")
async def history(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    problem_type: Optional[str] = None,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """List past optimization runs."""
    runs = await OptimizationRepository.list_runs(session, skip=skip, limit=limit, problem_type=problem_type)
    return [
        {
            "id": r.id,
            "problem_type": r.problem_type,
            "status": r.status,
            "objective_value": r.objective_value,
            "solve_time_ms": r.solve_time_ms,
            "fallback_used": r.fallback_used,
            "created_at": r.created_at,
        }
        for r in runs
    ]


@router.get("/stats")
async def stats(
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """Aggregate optimization performance statistics."""
    return await OptimizationRepository.get_stats(session)
