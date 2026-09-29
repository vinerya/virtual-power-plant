"""Database / event-bus plumbing for the optimization routes.

Keeps :mod:`vpp.api.routes.optimization` focused on HTTP concerns:

* :func:`load_fleet_assets` turns persisted resources (the database is the
  source of truth) into :class:`~vpp.optimization.planning.FleetAsset`
  descriptions, resolving capacity, SOC and SOH from the best data available.
* :func:`start_run` / :func:`finish_run` persist every optimization run in
  ``optimization_runs`` and publish ``OPTIMIZATION_*`` events, which the
  WebSocket bridge forwards on the ``optimization_events`` channel.
* :func:`run_to_dict` renders a run row in the ``DispatchRun`` shape the web
  UI consumes.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from vpp.db.models import OptimizationRunModel, ResourceModel
from vpp.db.repositories import OptimizationRepository
from vpp.events import Event, EventType, get_event_bus
from vpp.optimization.planning import FleetAsset
from vpp.portal.telemetry import latest_soc

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sqlalchemy.ext.asyncio import AsyncSession

    from vpp.control.actuator import OutputLimit

logger = logging.getLogger(__name__)

# Marker for the wrapped solution_json layout written by this module. Rows
# written by older code store the bare solution dict.
_SOLUTION_LAYOUT_VERSION = 2
# C/4 heuristic used when a battery has no recorded energy capacity, matching
# vpp.degradation.telemetry.DEFAULT_C_RATE_HOURS.
_DEFAULT_C_RATE_HOURS = 4.0
#: Oldest reading taken before a VPP output cap that still estimates a
#: curtailed generator's availability; after that the nameplate is used.
PRE_CURTAILMENT_MAX_AGE_S = 900.0


# ---------------------------------------------------------------------------
# Resources -> FleetAsset
# ---------------------------------------------------------------------------


def _loads(raw: str | None) -> dict[str, Any]:
    try:
        val = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        return {}
    return val if isinstance(val, dict) else {}


def _first_number(*candidates: Any) -> float | None:
    for c in candidates:
        if isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(float(c)):
            return float(c)
    return None


async def _latest_soc(session: AsyncSession, resource_id: str) -> float | None:
    """Newest telemetry SOC (0-1) from ``battery_states`` (MQTT) or
    ``resource_telemetry`` (Modbus / ingest endpoint), whichever is newer."""
    return (await latest_soc(session, [resource_id])).get(resource_id)


def _fraction(v: float | None) -> float | None:
    """Accept SOC as a fraction (0..1) or a percentage (1..100)."""
    if v is None:
        return None
    return v / 100.0 if v > 1.0 else v


async def resource_to_asset(
    session: AsyncSession,
    row: ResourceModel,
    limits: dict[str, OutputLimit] | None = None,
) -> FleetAsset:
    """One resource as a :class:`FleetAsset`.

    *limits* are the output caps the VPP currently holds on generation
    resources (:func:`vpp.control.actuator.active_output_limits`). A
    resource producing at such a cap is curtailed, so its polled output is
    not its availability (see ``availability_basis``); without this the
    optimiser could never allocate more than the cap and never lift it.
    """
    cfg = _loads(row.config_json)
    meta = _loads(row.metadata_json)
    asset = FleetAsset(
        id=row.id,
        name=row.name,
        resource_type=row.resource_type,
        rated_power_kw=float(row.rated_power or 0.0),
        online=bool(row.online),
        soh=float(row.state_of_health if row.state_of_health is not None else 1.0),
        chemistry=row.chemistry,
        metadata=meta,
    )
    if asset.is_battery:
        cap = _first_number(
            row.nominal_energy_kwh, cfg.get("capacity_kwh"), meta.get("capacity_kwh")
        )
        if cap is not None and cap > 0:
            asset.capacity_kwh = cap
            asset.capacity_source = "recorded"
        else:
            asset.capacity_kwh = max(1e-3, asset.rated_power_kw * _DEFAULT_C_RATE_HOURS)
            asset.capacity_source = "c4_heuristic"

        soc = await _latest_soc(session, row.id)
        source = "telemetry"
        if soc is None:
            charge = _first_number(cfg.get("current_charge_kwh"), meta.get("current_charge_kwh"))
            if charge is not None and asset.capacity_kwh:
                soc, source = charge / asset.capacity_kwh, "config"
        if soc is None:
            soc = _fraction(
                _first_number(
                    cfg.get("state_of_charge"), meta.get("state_of_charge"), meta.get("soc")
                )
            )
            source = "config"
        if soc is None:
            soc, source = 0.5, "assumed"
        asset.soc = max(0.0, min(1.0, soc))
        asset.soc_source = source

        # The row's single ``efficiency`` column is treated as the one-way
        # (per-leg) efficiency unless per-leg values were recorded.
        row_eff = _first_number(row.efficiency)
        if row_eff is not None and 0.0 < row_eff <= 1.0:
            asset.eta_charge = asset.eta_discharge = row_eff
        for key, attr in (
            ("charge_efficiency", "eta_charge"),
            ("discharge_efficiency", "eta_discharge"),
            ("soc_min", "soc_min"),
            ("soc_max", "soc_max"),
            ("max_charge_kw", "max_charge_kw"),
            ("max_discharge_kw", "max_discharge_kw"),
        ):
            v = _first_number(cfg.get(key), meta.get(key))
            if v is not None:
                setattr(asset, attr, v)
        if not (0.0 < asset.eta_charge <= 1.0 and 0.0 < asset.eta_discharge <= 1.0):
            asset.eta_charge = asset.eta_discharge = 0.95
        if not (0.0 <= asset.soc_min < asset.soc_max <= 1.0):
            asset.soc_min, asset.soc_max = 0.05, 0.95
        for attr in ("max_charge_kw", "max_discharge_kw"):
            v = getattr(asset, attr)
            if v is not None and v <= 0:
                setattr(asset, attr, None)
    else:
        avail = _first_number(meta.get("available_kw"), cfg.get("available_kw"))
        polled = float(row.current_power or 0.0)
        limit = (limits or {}).get(row.id)
        if limit is not None:
            asset.output_limit_kw = limit.limit_kw
        if avail is not None:
            asset.available_kw, asset.availability_basis = avail, "reported"
        elif limit is not None and polled >= limit.limit_kw - _curtailed_tolerance_kw(asset):
            # The output sits at a cap the VPP wrote: the reading is what the
            # VPP allowed, not what the resource could deliver. Neither the
            # AC output nor any point of models 103/113/123 tells the real
            # availability, so estimate it: the newest reading taken before
            # the cap if it is recent, else the nameplate (an upper bound).
            before = await _reading_before(session, row.id, limit.since)
            if before is not None:
                asset.available_kw = max(polled, before)
                asset.availability_basis = "pre_curtailment_telemetry"
            else:
                asset.available_kw = asset.rated_power_kw
                asset.availability_basis = "nameplate_curtailed"
        elif polled > 0:
            asset.available_kw, asset.availability_basis = polled, "telemetry"
        else:
            asset.available_kw, asset.availability_basis = asset.rated_power_kw, "nameplate"
    return asset


def _curtailed_tolerance_kw(asset: FleetAsset) -> float:
    """How close to a VPP cap a reading must be to count as held at it."""
    return max(0.05, 0.02 * max(0.0, asset.rated_power_kw))


async def _reading_before(session: AsyncSession, resource_id: str, since: float) -> float | None:
    """Newest telemetry power taken before *since* (epoch s), if recent enough."""
    from vpp.db.models import ResourceTelemetryModel

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(seconds=PRE_CURTAILMENT_MAX_AGE_S)
    before = datetime.fromtimestamp(since, tz=timezone.utc)
    rt = ResourceTelemetryModel
    value = (
        await session.execute(
            select(rt.power_kw)
            .where(rt.resource_id == resource_id, rt.timestamp < before, rt.timestamp >= cutoff)
            .order_by(rt.timestamp.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    return _first_number(value)


async def load_fleet_assets(
    session: AsyncSession,
    resource_ids: Iterable[str] | None = None,
    *,
    batteries_only: bool = False,
    online_only: bool = True,
    limit: int = 500,
) -> tuple[list[FleetAsset], list[str]]:
    """Load resources from the database as :class:`FleetAsset` objects.

    Returns ``(assets, missing_ids)``: ``missing_ids`` lists requested ids that
    do not exist (offline/non-battery resources are silently filtered out).
    """
    stmt = select(ResourceModel).order_by(ResourceModel.created_at.asc())
    ids = [str(i) for i in resource_ids] if resource_ids else None
    if ids:
        stmt = stmt.where(ResourceModel.id.in_(ids))
    if online_only:
        stmt = stmt.where(ResourceModel.online.is_(True))
    if batteries_only:
        stmt = stmt.where(ResourceModel.resource_type == "battery")
    rows = list((await session.execute(stmt.limit(limit))).scalars().all())
    missing: list[str] = []
    if ids:
        found_any = set(
            (await session.execute(select(ResourceModel.id).where(ResourceModel.id.in_(ids))))
            .scalars()
            .all()
        )
        missing = [i for i in ids if i not in found_any]
    from vpp.control.actuator import active_output_limits

    limits = active_output_limits()
    assets = [await resource_to_asset(session, r, limits) for r in rows]
    return assets, missing


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------


def _finite(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats (JSON has no inf/NaN) with None."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


async def publish_optimization_event(
    event_type: EventType, data: dict[str, Any], severity: str = "info"
) -> None:
    """Publish on the EventBus; failures are logged, never raised."""
    try:
        await get_event_bus().publish(
            Event(
                event_type=event_type,
                data=_json_safe(data),
                source="api.optimization",
                severity=severity,
            )
        )
    except Exception:
        logger.warning("failed to publish %s", event_type, exc_info=True)


async def start_run(
    session: AsyncSession, problem_type: str, parameters: dict[str, Any]
) -> OptimizationRunModel:
    """Persist a ``running`` row, commit it and announce the run."""
    row = await OptimizationRepository.record_run(
        session,
        problem_type=problem_type,
        status="running",
        parameters=_json_safe(parameters),
    )
    await session.commit()
    await publish_optimization_event(
        EventType.OPTIMIZATION_STARTED,
        {"run_id": row.id, "problem_type": problem_type, "status": "running", "method": "pending"},
    )
    return row


FAILED_STATUSES = frozenset({"failed", "infeasible", "unbounded", "timeout", "error"})


async def finish_run(
    session: AsyncSession,
    run_id: str,
    *,
    problem_type: str,
    status: str,
    objective_value: Any = None,
    solve_time_ms: float = 0.0,
    solver: str = "",
    fallback_used: bool = False,
    solution: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    iterations: int | None = None,
    gap: float | None = None,
) -> OptimizationRunModel | None:
    """Record the outcome of a run, commit, and publish completed/failed."""
    payload = {
        "__v": _SOLUTION_LAYOUT_VERSION,
        "solution": _json_safe(solution or {}),
        "metadata": _json_safe(metadata or {}),
        "iterations": iterations,
        "gap": _finite(gap),
    }
    obj = _finite(objective_value)
    row = await OptimizationRepository.update_run(
        session,
        run_id,
        status=status,
        objective_value=obj if obj is not None else 0.0,
        solve_time_ms=float(solve_time_ms),
        solver=solver[:100],
        fallback_used=bool(fallback_used),
        solution=payload,
    )
    await session.commit()
    failed = status in FAILED_STATUSES
    await publish_optimization_event(
        EventType.OPTIMIZATION_FAILED if failed else EventType.OPTIMIZATION_COMPLETED,
        {
            "run_id": run_id,
            "problem_type": problem_type,
            "status": status,
            "solver": solver,
            # ``problem_type`` / ``method`` / ``duration_s`` are consumed by
            # vpp.metrics.observe_event, which turns these events into
            # Prometheus metrics (do not also call the metrics helpers).
            "method": solver or "none",
            "fallback_used": bool(fallback_used),
            "objective_value": obj,
            "solve_time_ms": round(float(solve_time_ms), 3),
            "duration_s": float(solve_time_ms) / 1000.0,
            "duration_seconds": float(solve_time_ms) / 1000.0,
            "error": (metadata or {}).get("error"),
        },
        severity="warning" if failed else "info",
    )
    return row


# ---------------------------------------------------------------------------
# Reusable single-interval dispatch (used by the DR orchestrator)
# ---------------------------------------------------------------------------

_SHORTFALL_TOL_KW = 1e-3


async def execute_dispatch(
    session: AsyncSession,
    assets: list[FleetAsset],
    target_power_kw: float,
    *,
    interval_minutes: int = 15,
    constraints: dict[str, dict[str, Any]] | None = None,
    problem_type: str = "dispatch",
    extra_inputs: dict[str, Any] | None = None,
    timeout_ms: int = 5000,
    force_fallback: bool = False,
    wear_cost: bool = True,
    replacement_cost_per_kwh: float | None = None,
    apply: bool = False,
    actuator: Any = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Split *target_power_kw* (export-positive) across *assets* and record the run.

    The single implementation behind ``POST /api/v1/optimization/dispatch``
    and the DR orchestrator: the allocation (Pyomo/HiGHS LP with rule-based
    fallback via :func:`vpp.optimization.planning.allocate_power`) and the
    run bookkeeping (``optimization_runs`` row + ``OPTIMIZATION_*`` events).
    Per-resource power and energy limits are hard bounds of the allocation,
    so the result never exceeds a resource's limits; an unreachable target
    is reported as ``status="shortfall"``. Never raises for solver errors: a
    failed solve is recorded and returned with ``status="failed"`` and an
    ``error`` message.

    With ``apply=True`` the stationary allocations are handed to the
    setpoint actuator (:mod:`vpp.control.actuator`) for the dispatch
    interval; its per-resource results are returned as
    ``device_deliveries`` and stored in the run's metadata.
    """
    import time

    from starlette.concurrency import run_in_threadpool

    from vpp.optimization.planning import DEFAULT_REPLACEMENT_COST_PER_KWH, allocate_power

    replacement = (
        DEFAULT_REPLACEMENT_COST_PER_KWH
        if replacement_cost_per_kwh is None
        else float(replacement_cost_per_kwh)
    )
    dt_hours = max(1, int(interval_minutes)) / 60.0
    inputs = {
        "target_power_kw": target_power_kw,
        "interval_minutes": interval_minutes,
        "resource_ids": [a.id for a in assets],
        "resource_constraints": constraints or {},
        "wear_cost": wear_cost,
        "replacement_cost_per_kwh": replacement,
        "force_fallback": force_fallback,
        "apply": apply,
        "resources": [a.summary() for a in assets],
        **(extra_inputs or {}),
    }
    run = await start_run(session, problem_type, inputs)
    t0 = time.perf_counter()
    try:
        result = await run_in_threadpool(
            lambda: allocate_power(
                assets,
                target_power_kw,
                dt_hours=dt_hours,
                constraints=constraints,
                timeout_ms=timeout_ms,
                force_fallback=force_fallback,
                wear_cost=wear_cost,
                replacement_cost_per_kwh=replacement,
            )
        )
    except Exception as exc:
        solve_ms = (time.perf_counter() - t0) * 1000
        message = f"{type(exc).__name__}: {exc}"
        await finish_run(
            session,
            run.id,
            problem_type=problem_type,
            status="failed",
            solve_time_ms=solve_ms,
            metadata={"error": message},
        )
        return {
            "run_id": run.id,
            "status": "failed",
            "success": False,
            "method": "none",
            "fallback_used": False,
            "allocations": {},
            "bounds": {},
            "delivered_kw": 0.0,
            "shortfall_kw": float(target_power_kw),
            "message": message,
            "error": message,
            "solve_time_ms": solve_ms,
            "objective_value": None,
            "device_deliveries": [],
        }
    solve_ms = (time.perf_counter() - t0) * 1000

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
    deliveries: list[dict[str, Any]] = []
    run_metadata: dict[str, Any] = {"message": message, "total_cost": result["objective_value"]}
    if apply and solved:
        if actuator is None:
            from vpp.control.actuator import get_setpoint_actuator

            actuator = get_setpoint_actuator()
        deliveries = await actuator.apply(
            assets,
            dict(result["allocations"]),
            source=source or problem_type,
            run_id=run.id,
            ttl_s=max(1, int(interval_minutes)) * 60.0,
            session=session,
        )
        run_metadata["device_deliveries"] = deliveries
    await finish_run(
        session,
        run.id,
        problem_type=problem_type,
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
        metadata=run_metadata,
    )
    return {
        "run_id": run.id,
        "status": run_status,
        "success": success,
        "method": result["method"],
        "fallback_used": bool(result["fallback_used"]),
        "allocations": dict(result["allocations"]),
        "bounds": result["bounds"],
        "delivered_kw": float(result["delivered_kw"]),
        "shortfall_kw": shortfall,
        "message": message,
        "solve_time_ms": solve_ms,
        "objective_value": result["objective_value"],
        "device_deliveries": deliveries,
    }


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def unpack_solution(row: OptimizationRunModel) -> dict[str, Any]:
    """Return ``{'solution', 'metadata', 'iterations', 'gap'}`` for any row layout."""
    raw = _loads(row.solution_json)
    if raw.get("__v") == _SOLUTION_LAYOUT_VERSION:
        return {
            "solution": raw.get("solution") or {},
            "metadata": raw.get("metadata") or {},
            "iterations": raw.get("iterations"),
            "gap": raw.get("gap"),
        }
    return {"solution": raw, "metadata": {}, "iterations": None, "gap": None}


def run_to_dict(row: OptimizationRunModel, *, include_details: bool = True) -> dict[str, Any]:
    """Render a run in the web UI's ``DispatchRun`` shape (a superset of the
    legacy ``/history`` item)."""
    unpacked = unpack_solution(row)
    inputs = _loads(row.parameters_json)
    metadata = unpacked["metadata"]
    resource_ids = inputs.get("resource_ids") or []
    out: dict[str, Any] = {
        "id": row.id,
        "problem_type": row.problem_type,
        "status": row.status,
        "objective_value": _finite(row.objective_value),
        "solve_time_ms": _finite(row.solve_time_ms),
        "fallback_used": bool(row.fallback_used),
        "solver": row.solver or None,
        "iterations": unpacked["iterations"],
        "gap": unpacked["gap"],
        "created_at": _iso(row.created_at),
        "started_at": _iso(row.created_at),
        "finished_at": None if row.status == "running" else _iso(row.updated_at),
        "resource_id": resource_ids[0] if len(resource_ids) == 1 else None,
        "total_cost": _finite(metadata.get("total_cost")),
    }
    if include_details:
        out["inputs"] = inputs
        out["solution"] = unpacked["solution"]
        out["metadata"] = metadata
    safe: dict[str, Any] = _json_safe(out)
    return safe


__all__ = [
    "FAILED_STATUSES",
    "execute_dispatch",
    "finish_run",
    "load_fleet_assets",
    "publish_optimization_event",
    "resource_to_asset",
    "run_to_dict",
    "start_run",
    "unpack_solution",
]
