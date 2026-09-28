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
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from vpp.db.models import BatteryStateModel, OptimizationRunModel, ResourceModel
from vpp.db.repositories import OptimizationRepository
from vpp.events import Event, EventType, get_event_bus
from vpp.optimization.planning import FleetAsset

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Marker for the wrapped solution_json layout written by this module. Rows
# written by older code store the bare solution dict.
_SOLUTION_LAYOUT_VERSION = 2
# C/4 heuristic used when a battery has no recorded energy capacity, matching
# vpp.degradation.telemetry.DEFAULT_C_RATE_HOURS.
_DEFAULT_C_RATE_HOURS = 4.0


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
    stmt = (
        select(BatteryStateModel.soc)
        .where(BatteryStateModel.resource_id == resource_id)
        .order_by(BatteryStateModel.timestamp.desc())
        .limit(1)
    )
    val = (await session.execute(stmt)).scalar_one_or_none()
    return None if val is None else float(val) / 100.0


def _fraction(v: float | None) -> float | None:
    """Accept SOC as a fraction (0..1) or a percentage (1..100)."""
    if v is None:
        return None
    return v / 100.0 if v > 1.0 else v


async def resource_to_asset(session: AsyncSession, row: ResourceModel) -> FleetAsset:
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
        ):
            v = _first_number(cfg.get(key), meta.get(key))
            if v is not None:
                setattr(asset, attr, v)
        if not (0.0 < asset.eta_charge <= 1.0 and 0.0 < asset.eta_discharge <= 1.0):
            asset.eta_charge = asset.eta_discharge = 0.95
        if not (0.0 <= asset.soc_min < asset.soc_max <= 1.0):
            asset.soc_min, asset.soc_max = 0.05, 0.95
    else:
        avail = _first_number(meta.get("available_kw"), cfg.get("available_kw"))
        if avail is not None:
            asset.available_kw, asset.availability_basis = avail, "reported"
        elif (row.current_power or 0.0) > 0:
            asset.available_kw, asset.availability_basis = float(row.current_power), "telemetry"
        else:
            asset.available_kw, asset.availability_basis = asset.rated_power_kw, "nameplate"
    return asset


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
    assets = [await resource_to_asset(session, r) for r in rows]
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
    return _json_safe(out)


__all__ = [
    "FAILED_STATUSES",
    "finish_run",
    "load_fleet_assets",
    "publish_optimization_event",
    "resource_to_asset",
    "run_to_dict",
    "start_run",
    "unpack_solution",
]
