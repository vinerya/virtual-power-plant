"""Database persistence for the V2G fleet.

The ``v2g_vehicles`` table is the source of truth for the fleet; the
in-memory :class:`~vpp.v2g.models.EVFleet` / :class:`EVBattery` objects the
scheduler and aggregator work on are built from it per request
(:func:`load_fleet`), so the fleet survives restarts and is shared by every
API worker process.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from vpp.db.models import V2GChargingSessionModel, V2GScheduleModel, V2GVehicleModel
from vpp.v2g.models import EVBattery, EVConnectionState, EVFleet

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

# Assets the DR orchestrator builds from vehicles are namespaced so they can
# never collide with ``resources`` ids in a dispatch run.
EV_ASSET_PREFIX = "ev:"


def utc(dt: datetime | None) -> datetime | None:
    """Treat naive datetimes (SQLite) as UTC."""
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def ts_to_dt(ts: float | None) -> datetime | None:
    if ts is None or not math.isfinite(ts):
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def dt_to_ts(dt: datetime | None) -> float | None:
    dt = utc(dt)
    return None if dt is None else dt.timestamp()


def loads(raw: str | None, default: Any = None) -> Any:
    try:
        return json.loads(raw) if raw else default
    except (TypeError, ValueError):
        return default


def row_to_ev(row: V2GVehicleModel) -> EVBattery:
    """Build the scheduler/aggregator view of a persisted vehicle."""
    try:
        state = EVConnectionState(row.connection_state)
    except ValueError:
        state = EVConnectionState.DISCONNECTED
    return EVBattery(
        ev_id=row.id,
        capacity_kwh=row.capacity_kwh,
        current_soc=row.current_soc,
        min_soc=row.min_soc,
        max_charge_kw=row.max_charge_kw,
        max_discharge_kw=row.max_discharge_kw,
        charge_efficiency=row.charge_efficiency,
        discharge_efficiency=row.discharge_efficiency,
        v2g_capable=row.v2g_capable,
        degradation_cost_per_kwh=row.degradation_cost_per_kwh,
        connection_state=state,
        connected_at=dt_to_ts(row.connected_at),
        target_soc=row.target_soc,
        departure_time=dt_to_ts(row.departure_time),
        vehicle_make=row.vehicle_make or "",
        vehicle_model=row.vehicle_model or "",
        owner_id=row.owner_id or "",
    )


def vehicle_to_dict(row: V2GVehicleModel) -> dict[str, Any]:
    """API representation: the legacy ``EVResponse`` fields plus binding/live state."""
    ev = row_to_ev(row)
    return {
        "ev_id": row.id,
        "name": row.name or "",
        "capacity_kwh": ev.capacity_kwh,
        "current_soc": ev.current_soc,
        "min_soc": ev.min_soc,
        "max_charge_kw": ev.max_charge_kw,
        "max_discharge_kw": ev.max_discharge_kw,
        "v2g_capable": ev.v2g_capable,
        "connection_state": ev.connection_state.value,
        "target_soc": ev.target_soc,
        "energy_needed_kwh": round(ev.energy_needed_kwh, 2),
        "available_discharge_kwh": round(ev.available_discharge_kwh, 2),
        "has_flexibility": ev.has_flexibility,
        "departure_time": ev.departure_time,
        "vehicle_make": ev.vehicle_make,
        "vehicle_model": ev.vehicle_model,
        "owner_id": ev.owner_id,
        "id_tag": row.id_tag,
        "charge_point_id": row.charge_point_id,
        "connector_id": row.connector_id,
        "binding_source": row.binding_source,
        "active_transaction_id": row.active_transaction_id,
        "charger_status": row.charger_status,
        "current_power_kw": row.current_power_kw,
        "soc_source": row.soc_source,
        "soc_updated_at": dt_to_ts(row.soc_updated_at),
        "metadata": loads(row.metadata_json, {}) or {},
    }


class V2GRepository:
    """Async data access for the V2G tables."""

    @staticmethod
    async def list_vehicles(session: AsyncSession) -> list[V2GVehicleModel]:
        stmt = select(V2GVehicleModel).order_by(V2GVehicleModel.created_at.asc())
        return list((await session.execute(stmt)).scalars().all())

    @staticmethod
    async def get(session: AsyncSession, ev_id: str) -> V2GVehicleModel | None:
        row: V2GVehicleModel | None = await session.get(V2GVehicleModel, ev_id)
        return row

    @staticmethod
    async def by_id_tag(session: AsyncSession, id_tag: str) -> V2GVehicleModel | None:
        stmt = select(V2GVehicleModel).where(V2GVehicleModel.id_tag == id_tag)
        found: V2GVehicleModel | None = (await session.execute(stmt)).scalar_one_or_none()
        return found

    @staticmethod
    async def by_connector(
        session: AsyncSession, charge_point_id: str, connector_id: int
    ) -> V2GVehicleModel | None:
        stmt = select(V2GVehicleModel).where(
            V2GVehicleModel.charge_point_id == charge_point_id,
            V2GVehicleModel.connector_id == connector_id,
        )
        found: V2GVehicleModel | None = (await session.execute(stmt)).scalar_one_or_none()
        return found

    @staticmethod
    async def by_transaction(
        session: AsyncSession, charge_point_id: str, transaction_id: int
    ) -> V2GVehicleModel | None:
        stmt = select(V2GVehicleModel).where(
            V2GVehicleModel.charge_point_id == charge_point_id,
            V2GVehicleModel.active_transaction_id == transaction_id,
        )
        found: V2GVehicleModel | None = (await session.execute(stmt)).scalars().first()
        return found

    @staticmethod
    async def open_session(
        session: AsyncSession, charge_point_id: str, transaction_id: int
    ) -> V2GChargingSessionModel | None:
        stmt = (
            select(V2GChargingSessionModel)
            .where(
                V2GChargingSessionModel.charge_point_id == charge_point_id,
                V2GChargingSessionModel.transaction_id == transaction_id,
            )
            .order_by(V2GChargingSessionModel.created_at.desc())
        )
        row: V2GChargingSessionModel | None = (await session.execute(stmt)).scalars().first()
        return row

    @staticmethod
    async def list_sessions(
        session: AsyncSession, *, vehicle_id: str | None = None, limit: int = 100
    ) -> list[V2GChargingSessionModel]:
        stmt = select(V2GChargingSessionModel).order_by(V2GChargingSessionModel.created_at.desc())
        if vehicle_id is not None:
            stmt = stmt.where(V2GChargingSessionModel.vehicle_id == vehicle_id)
        return list((await session.execute(stmt.limit(limit))).scalars().all())

    @staticmethod
    async def record_schedule(
        session: AsyncSession,
        *,
        kind: str,
        method: str,
        created_by: str | None,
        vehicle_count: int,
        total_cost: float = 0.0,
        total_revenue: float = 0.0,
        parameters: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        deliveries: list[dict[str, Any]] | None = None,
    ) -> V2GScheduleModel:
        row = V2GScheduleModel(
            created_at=datetime.now(timezone.utc),
            kind=kind,
            method=method[:32],
            created_by=created_by,
            vehicle_count=vehicle_count,
            total_cost=float(total_cost),
            total_revenue=float(total_revenue),
            parameters_json=json.dumps(parameters or {}, default=str),
            result_json=json.dumps(result or {}, default=str),
            deliveries_json=json.dumps(deliveries or [], default=str),
        )
        session.add(row)
        await session.flush()
        return row

    @staticmethod
    async def list_schedules(session: AsyncSession, limit: int = 50) -> list[V2GScheduleModel]:
        stmt = select(V2GScheduleModel).order_by(V2GScheduleModel.created_at.desc()).limit(limit)
        return list((await session.execute(stmt)).scalars().all())


async def load_fleet(session: AsyncSession, ev_ids: list[str] | None = None) -> EVFleet:
    """Build an :class:`EVFleet` from the database (optionally a subset)."""
    fleet = EVFleet(fleet_id="default", name="default_fleet")
    wanted = set(ev_ids) if ev_ids else None
    for row in await V2GRepository.list_vehicles(session):
        if wanted is None or row.id in wanted:
            fleet.add_vehicle(row_to_ev(row))
    return fleet


def session_to_dict(row: V2GChargingSessionModel) -> dict[str, Any]:
    return {
        "id": row.id,
        "vehicle_id": row.vehicle_id,
        "charge_point_id": row.charge_point_id,
        "connector_id": row.connector_id,
        "transaction_id": row.transaction_id,
        "id_tag": row.id_tag,
        "status": row.status,
        "started_at": dt_to_ts(row.started_at),
        "stopped_at": dt_to_ts(row.stopped_at),
        "meter_start_wh": row.meter_start_wh,
        "meter_stop_wh": row.meter_stop_wh,
        "energy_kwh": row.energy_kwh,
        "stop_reason": row.stop_reason,
    }


def schedule_to_dict(row: V2GScheduleModel) -> dict[str, Any]:
    return {
        "schedule_id": row.id,
        "kind": row.kind,
        "method": row.method,
        "created_by": row.created_by,
        "created_at": dt_to_ts(row.created_at),
        "vehicle_count": row.vehicle_count,
        "total_cost": row.total_cost,
        "total_revenue": row.total_revenue,
        "parameters": loads(row.parameters_json, {}),
        "deliveries": loads(row.deliveries_json, []),
    }


__all__ = [
    "EV_ASSET_PREFIX",
    "V2GRepository",
    "dt_to_ts",
    "load_fleet",
    "row_to_ev",
    "schedule_to_dict",
    "session_to_dict",
    "ts_to_dt",
    "utc",
    "vehicle_to_dict",
]
