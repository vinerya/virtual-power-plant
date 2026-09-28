"""V2G (Vehicle-to-Grid) API endpoints.

The fleet lives in the database (``v2g_vehicles``): it survives restarts and
is shared by every API worker. Vehicles can be bound to an OCPP charge
point + connector (explicitly here, or automatically when a charger reports
a StartTransaction with the vehicle's ``id_tag``); bound vehicles get their
SOC / state from OCPP MeterValues and StatusNotification, and schedules /
dispatches are pushed to their chargers as OCPP SetChargingProfile, with the
per-vehicle delivery result reported in the response.

Flexibility bids and the aggregator's dispatch counters are still kept in
process memory (per worker, lost on restart).
"""

from __future__ import annotations

import json
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from vpp.api.routes.protocols import get_registry
from vpp.auth.security import get_current_user, require_role
from vpp.db.engine import get_db
from vpp.db.models import V2GVehicleModel
from vpp.events import EventType
from vpp.protocols.base import ProtocolRegistry
from vpp.settings import get_settings
from vpp.v2g.aggregator import DispatchSignal, GridService, V2GAggregator
from vpp.v2g.models import EVConnectionState, EVFleet
from vpp.v2g.ocpp_bridge import (
    DISPATCH_PROFILE_ID,
    DISPATCH_STACK_LEVEL,
    deliver_slots,
    ocpp_adapter_from,
    publish,
    setpoint_slots,
    summarize_deliveries,
)
from vpp.v2g.scheduler import V2GScheduler
from vpp.v2g.store import (
    V2GRepository,
    load_fleet,
    schedule_to_dict,
    session_to_dict,
    ts_to_dt,
    vehicle_to_dict,
)

router = APIRouter(prefix="/api/v1/v2g", tags=["v2g"])

# Bids / dispatch counters only; the fleet itself is loaded from the DB.
_aggregator = V2GAggregator(EVFleet(fleet_id="default", name="default_fleet"), V2GScheduler())


async def get_fleet(session: AsyncSession = Depends(get_db)) -> EVFleet:
    """The persisted fleet as an :class:`EVFleet` (fresh from the DB)."""
    return await load_fleet(session)


async def get_aggregator(fleet: EVFleet = Depends(get_fleet)) -> V2GAggregator:
    _aggregator.fleet = fleet
    return _aggregator


def _ocpp(registry: ProtocolRegistry):
    return ocpp_adapter_from(registry.get("ocpp"))


# -- Schemas -----------------------------------------------------------------


class EVCreate(BaseModel):
    ev_id: str | None = Field(None, min_length=1, max_length=36)
    name: str = Field("", max_length=255)
    capacity_kwh: float = Field(60.0, gt=0, le=10_000)
    current_soc: float = Field(0.5, ge=0, le=1)
    min_soc: float = Field(0.2, ge=0, lt=1)
    max_charge_kw: float = Field(11.0, ge=0, le=10_000)
    max_discharge_kw: float = Field(11.0, ge=0, le=10_000)
    charge_efficiency: float = Field(0.92, gt=0, le=1)
    discharge_efficiency: float = Field(0.92, gt=0, le=1)
    v2g_capable: bool = True
    target_soc: float = Field(0.8, ge=0, le=1)
    departure_time: float | None = None  # Unix seconds
    vehicle_make: str = Field("", max_length=128)
    vehicle_model: str = Field("", max_length=128)
    owner_id: str = Field("", max_length=64)
    connection_state: EVConnectionState = EVConnectionState.CONNECTED_IDLE
    # OCPP binding
    id_tag: str | None = Field(None, min_length=1, max_length=64)
    charge_point_id: str | None = Field(None, min_length=1, max_length=64)
    connector_id: int | None = Field(None, ge=1, le=100)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EVUpdate(BaseModel):
    name: str | None = Field(None, max_length=255)
    capacity_kwh: float | None = Field(None, gt=0, le=10_000)
    current_soc: float | None = Field(None, ge=0, le=1)
    min_soc: float | None = Field(None, ge=0, lt=1)
    max_charge_kw: float | None = Field(None, ge=0, le=10_000)
    max_discharge_kw: float | None = Field(None, ge=0, le=10_000)
    v2g_capable: bool | None = None
    target_soc: float | None = Field(None, ge=0, le=1)
    departure_time: float | None = None
    clear_departure_time: bool = False
    connection_state: EVConnectionState | None = None
    id_tag: str | None = Field(None, min_length=1, max_length=64)
    vehicle_make: str | None = Field(None, max_length=128)
    vehicle_model: str | None = Field(None, max_length=128)
    metadata: dict[str, Any] | None = None


class EVResponse(BaseModel):
    ev_id: str
    capacity_kwh: float
    current_soc: float
    min_soc: float
    max_charge_kw: float
    max_discharge_kw: float
    v2g_capable: bool
    connection_state: str
    target_soc: float
    energy_needed_kwh: float
    available_discharge_kwh: float
    has_flexibility: bool
    # Additive (persistent fleet + OCPP binding)
    name: str = ""
    departure_time: float | None = None
    vehicle_make: str = ""
    vehicle_model: str = ""
    owner_id: str = ""
    id_tag: str | None = None
    charge_point_id: str | None = None
    connector_id: int | None = None
    binding_source: str | None = None
    active_transaction_id: int | None = None
    charger_status: str | None = None
    current_power_kw: float = 0.0
    soc_source: str = "api"
    soc_updated_at: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BindingRequest(BaseModel):
    charge_point_id: str = Field(..., min_length=1, max_length=64)
    connector_id: int = Field(1, ge=1, le=100)


class ScheduleRequest(BaseModel):
    ev_ids: list[str] | None = None  # None = whole fleet
    prices: list[float] | None = None
    time_horizon_hours: float = Field(24.0, gt=0, le=168)
    use_optimiser: bool = False
    push_to_chargers: bool = True


class DispatchRequest(BaseModel):
    target_power_kw: float
    duration_seconds: int = Field(900, ge=60, le=86_400)
    service: str = "energy_arbitrage"
    push_to_chargers: bool = True


class BidRequest(BaseModel):
    service: str = "frequency_regulation"
    capacity_fraction: float = Field(0.8, gt=0, le=1)
    duration_hours: float = 1.0
    price_per_kw: float = 0.05


# -- Helpers -----------------------------------------------------------------


async def _get_or_404(session: AsyncSession, ev_id: str) -> V2GVehicleModel:
    row = await V2GRepository.get(session, ev_id)
    if row is None:
        raise HTTPException(status_code=404, detail="EV not found")
    return row


async def _commit_or_409(session: AsyncSession, detail: str) -> None:
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail=detail) from exc


def _adopt_charger_state(row: V2GVehicleModel, adapter: Any) -> None:
    """On binding, take connector status / open transaction from the live charger."""
    if adapter is None or row.charge_point_id is None:
        return
    cp = adapter.get_charge_point(row.charge_point_id)
    if cp is None:
        return
    connector_status = cp.connector_status.get(row.connector_id)
    if connector_status is not None:
        row.charger_status = connector_status.value
        if connector_status.value == "Available":
            row.connection_state = EVConnectionState.DISCONNECTED.value
        elif connector_status.value == "Charging":
            row.connection_state = EVConnectionState.CHARGING.value
        elif connector_status.value in ("Preparing", "SuspendedEV", "SuspendedEVSE", "Finishing"):
            row.connection_state = EVConnectionState.CONNECTED_IDLE.value
    for tx in adapter.list_transactions():
        if (
            tx.get("charge_point_id") == row.charge_point_id
            and tx.get("connector_id") == row.connector_id
        ):
            row.active_transaction_id = int(tx["transaction_id"])


# -- Vehicles ----------------------------------------------------------------


@router.post("/vehicles", response_model=EVResponse, status_code=status.HTTP_201_CREATED)
async def add_vehicle(
    body: EVCreate,
    _user=Depends(require_role("admin", "operator")),
    session: AsyncSession = Depends(get_db),
    registry: ProtocolRegistry = Depends(get_registry),
):
    """Register an EV in the (persistent) fleet, optionally bound to an OCPP connector."""
    if (body.charge_point_id is None) != (body.connector_id is None):
        raise HTTPException(
            status_code=422, detail="charge_point_id and connector_id must be given together"
        )
    if body.ev_id and await V2GRepository.get(session, body.ev_id) is not None:
        raise HTTPException(status_code=409, detail=f"EV {body.ev_id} already exists")
    row = V2GVehicleModel(
        name=body.name,
        capacity_kwh=body.capacity_kwh,
        current_soc=body.current_soc,
        min_soc=body.min_soc,
        target_soc=body.target_soc,
        max_charge_kw=body.max_charge_kw,
        max_discharge_kw=body.max_discharge_kw,
        charge_efficiency=body.charge_efficiency,
        discharge_efficiency=body.discharge_efficiency,
        degradation_cost_per_kwh=0.02,
        v2g_capable=body.v2g_capable,
        connection_state=body.connection_state.value,
        departure_time=ts_to_dt(body.departure_time),
        vehicle_make=body.vehicle_make,
        vehicle_model=body.vehicle_model,
        owner_id=body.owner_id,
        id_tag=body.id_tag,
        charge_point_id=body.charge_point_id,
        connector_id=body.connector_id,
        binding_source="manual" if body.charge_point_id else None,
        current_power_kw=0.0,
        soc_source="api",
        metadata_json=json.dumps(body.metadata),
    )
    if body.ev_id:
        row.id = body.ev_id
    _adopt_charger_state(row, _ocpp(registry))
    session.add(row)
    await _commit_or_409(session, "id_tag already used, or another EV is bound to that connector")
    await session.refresh(row)
    return vehicle_to_dict(row)


@router.get("/vehicles", response_model=list[EVResponse])
async def list_vehicles(
    _user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
):
    """List all EVs in the fleet."""
    return [vehicle_to_dict(r) for r in await V2GRepository.list_vehicles(session)]


@router.get("/vehicles/{ev_id}", response_model=EVResponse)
async def get_vehicle(
    ev_id: str,
    _user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
):
    """Get details of a specific EV."""
    return vehicle_to_dict(await _get_or_404(session, ev_id))


@router.patch("/vehicles/{ev_id}", response_model=EVResponse)
async def update_vehicle(
    ev_id: str,
    body: EVUpdate,
    _user=Depends(require_role("admin", "operator")),
    session: AsyncSession = Depends(get_db),
):
    """Update departure time / SOC targets / limits / metadata of an EV."""

    row = await _get_or_404(session, ev_id)
    data = body.model_dump(exclude_unset=True)
    for key in (
        "name",
        "capacity_kwh",
        "min_soc",
        "max_charge_kw",
        "max_discharge_kw",
        "v2g_capable",
        "target_soc",
        "id_tag",
        "vehicle_make",
        "vehicle_model",
    ):
        if key in data and data[key] is not None:
            setattr(row, key, data[key])
    if data.get("current_soc") is not None:
        row.current_soc = data["current_soc"]
        row.soc_source = "api"
        row.soc_updated_at = ts_to_dt(time.time())
    if data.get("departure_time") is not None:
        row.departure_time = ts_to_dt(data["departure_time"])
    elif body.clear_departure_time:
        row.departure_time = None
    if data.get("connection_state") is not None:
        row.connection_state = EVConnectionState(data["connection_state"]).value
    if data.get("metadata") is not None:
        row.metadata_json = json.dumps(data["metadata"])
    await _commit_or_409(session, "id_tag already used by another EV")
    await session.refresh(row)
    return vehicle_to_dict(row)


@router.delete("/vehicles/{ev_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_vehicle(
    ev_id: str,
    _user=Depends(require_role("admin", "operator")),
    session: AsyncSession = Depends(get_db),
):
    """Remove an EV from the fleet (idempotent)."""
    row = await V2GRepository.get(session, ev_id)
    if row is not None:
        await session.delete(row)
        await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.put("/vehicles/{ev_id}/binding", response_model=EVResponse)
async def bind_vehicle(
    ev_id: str,
    body: BindingRequest,
    _user=Depends(require_role("admin", "operator")),
    session: AsyncSession = Depends(get_db),
    registry: ProtocolRegistry = Depends(get_registry),
):
    """Bind an EV to an OCPP charge point connector (one EV per connector)."""
    row = await _get_or_404(session, ev_id)
    occupant = await V2GRepository.by_connector(session, body.charge_point_id, body.connector_id)
    if occupant is not None and occupant.id != row.id:
        raise HTTPException(
            status_code=409,
            detail=f"EV {occupant.id} is already bound to {body.charge_point_id}/"
            f"{body.connector_id}; unbind it first",
        )
    row.charge_point_id = body.charge_point_id
    row.connector_id = body.connector_id
    row.binding_source = "manual"
    row.active_transaction_id = None
    _adopt_charger_state(row, _ocpp(registry))
    await _commit_or_409(session, "another EV is bound to that connector")
    await session.refresh(row)
    return vehicle_to_dict(row)


@router.delete("/vehicles/{ev_id}/binding", response_model=EVResponse)
async def unbind_vehicle(
    ev_id: str,
    _user=Depends(require_role("admin", "operator")),
    session: AsyncSession = Depends(get_db),
):
    """Remove an EV's charge point binding."""
    row = await _get_or_404(session, ev_id)
    row.charge_point_id = None
    row.connector_id = None
    row.binding_source = None
    row.active_transaction_id = None
    row.charger_status = None
    await session.commit()
    await session.refresh(row)
    return vehicle_to_dict(row)


@router.get("/sessions")
async def list_sessions(
    ev_id: str | None = None,
    limit: int = Query(100, ge=1, le=1000),
    _user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
):
    """OCPP charging sessions (transactions), newest first."""
    rows = await V2GRepository.list_sessions(session, vehicle_id=ev_id, limit=limit)
    return [session_to_dict(r) for r in rows]


# -- Fleet -------------------------------------------------------------------


@router.get("/fleet")
async def fleet_status(
    _user=Depends(get_current_user),
    fleet: EVFleet = Depends(get_fleet),
):
    """Get aggregated fleet status."""
    return fleet.to_dict()


@router.get("/flexibility")
async def fleet_flexibility(
    _user=Depends(get_current_user),
    aggregator: V2GAggregator = Depends(get_aggregator),
):
    """Assess current fleet flexibility for grid services."""
    return aggregator.assess_flexibility()


@router.post("/schedule")
async def create_schedule(
    body: ScheduleRequest,
    user=Depends(require_role("admin", "operator")),
    session: AsyncSession = Depends(get_db),
    registry: ProtocolRegistry = Depends(get_registry),
):
    """Generate a V2G schedule and push it to the bound chargers.

    Each vehicle's slots become one absolute OCPP charging profile
    (``TxProfile`` when a transaction is running, otherwise
    ``TxDefaultProfile``); ``deliveries`` reports per vehicle what the
    charger answered (``accepted``/``rejected``/``not_supported``), or why
    nothing was sent (``not_bound``, ``not_connected``, ``no_ocpp``,
    ``no_slots``, ``skipped``) -- ``simulated`` when the OCPP adapter runs
    without a real Central System.
    """
    fleet = await load_fleet(session, body.ev_ids)
    missing = sorted(set(body.ev_ids or []) - set(fleet.vehicles))
    result = V2GScheduler().schedule_fleet(
        fleet,
        prices=body.prices,
        time_horizon_hours=body.time_horizon_hours,
        use_optimiser=body.use_optimiser,
    )
    slots_by_ev: dict[str, list[Any]] = {}
    for slot in result.schedule:
        slots_by_ev.setdefault(slot.ev_id, []).append(slot)
    rows = [r for r in await V2GRepository.list_vehicles(session) if r.id in fleet.vehicles]
    deliveries = await deliver_slots(
        _ocpp(registry),
        rows,
        slots_by_ev,
        max_periods=get_settings().v2g_max_profile_periods,
        push=body.push_to_chargers,
    )
    out = result.to_dict()
    record = await V2GRepository.record_schedule(
        session,
        kind="schedule",
        method=result.method,
        created_by=getattr(user, "id", None),
        vehicle_count=len(rows),
        total_cost=result.total_cost,
        total_revenue=result.total_revenue,
        parameters=body.model_dump(),
        result=out,
        deliveries=deliveries,
    )
    await session.commit()
    summary = summarize_deliveries(deliveries)
    await publish(
        EventType.V2G_SCHEDULE_CREATED,
        {
            "schedule_id": record.id,
            "method": result.method,
            "vehicle_count": len(rows),
            "slot_count": len(result.schedule),
            "deliveries": summary,
        },
        "api.v2g",
    )
    out.update(
        {
            "schedule_id": record.id,
            "deliveries": deliveries,
            "delivery_summary": summary,
            "missing_ev_ids": missing,
        }
    )
    return out


@router.get("/schedules")
async def list_schedules(
    limit: int = Query(50, ge=1, le=500),
    _user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
):
    """Recent schedule / dispatch records with their delivery results."""
    return [schedule_to_dict(r) for r in await V2GRepository.list_schedules(session, limit)]


@router.post("/dispatch")
async def dispatch_signal(
    body: DispatchRequest,
    user=Depends(require_role("admin", "operator")),
    aggregator: V2GAggregator = Depends(get_aggregator),
    session: AsyncSession = Depends(get_db),
    registry: ProtocolRegistry = Depends(get_registry),
):
    """Split a fleet power target (+ charge / - discharge) across plugged-in EVs.

    Each EV's share is pushed to its charger as a one-period profile for
    ``duration_seconds`` (stacked above the V2G schedule); ``deliveries``
    reports the per-vehicle outcome.
    """
    try:
        service = GridService(body.service)
    except ValueError:
        service = GridService.ENERGY_ARBITRAGE

    signal = DispatchSignal(
        target_power_kw=body.target_power_kw,
        duration_seconds=body.duration_seconds,
        service=service,
    )
    result = aggregator.dispatch(signal)
    slots = {
        ev_id: setpoint_slots(kw, body.duration_seconds)
        for ev_id, kw in result.ev_allocations.items()
    }
    rows = [r for r in await V2GRepository.list_vehicles(session) if r.id in result.ev_allocations]
    deliveries = await deliver_slots(
        _ocpp(registry),
        rows,
        slots,
        profile_id=DISPATCH_PROFILE_ID,
        stack_level=DISPATCH_STACK_LEVEL,
        push=body.push_to_chargers,
    )
    out = result.to_dict()
    out["ev_allocations"] = {k: round(v, 3) for k, v in result.ev_allocations.items()}
    record = await V2GRepository.record_schedule(
        session,
        kind="dispatch",
        method="proportional",
        created_by=getattr(user, "id", None),
        vehicle_count=len(rows),
        parameters=body.model_dump(),
        result=out,
        deliveries=deliveries,
    )
    await session.commit()
    fleet = aggregator.fleet
    await publish(
        EventType.V2G_DISPATCH,
        {
            "source": "api",
            "schedule_id": record.id,
            "dispatch_kw": result.achieved_power_kw,
            "target_power_kw": body.target_power_kw,
            "connected_evs": fleet.connected_count,
            "avg_soc": fleet.average_soc,
            "deliveries": summarize_deliveries(deliveries),
        },
        "api.v2g",
    )
    out.update(
        {
            "schedule_id": record.id,
            "deliveries": deliveries,
            "delivery_summary": summarize_deliveries(deliveries),
        }
    )
    return out


@router.post("/bid")
async def create_bid(
    body: BidRequest,
    _user=Depends(require_role("admin", "operator")),
    aggregator: V2GAggregator = Depends(get_aggregator),
):
    """Create a flexibility bid for grid services."""
    try:
        service = GridService(body.service)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown service: {body.service}") from None

    bid = aggregator.create_flexibility_bid(
        service=service,
        capacity_fraction=body.capacity_fraction,
        duration_hours=body.duration_hours,
        price_per_kw=body.price_per_kw,
    )
    if bid is None:
        raise HTTPException(status_code=409, detail="Insufficient flexibility for bid")
    return bid.to_dict()


@router.get("/bids")
async def list_bids(
    _user=Depends(get_current_user),
    aggregator: V2GAggregator = Depends(get_aggregator),
):
    """List active flexibility bids (process-local)."""
    return [b.to_dict() for b in aggregator.get_active_bids()]


@router.get("/metrics")
async def aggregator_metrics(
    _user=Depends(get_current_user),
    aggregator: V2GAggregator = Depends(get_aggregator),
):
    """Get V2G aggregator performance metrics."""
    return aggregator.get_metrics()
