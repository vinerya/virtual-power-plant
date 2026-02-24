"""V2G (Vehicle-to-Grid) API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Any

from vpp.auth.security import get_current_user, require_role
from vpp.v2g.models import EVBattery, EVFleet, EVConnectionState
from vpp.v2g.scheduler import V2GScheduler
from vpp.v2g.aggregator import V2GAggregator, DispatchSignal, GridService

router = APIRouter(prefix="/api/v1/v2g", tags=["v2g"])

# Module-level fleet + aggregator (wired in app.py lifespan)
_fleet = EVFleet()
_scheduler = V2GScheduler()
_aggregator = V2GAggregator(_fleet, _scheduler)


def get_fleet() -> EVFleet:
    return _fleet


def get_aggregator() -> V2GAggregator:
    return _aggregator


# -- Schemas -----------------------------------------------------------------

class EVCreate(BaseModel):
    ev_id: str | None = None
    capacity_kwh: float = 60.0
    current_soc: float = Field(0.5, ge=0, le=1)
    min_soc: float = Field(0.2, ge=0, le=1)
    max_charge_kw: float = 11.0
    max_discharge_kw: float = 11.0
    v2g_capable: bool = True
    target_soc: float = Field(0.8, ge=0, le=1)
    departure_time: float | None = None
    vehicle_make: str = ""
    vehicle_model: str = ""


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


class ScheduleRequest(BaseModel):
    ev_ids: list[str] | None = None  # None = whole fleet
    prices: list[float] | None = None
    time_horizon_hours: float = 24.0
    use_optimiser: bool = False


class DispatchRequest(BaseModel):
    target_power_kw: float
    duration_seconds: int = 900
    service: str = "energy_arbitrage"


class BidRequest(BaseModel):
    service: str = "frequency_regulation"
    capacity_fraction: float = Field(0.8, gt=0, le=1)
    duration_hours: float = 1.0
    price_per_kw: float = 0.05


# -- Endpoints ---------------------------------------------------------------

@router.post("/vehicles", response_model=EVResponse, status_code=status.HTTP_201_CREATED)
async def add_vehicle(
    body: EVCreate,
    _user=Depends(require_role("admin", "operator")),
    fleet: EVFleet = Depends(get_fleet),
):
    """Register an EV in the fleet."""
    ev = EVBattery(
        capacity_kwh=body.capacity_kwh,
        current_soc=body.current_soc,
        min_soc=body.min_soc,
        max_charge_kw=body.max_charge_kw,
        max_discharge_kw=body.max_discharge_kw,
        v2g_capable=body.v2g_capable,
        target_soc=body.target_soc,
        departure_time=body.departure_time,
        vehicle_make=body.vehicle_make,
        vehicle_model=body.vehicle_model,
        connection_state=EVConnectionState.CONNECTED_IDLE,
    )
    if body.ev_id:
        ev.ev_id = body.ev_id
    fleet.add_vehicle(ev)
    return _ev_to_response(ev)


@router.get("/vehicles", response_model=list[EVResponse])
async def list_vehicles(
    _user=Depends(get_current_user),
    fleet: EVFleet = Depends(get_fleet),
):
    """List all EVs in the fleet."""
    return [_ev_to_response(ev) for ev in fleet.vehicles.values()]


@router.get("/vehicles/{ev_id}", response_model=EVResponse)
async def get_vehicle(
    ev_id: str,
    _user=Depends(get_current_user),
    fleet: EVFleet = Depends(get_fleet),
):
    """Get details of a specific EV."""
    ev = fleet.get_vehicle(ev_id)
    if ev is None:
        raise HTTPException(status_code=404, detail="EV not found")
    return _ev_to_response(ev)


@router.delete("/vehicles/{ev_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_vehicle(
    ev_id: str,
    _user=Depends(require_role("admin", "operator")),
    fleet: EVFleet = Depends(get_fleet),
):
    """Remove an EV from the fleet."""
    fleet.remove_vehicle(ev_id)


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
    _user=Depends(require_role("admin", "operator")),
    fleet: EVFleet = Depends(get_fleet),
):
    """Generate a V2G schedule for the fleet or selected EVs."""
    scheduler = V2GScheduler()

    if body.ev_ids:
        # Schedule specific EVs
        sub_fleet = EVFleet()
        for eid in body.ev_ids:
            ev = fleet.get_vehicle(eid)
            if ev is not None:
                sub_fleet.add_vehicle(ev)
        target = sub_fleet
    else:
        target = fleet

    result = scheduler.schedule_fleet(
        target,
        prices=body.prices,
        time_horizon_hours=body.time_horizon_hours,
        use_optimiser=body.use_optimiser,
    )
    return result.to_dict()


@router.post("/dispatch")
async def dispatch_signal(
    body: DispatchRequest,
    _user=Depends(require_role("admin", "operator")),
    aggregator: V2GAggregator = Depends(get_aggregator),
):
    """Dispatch a V2G signal across the fleet."""
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
    return result.to_dict()


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
        raise HTTPException(status_code=400, detail=f"Unknown service: {body.service}")

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
    """List active flexibility bids."""
    return [b.to_dict() for b in aggregator.get_active_bids()]


@router.get("/metrics")
async def aggregator_metrics(
    _user=Depends(get_current_user),
    aggregator: V2GAggregator = Depends(get_aggregator),
):
    """Get V2G aggregator performance metrics."""
    return aggregator.get_metrics()


# -- Helpers -----------------------------------------------------------------

def _ev_to_response(ev: EVBattery) -> EVResponse:
    return EVResponse(
        ev_id=ev.ev_id,
        capacity_kwh=ev.capacity_kwh,
        current_soc=ev.current_soc,
        min_soc=ev.min_soc,
        max_charge_kw=ev.max_charge_kw,
        max_discharge_kw=ev.max_discharge_kw,
        v2g_capable=ev.v2g_capable,
        connection_state=ev.connection_state.value,
        target_soc=ev.target_soc,
        energy_needed_kwh=round(ev.energy_needed_kwh, 2),
        available_discharge_kwh=round(ev.available_discharge_kwh, 2),
        has_flexibility=ev.has_flexibility,
    )
