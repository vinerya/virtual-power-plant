"""Battery degradation / SOH API routes (M3)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from vpp.auth.security import get_current_user, require_role
from vpp.db.engine import get_db, get_session_factory
from vpp.db.models import ResourceModel, UserModel
from vpp.db.repositories import BatteryDegradationRepository
from vpp.degradation.telemetry import DegradationUpdater, TelemetryWindow
from vpp.schemas.degradation import (
    SOHHistoryItem,
    SOHResponse,
    SOHUpdateRequest,
)

router = APIRouter(prefix="/api/v1/batteries", tags=["Degradation"])


# ---------------------------------------------------------------------------
# In-memory TTL cache for projected_eol_date / daily_efc.
# Keyed by (battery_id, last_update_ts).  Cheap to compute but FE will hit
# this endpoint frequently so we still cache 5 minutes.
# ---------------------------------------------------------------------------

_TTL_SECONDS = 300
_eol_cache: dict[tuple[str, str], tuple[float, tuple[float, datetime | None]]] = {}


def _eol_cache_get(key: tuple[str, str]) -> tuple[float, datetime | None] | None:
    entry = _eol_cache.get(key)
    if entry is None:
        return None
    inserted_at, value = entry
    now = datetime.now(timezone.utc).timestamp()
    if now - inserted_at > _TTL_SECONDS:
        _eol_cache.pop(key, None)
        return None
    return value


def _eol_cache_set(key: tuple[str, str], value: tuple[float, datetime | None]) -> None:
    _eol_cache[key] = (datetime.now(timezone.utc).timestamp(), value)


def _projected_eol(
    battery: ResourceModel, eol_capacity_fraction: float = 0.8
) -> tuple[float, datetime | None]:
    """Return (daily_efc, projected_eol_date).

    daily_efc = cumulative_throughput / capacity / age_in_days / 2
        (one EFC = a full charge + discharge of the rated energy = 2x capacity).

    projected_eol_date = today + (state_of_health - eol_threshold)
                                 / (loss_per_efc * daily_efc)

    where loss_per_efc = (1 - eol_capacity_fraction) / cycles_to_eol.
    For lack of a stored "cycles_to_eol" we fall back to the chemistry preset.
    """
    from vpp.degradation.models import LFP_PRESET, NMC_PRESET

    preset = NMC_PRESET if (battery.chemistry or "").lower() == "nmc" else LFP_PRESET
    cycles_to_eol = preset["throughput"]["cycles_to_eol"]
    loss_per_efc = (1.0 - eol_capacity_fraction) / cycles_to_eol

    capacity_kwh = float(battery.rated_power) or 1.0
    age_seconds = (
        (
            datetime.now(timezone.utc) - battery.created_at.replace(tzinfo=timezone.utc)
            if battery.created_at and battery.created_at.tzinfo is None
            else datetime.now(timezone.utc) - battery.created_at
        ).total_seconds()
        if battery.created_at
        else 0.0
    )
    age_days = max(age_seconds / 86400.0, 1.0 / 24.0)  # min 1h to avoid div-by-zero

    efc_total = battery.cumulative_throughput_kwh / (2.0 * capacity_kwh)
    daily_efc = efc_total / age_days

    projected: datetime | None = None
    soh_above_eol = battery.state_of_health - eol_capacity_fraction
    if soh_above_eol > 0 and daily_efc > 0 and loss_per_efc > 0:
        days_left = soh_above_eol / (loss_per_efc * daily_efc)
        projected = datetime.now(timezone.utc) + timedelta(days=days_left)
    return daily_efc, projected


async def _build_soh_response(battery: ResourceModel) -> SOHResponse:
    cache_key = (
        battery.id,
        battery.last_degradation_update.isoformat()
        if battery.last_degradation_update
        else "never",
    )
    cached = _eol_cache_get(cache_key)
    if cached is None:
        cached = _projected_eol(battery)
        _eol_cache_set(cache_key, cached)
    daily_efc, projected = cached
    return SOHResponse(
        battery_id=battery.id,
        state_of_health=battery.state_of_health,
        cumulative_throughput_kwh=battery.cumulative_throughput_kwh,
        last_update=battery.last_degradation_update,
        chemistry=battery.chemistry,
        daily_efc=daily_efc,
        projected_eol_date=projected,
    )


async def _load_battery(session: AsyncSession, battery_id: str) -> ResourceModel:
    obj = await session.get(ResourceModel, battery_id)
    if obj is None or obj.resource_type != "battery":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Battery not found")
    return obj


@router.get("/{battery_id}/soh", response_model=SOHResponse)
async def get_battery_soh(
    battery_id: str,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """Return current SOH snapshot, daily EFC and projected EOL date."""
    battery = await _load_battery(session, battery_id)
    return await _build_soh_response(battery)


@router.get("/{battery_id}/soh/history", response_model=list[SOHHistoryItem])
async def get_battery_soh_history(
    battery_id: str,
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(500, ge=1, le=5000),
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """Return time-series SOH samples for a battery."""
    await _load_battery(session, battery_id)
    samples = await BatteryDegradationRepository.get_soh_history(
        session, battery_id, days=days, limit=limit
    )
    return [
        SOHHistoryItem(
            timestamp=s.timestamp,
            state_of_health=s.state_of_health,
            cumulative_throughput_kwh=s.cumulative_throughput_kwh,
            loss_fraction=s.loss_fraction,
        )
        for s in samples
    ]


@router.post("/{battery_id}/soh/update", response_model=SOHResponse)
async def trigger_soh_update(
    battery_id: str,
    body: SOHUpdateRequest,
    session: AsyncSession = Depends(get_db),
    _admin: UserModel = Depends(require_role("admin")),
):
    """Admin-only: manually run the degradation updater on a SOC window."""
    if len(body.soc_trace) != len(body.timestamps):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="soc_trace and timestamps must have the same length",
        )
    await _load_battery(session, battery_id)

    updater = DegradationUpdater(session_factory=get_session_factory())
    window = TelemetryWindow(
        battery_id=battery_id,
        soc_trace=list(body.soc_trace),
        timestamps=list(body.timestamps),
        temperatures_c=body.temperatures_c,
    )
    update = await updater.apply_window(window)
    if update is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not apply degradation window",
        )

    # Re-load through the route session to return a consistent view.
    await session.refresh(await _load_battery(session, battery_id))
    battery = await _load_battery(session, battery_id)
    return await _build_soh_response(battery)
