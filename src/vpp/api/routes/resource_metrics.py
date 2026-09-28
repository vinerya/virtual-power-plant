"""Resource telemetry history (read) and ingestion (write).

API surface
-----------
- GET  /api/v1/resources/{id}/metrics     downsampled power/SOC history
- POST /api/v1/resources/{id}/telemetry   append samples + update live power (admin, operator)

``GET .../metrics`` takes either a named ``window`` (1h, 6h, 24h, 7d, 30d;
ending now) or an explicit ``start``/``end`` range (max 366 days). Samples
are averaged into fixed buckets; ``bucket_seconds`` may be given, otherwise
the smallest "nice" bucket keeping the series within ``max_points`` is
used. Customers may read metrics only for resources on sites they own
(others are 404).
"""

from __future__ import annotations

import contextlib
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import (
    AsyncSession,  # noqa: TC002 -- FastAPI resolves dependency annotations at runtime
)

from vpp.auth.security import get_current_principal, require_role
from vpp.db.engine import get_db
from vpp.db.models import ResourceModel, UserModel
from vpp.events import Event, EventType, get_event_bus
from vpp.portal.access import visible_resource
from vpp.portal.telemetry import (
    MAX_SPAN,
    WINDOWS,
    as_utc,
    choose_bucket_seconds,
    query_history,
    record_samples,
)
from vpp.schemas.auth import UserRole
from vpp.schemas.sites import ResourceMetricsResponse, TelemetryIngest, TelemetryIngestResult

router = APIRouter(prefix="/api/v1/resources", tags=["Resources"])

#: Samples stamped further than this in the future are rejected (clock skew guard).
MAX_FUTURE_SKEW = timedelta(minutes=5)
#: Only samples at least this fresh update the resource's live ``current_power``.
LIVE_STATE_MAX_AGE = timedelta(minutes=15)


@router.get("/{resource_id}/metrics", response_model=ResourceMetricsResponse, response_model_exclude_none=True)
async def get_resource_metrics(
    resource_id: str,
    window: str | None = Query(None, description="1h | 6h | 24h | 7d | 30d (default 24h)"),
    start: datetime | None = Query(None, description="Range start (ISO-8601, inclusive)"),
    end: datetime | None = Query(None, description="Range end (ISO-8601, exclusive; default now)"),
    bucket_seconds: int | None = Query(None, ge=1, le=86_400),
    max_points: int = Query(500, ge=10, le=5_000),
    session: AsyncSession = Depends(get_db),
    user: UserModel = Depends(get_current_principal),
):
    """Time-bucketed power (kW) and state-of-charge (0-1) history."""
    resource = await visible_resource(session, user, resource_id)
    if resource is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Resource not found")

    if window is not None and window not in WINDOWS:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"window must be one of {sorted(WINDOWS)}",
        )
    if window is not None and start is not None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="pass either window or start, not both")

    end_utc = as_utc(end) if end is not None else datetime.now(timezone.utc)
    if start is not None:
        start_utc = as_utc(start)
        label = "custom"
    else:
        label = window or "24h"
        start_utc = end_utc - WINDOWS[label]
        if end is not None:
            label = "custom"
    if start_utc >= end_utc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="start must be before end")
    span = end_utc - start_utc
    if span > MAX_SPAN:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="range exceeds 366 days")

    bucket = bucket_seconds or choose_bucket_seconds(span, max_points)
    if span.total_seconds() / bucket > 5_000:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="bucket_seconds too small for this range (max 5000 points)",
        )
    points = await query_history(session, resource_id, start=start_utc, end=end_utc, bucket_s=bucket)
    return {
        "resource_id": resource_id,
        "window": label,
        "start": start_utc,
        "end": end_utc,
        "bucket_seconds": bucket,
        "points": points,
    }


@router.post("/{resource_id}/telemetry", response_model=TelemetryIngestResult, status_code=status.HTTP_202_ACCEPTED)
async def ingest_resource_telemetry(
    resource_id: str,
    body: TelemetryIngest,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(require_role(UserRole.ADMIN, UserRole.OPERATOR)),
):
    """Append telemetry samples.

    The newest sample becomes the resource's live ``current_power`` only if
    it is at most 15 minutes old, so historical backfills never overwrite
    live state.
    """
    resource = await session.get(ResourceModel, resource_id)
    if resource is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Resource not found")
    now = datetime.now(timezone.utc)
    samples = []
    for s in body.samples:
        ts = as_utc(s.timestamp) if s.timestamp is not None else now
        if ts > now + MAX_FUTURE_SKEW:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"sample timestamp {ts.isoformat()} is in the future",
            )
        samples.append((ts, s.power_kw, s.state_of_charge))
    await record_samples(session, resource_id, samples, source=body.source)
    newest = max(samples, key=lambda x: x[0])
    current_power = resource.current_power
    if now - newest[0] > LIVE_STATE_MAX_AGE:
        # Historical backfill: keep the live state untouched.
        return {"resource_id": resource_id, "accepted": len(samples), "current_power": current_power}
    resource.current_power = newest[1]
    await session.flush()
    current_power = resource.current_power
    # A broadcast failure must not drop the write.
    with contextlib.suppress(Exception):
        await get_event_bus().publish(
            Event(
                event_type=EventType.RESOURCE_UPDATED,
                data={"resource_id": resource_id, "current_power_kw": current_power},
                source="api.telemetry_ingest",
            )
        )
    return {"resource_id": resource_id, "accepted": len(samples), "current_power": current_power}
