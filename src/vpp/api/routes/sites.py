"""Sites: geographic grouping of resources for the map, plus site metering.

API surface
-----------
- GET    /api/v1/sites                         list (customers: own sites only)
- POST   /api/v1/sites                         create (admin, operator)
- GET    /api/v1/sites/{id}                    read (customers: own site only)
- PATCH  /api/v1/sites/{id}                    update / reassign owner / set members (admin, operator)
- DELETE /api/v1/sites/{id}                    delete; member resources are unassigned (admin)
- POST   /api/v1/sites/{id}/meter-readings     upsert revenue-meter interval data (admin, operator)
- GET    /api/v1/sites/{id}/meter-readings     read interval data (customers: own site only)

Customers get 404 (not 403) for sites they do not own so ids cannot be probed.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncSession,  # noqa: TC002 -- FastAPI resolves dependency annotations at runtime
)

from vpp.auth.security import get_current_principal, require_role
from vpp.db.engine import get_db
from vpp.db.models import MeterReadingModel, ResourceModel, SiteModel, UserModel
from vpp.portal.access import is_customer, visible_site
from vpp.portal.sites import summarize_sites
from vpp.portal.telemetry import as_utc
from vpp.schemas.auth import UserRole
from vpp.schemas.sites import (
    MeterReadingOut,
    MeterReadingsIngest,
    MeterReadingsIngestResult,
    SiteCreate,
    SiteResponse,
    SiteUpdate,
)

router = APIRouter(prefix="/api/v1/sites", tags=["Sites"])

_writer = require_role(UserRole.ADMIN, UserRole.OPERATOR)


async def _validate_owner(session: AsyncSession, owner_id: str | None) -> None:
    if owner_id is None:
        return
    owner = await session.get(UserModel, owner_id)
    if owner is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"owner {owner_id!r} not found")
    if owner.role != UserRole.CUSTOMER.value:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="site owners must have the 'customer' role",
        )


async def _assign_resources(
    session: AsyncSession, site_id: str, resource_ids: list[str]
) -> None:
    """Make ``resource_ids`` exactly the members of ``site_id``.

    Resources already assigned to a *different* site are rejected (409)
    rather than silently moved, since that would change what another
    customer sees and bills against.
    """
    wanted = list(dict.fromkeys(resource_ids))
    rows = list(
        (await session.execute(select(ResourceModel).where(ResourceModel.id.in_(wanted)))).scalars().all()
    ) if wanted else []
    found = {r.id: r for r in rows}
    missing = [rid for rid in wanted if rid not in found]
    if missing:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"unknown resource ids: {missing}")
    taken = [r.id for r in rows if r.site_id not in (None, site_id)]
    if taken:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=f"resources already assigned to another site: {taken}",
        )
    current = (
        await session.execute(select(ResourceModel).where(ResourceModel.site_id == site_id))
    ).scalars().all()
    for r in current:
        if r.id not in found:
            r.site_id = None
    for r in rows:
        r.site_id = site_id
    await session.flush()


async def _one(session: AsyncSession, site: SiteModel) -> dict:
    await session.refresh(site)
    return (await summarize_sites(session, [site]))[0]


@router.get("", response_model=list[SiteResponse])
@router.get("/", response_model=list[SiteResponse], include_in_schema=False)
async def list_sites(
    region: str | None = None,
    session: AsyncSession = Depends(get_db),
    user: UserModel = Depends(get_current_principal),
):
    """List sites with live aggregates (power, capacity, SOC, health)."""
    stmt = select(SiteModel).order_by(SiteModel.name, SiteModel.id)
    if is_customer(user):
        stmt = stmt.where(SiteModel.owner_id == user.id)
    if region is not None:
        stmt = stmt.where(SiteModel.region == region)
    sites = list((await session.execute(stmt)).scalars().all())
    return await summarize_sites(session, sites)


@router.post("", response_model=SiteResponse, status_code=status.HTTP_201_CREATED)
@router.post("/", response_model=SiteResponse, status_code=status.HTTP_201_CREATED, include_in_schema=False)
async def create_site(
    body: SiteCreate,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(_writer),
):
    await _validate_owner(session, body.owner_id)
    site = SiteModel(
        name=body.name,
        lat=body.lat,
        lon=body.lon,
        region=body.region,
        address=body.address,
        timezone=body.timezone,
        owner_id=body.owner_id,
        metadata_json=json.dumps(body.metadata),
    )
    session.add(site)
    await session.flush()
    if body.resource_ids:
        await _assign_resources(session, site.id, body.resource_ids)
    return await _one(session, site)


@router.get("/{site_id}", response_model=SiteResponse)
async def get_site(
    site_id: str,
    session: AsyncSession = Depends(get_db),
    user: UserModel = Depends(get_current_principal),
):
    site = await visible_site(session, user, site_id)
    if site is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Site not found")
    return (await summarize_sites(session, [site]))[0]


@router.patch("/{site_id}", response_model=SiteResponse)
async def update_site(
    site_id: str,
    body: SiteUpdate,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(_writer),
):
    site = await session.get(SiteModel, site_id)
    if site is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Site not found")
    fields = body.model_fields_set
    if "owner_id" in fields:
        await _validate_owner(session, body.owner_id)
        site.owner_id = body.owner_id
    for name in ("name", "lat", "lon", "region", "address", "timezone"):
        if name in fields:
            value = getattr(body, name)
            if value is None and name in ("name", "lat", "lon", "timezone"):
                raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"{name} cannot be null")
            setattr(site, name, value)
    if "metadata" in fields:
        site.metadata_json = json.dumps(body.metadata or {})
    if body.resource_ids is not None:
        await _assign_resources(session, site.id, body.resource_ids)
    site.updated_at = datetime.now(timezone.utc)
    await session.flush()
    return await _one(session, site)


@router.delete("/{site_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_site(
    site_id: str,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(require_role(UserRole.ADMIN)),
):
    site = await session.get(SiteModel, site_id)
    if site is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Site not found")
    # Explicit rather than relying on FK actions (SQLite doesn't enforce them
    # unless PRAGMA foreign_keys is on).
    for r in (
        await session.execute(select(ResourceModel).where(ResourceModel.site_id == site_id))
    ).scalars().all():
        r.site_id = None
    for m in (
        await session.execute(select(MeterReadingModel).where(MeterReadingModel.site_id == site_id))
    ).scalars().all():
        await session.delete(m)
    await session.delete(site)
    await session.flush()


@router.post("/{site_id}/meter-readings", response_model=MeterReadingsIngestResult)
async def ingest_meter_readings(
    site_id: str,
    body: MeterReadingsIngest,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(_writer),
):
    """Upsert interval data keyed by (site, interval start).

    Re-sending an interval overwrites it (meter data gets corrected). All
    readings for a site must share one interval length; timestamps must be
    aligned to it.
    """
    site = await session.get(SiteModel, site_id)
    if site is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Site not found")

    existing_iv = (
        await session.execute(
            select(MeterReadingModel.interval_minutes)
            .where(MeterReadingModel.site_id == site_id)
            .limit(1)
        )
    ).scalar_one_or_none()
    if existing_iv is not None and existing_iv != body.interval_minutes:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=f"site already has {existing_iv}-minute interval data; "
                   f"cannot mix with {body.interval_minutes}-minute readings",
        )

    step = body.interval_minutes * 60
    by_ts: dict[datetime, tuple[float, float]] = {}
    for rd in body.readings:
        ts = as_utc(rd.timestamp)
        if int(ts.timestamp()) % step or ts.microsecond:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"timestamp {ts.isoformat()} is not aligned to {body.interval_minutes} minutes",
            )
        by_ts[ts] = (rd.import_kwh, rd.export_kwh)

    lo, hi = min(by_ts), max(by_ts)
    current = {
        as_utc(m.timestamp): m
        for m in (
            await session.execute(
                select(MeterReadingModel).where(
                    MeterReadingModel.site_id == site_id,
                    MeterReadingModel.timestamp >= lo,
                    MeterReadingModel.timestamp <= hi,
                )
            )
        ).scalars().all()
    }
    inserted = updated = 0
    for ts, (imp, exp) in by_ts.items():
        row = current.get(ts)
        if row is None:
            session.add(MeterReadingModel(
                site_id=site_id, timestamp=ts, interval_minutes=body.interval_minutes,
                import_kwh=imp, export_kwh=exp,
            ))
            inserted += 1
        else:
            row.import_kwh, row.export_kwh = imp, exp
            updated += 1
    await session.flush()
    return {"site_id": site_id, "received": len(body.readings), "inserted": inserted, "updated": updated}


@router.get("/{site_id}/meter-readings", response_model=list[MeterReadingOut])
async def list_meter_readings(
    site_id: str,
    start: datetime | None = Query(None, description="Inclusive; default end - 7 days"),
    end: datetime | None = Query(None, description="Exclusive; default now"),
    limit: int = Query(5000, ge=1, le=50_000),
    session: AsyncSession = Depends(get_db),
    user: UserModel = Depends(get_current_principal),
):
    site = await visible_site(session, user, site_id)
    if site is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Site not found")
    end_utc = as_utc(end) if end else datetime.now(timezone.utc)
    start_utc = as_utc(start) if start else end_utc - timedelta(days=7)
    if start_utc >= end_utc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="start must be before end")
    rows = (
        await session.execute(
            select(MeterReadingModel)
            .where(
                MeterReadingModel.site_id == site_id,
                MeterReadingModel.timestamp >= start_utc,
                MeterReadingModel.timestamp < end_utc,
            )
            .order_by(MeterReadingModel.timestamp)
            .limit(limit)
        )
    ).scalars().all()
    return [
        {
            "timestamp": as_utc(m.timestamp),
            "interval_minutes": m.interval_minutes,
            "import_kwh": m.import_kwh,
            "export_kwh": m.export_kwh,
        }
        for m in rows
    ]
