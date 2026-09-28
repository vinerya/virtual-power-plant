"""Customer (member) portal: the logged-in customer's own account data.

API surface (``customer`` role only; every response is scoped to the caller)
------------------------------------------------------------------------------
- GET    /api/v1/customer/me                      profile
- GET    /api/v1/customer/me/bill?month=YYYY-MM   tariff-engine bill from own meter data
- GET    /api/v1/customer/me/devices              resources on own sites
- GET    /api/v1/customer/programs                active DR programs (+ ``enrolled`` flag)
- POST   /api/v1/customer/enrollments             enroll in programs (explicit consent)
- DELETE /api/v1/customer/enrollments/{program}   leave a program

Operator accounts get 403 here; operators inspect customers through
``/api/v1/customers``.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
)

from vpp.auth.security import require_role
from vpp.db.engine import get_db
from vpp.db.models import (
    CustomerProfileModel,
    DRProgramModel,
    ProgramEnrollmentModel,
    ResourceModel,
    UserModel,
)
from vpp.portal.access import owned_resources, owned_sites
from vpp.portal.billing import BillingError, compute_customer_bill
from vpp.portal.telemetry import latest_soc
from vpp.schemas.auth import UserRole
from vpp.schemas.customer import (
    CustomerBillResponse,
    CustomerDevice,
    CustomerResponse,
    DRProgramResponse,
    EnrollmentRequest,
    EnrollmentResponse,
)

router = APIRouter(prefix="/api/v1/customer", tags=["Customer portal"])

_customer = require_role(UserRole.CUSTOMER)

#: |power| below this (kW) is reported as idle.
_IDLE_KW = 0.05


async def customer_payload(session: AsyncSession, user: UserModel) -> dict:
    profile = (
        await session.execute(
            select(CustomerProfileModel).where(CustomerProfileModel.user_id == user.id)
        )
    ).scalar_one_or_none()
    sites = await owned_sites(session, user.id)
    return {
        "id": user.id,
        "username": user.username,
        "name": profile.name if profile else user.username,
        "email": profile.email if profile else None,
        "address": profile.address if profile else None,
        "tariff_id": profile.tariff_id if profile else None,
        "baseline_kwh_per_month": profile.baseline_kwh_per_month if profile else None,
        "site_ids": [s.id for s in sites],
        "is_active": user.is_active,
    }


def device_state(r: ResourceModel) -> str:
    """Human state from live power. Sign convention: + charging/consuming, - discharging."""
    if not r.online:
        return "offline"
    p = r.current_power or 0.0
    if abs(p) < _IDLE_KW:
        return "idle"
    if r.resource_type == "battery":
        return "charging" if p > 0 else "discharging"
    if r.resource_type in ("solar", "wind_turbine", "wind"):
        return "generating"
    return "consuming" if p > 0 else "exporting"


async def devices_for(session: AsyncSession, user_id: str) -> list[dict]:
    resources = await owned_resources(session, user_id)
    socs = await latest_soc(session, [r.id for r in resources if r.resource_type == "battery"])
    return [
        {
            "id": r.id,
            "kind": r.resource_type,
            "name": r.name,
            "state": device_state(r),
            "current_power": r.current_power or 0.0,
            "state_of_charge": socs.get(r.id),
            "online": r.online,
            "rated_power": r.rated_power,
            "site_id": r.site_id,
        }
        for r in resources
    ]


def raise_billing(exc: BillingError) -> None:
    raise HTTPException(status_code=exc.status, detail=exc.detail) from exc


@router.get("/me", response_model=CustomerResponse)
async def get_me(
    session: AsyncSession = Depends(get_db),
    user: UserModel = Depends(_customer),
):
    return await customer_payload(session, user)


@router.get("/me/bill", response_model=CustomerBillResponse)
async def get_my_bill(
    month: str | None = Query(None, description="YYYY-MM; default: current month"),
    session: AsyncSession = Depends(get_db),
    user: UserModel = Depends(_customer),
):
    """This month's bill from the customer's own meter data and assigned tariff.

    409 when no tariff is assigned (the portal must not invent a bill).
    """
    try:
        return await compute_customer_bill(session, user.id, month)
    except BillingError as exc:
        raise_billing(exc)


@router.get("/me/devices", response_model=list[CustomerDevice], response_model_exclude_none=True)
async def get_my_devices(
    session: AsyncSession = Depends(get_db),
    user: UserModel = Depends(_customer),
):
    return await devices_for(session, user.id)


@router.get("/programs", response_model=list[DRProgramResponse], response_model_exclude_none=True)
async def list_programs(
    session: AsyncSession = Depends(get_db),
    user: UserModel = Depends(_customer),
):
    """Active programs, flagged with whether the caller is enrolled."""
    programs = (
        (
            await session.execute(
                select(DRProgramModel)
                .where(DRProgramModel.active.is_(True))
                .order_by(DRProgramModel.name)
            )
        )
        .scalars()
        .all()
    )
    enrolled = set(
        (
            await session.execute(
                select(ProgramEnrollmentModel.program_id).where(
                    ProgramEnrollmentModel.user_id == user.id
                )
            )
        )
        .scalars()
        .all()
    )
    return [
        {
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "utility": p.utility,
            "incentive_per_event": p.incentive_per_event,
            "active": p.active,
            "enrolled": p.id in enrolled,
        }
        for p in programs
    ]


@router.post(
    "/enrollments", response_model=EnrollmentResponse, status_code=status.HTTP_201_CREATED
)
async def enroll(
    body: EnrollmentRequest,
    session: AsyncSession = Depends(get_db),
    user: UserModel = Depends(_customer),
):
    """Enroll the caller's devices in one or more DR programs.

    Requires explicit consent (``acknowledged: true``, 422 otherwise), at
    least one device on an owned site (409 otherwise -- there is nothing to
    dispatch), and active program ids (422 for unknown/inactive ones).
    Re-enrolling is idempotent.
    """
    if not body.acknowledged:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Program terms must be acknowledged"
        )
    wanted = list(dict.fromkeys(body.program_ids))
    programs = {
        p.id: p
        for p in (
            await session.execute(select(DRProgramModel).where(DRProgramModel.id.in_(wanted)))
        )
        .scalars()
        .all()
    }
    bad = [pid for pid in wanted if pid not in programs or not programs[pid].active]
    if bad:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"unknown or inactive programs: {bad}"
        )
    devices = await owned_resources(session, user.id)
    if not devices:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail="No devices are linked to your account yet; contact your operator",
        )
    existing = set(
        (
            await session.execute(
                select(ProgramEnrollmentModel.program_id).where(
                    ProgramEnrollmentModel.user_id == user.id
                )
            )
        )
        .scalars()
        .all()
    )
    now = datetime.now(timezone.utc)
    for pid in wanted:
        if pid not in existing:
            session.add(
                ProgramEnrollmentModel(user_id=user.id, program_id=pid, acknowledged_at=now)
            )
    await session.flush()
    return {
        "ok": True,
        "enrolled": sorted(existing | set(wanted)),
        "device_ids": [d.id for d in devices],
    }


@router.delete("/enrollments/{program_id}", status_code=status.HTTP_204_NO_CONTENT)
async def unenroll(
    program_id: str,
    session: AsyncSession = Depends(get_db),
    user: UserModel = Depends(_customer),
):
    result = await session.execute(
        delete(ProgramEnrollmentModel).where(
            ProgramEnrollmentModel.user_id == user.id,
            ProgramEnrollmentModel.program_id == program_id,
        )
    )
    if result.rowcount == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Not enrolled in this program")
