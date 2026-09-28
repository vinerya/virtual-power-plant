"""Operator-side administration of customers and demand-response programs.

API surface
-----------
- POST  /api/v1/customers                 onboard a customer (user + profile) (admin)
- GET   /api/v1/customers                 list customers (admin, operator)
- GET   /api/v1/customers/{id}            read (admin, operator)
- PATCH /api/v1/customers/{id}            update profile / tariff (admin)
- GET   /api/v1/customers/{id}/bill       same bill the customer sees (admin, operator)
- GET   /api/v1/customers/{id}/devices    same devices the customer sees (admin, operator)
- GET   /api/v1/programs                  all programs incl. inactive + enrollment counts (admin, operator)
- POST  /api/v1/programs                  create (admin)
- PATCH /api/v1/programs/{id}             update / deactivate (admin)

Linking a customer to their premise and devices is done by assigning the
customer as a site's ``owner_id`` (``POST/PATCH /api/v1/sites``).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import (
    AsyncSession,  # noqa: TC002 -- FastAPI resolves dependency annotations at runtime
)

from vpp.api.routes.customer import customer_payload, devices_for, raise_billing
from vpp.auth.security import get_password_hash, require_role
from vpp.db.engine import get_db
from vpp.db.models import (
    CustomerProfileModel,
    DRProgramModel,
    ProgramEnrollmentModel,
    UserModel,
)
from vpp.db.repositories import TariffRepository, UserRepository
from vpp.portal.billing import BillingError, compute_customer_bill
from vpp.schemas.auth import UserRole
from vpp.schemas.customer import (
    CustomerBillResponse,
    CustomerCreate,
    CustomerDevice,
    CustomerResponse,
    CustomerUpdate,
    DRProgramCreate,
    DRProgramResponse,
    DRProgramUpdate,
)

router = APIRouter(tags=["Customers"])

_admin = require_role(UserRole.ADMIN)
_staff = require_role(UserRole.ADMIN, UserRole.OPERATOR)


async def _customer_user(session: AsyncSession, customer_id: str) -> UserModel:
    user = await session.get(UserModel, customer_id)
    if user is None or user.role != UserRole.CUSTOMER.value:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Customer not found")
    return user


async def _check_tariff(session: AsyncSession, tariff_id: str | None) -> None:
    if tariff_id is not None and await TariffRepository.get(session, tariff_id) is None:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"tariff {tariff_id!r} not found"
        )


# ---------------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------------


@router.post(
    "/api/v1/customers", response_model=CustomerResponse, status_code=status.HTTP_201_CREATED
)
async def create_customer(
    body: CustomerCreate,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(_admin),
):
    if await UserRepository.get_by_username(session, body.username) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, detail="Username already taken")
    await _check_tariff(session, body.tariff_id)
    user = await UserRepository.create_user(
        session,
        username=body.username,
        hashed_password=get_password_hash(body.password),
        role=UserRole.CUSTOMER.value,
    )
    session.add(
        CustomerProfileModel(
            user_id=user.id,
            name=body.name,
            email=body.email,
            address=body.address,
            tariff_id=body.tariff_id,
            baseline_kwh_per_month=body.baseline_kwh_per_month,
        )
    )
    await session.flush()
    await session.refresh(user)
    return await customer_payload(session, user)


@router.get("/api/v1/customers", response_model=list[CustomerResponse])
async def list_customers(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(_staff),
):
    users = (
        (
            await session.execute(
                select(UserModel)
                .where(UserModel.role == UserRole.CUSTOMER.value)
                .order_by(UserModel.username)
                .offset(skip)
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    return [await customer_payload(session, u) for u in users]


@router.get("/api/v1/customers/{customer_id}", response_model=CustomerResponse)
async def get_customer(
    customer_id: str,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(_staff),
):
    return await customer_payload(session, await _customer_user(session, customer_id))


@router.patch("/api/v1/customers/{customer_id}", response_model=CustomerResponse)
async def update_customer(
    customer_id: str,
    body: CustomerUpdate,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(_admin),
):
    user = await _customer_user(session, customer_id)
    fields = body.model_dump(exclude_unset=True)
    if "tariff_id" in fields:
        await _check_tariff(session, fields["tariff_id"])
    if "name" in fields and fields["name"] is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="name cannot be null")
    profile = (
        await session.execute(
            select(CustomerProfileModel).where(CustomerProfileModel.user_id == user.id)
        )
    ).scalar_one_or_none()
    if profile is None:
        profile = CustomerProfileModel(user_id=user.id, name=fields.get("name") or user.username)
        session.add(profile)
    for k, v in fields.items():
        setattr(profile, k, v)
    await session.flush()
    return await customer_payload(session, user)


@router.get("/api/v1/customers/{customer_id}/bill", response_model=CustomerBillResponse)
async def get_customer_bill(
    customer_id: str,
    month: str | None = Query(None, description="YYYY-MM; default: current month"),
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(_staff),
):
    user = await _customer_user(session, customer_id)
    try:
        return await compute_customer_bill(session, user.id, month)
    except BillingError as exc:
        raise_billing(exc)


@router.get(
    "/api/v1/customers/{customer_id}/devices",
    response_model=list[CustomerDevice],
    response_model_exclude_none=True,
)
async def get_customer_devices(
    customer_id: str,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(_staff),
):
    user = await _customer_user(session, customer_id)
    return await devices_for(session, user.id)


# ---------------------------------------------------------------------------
# Demand-response programs
# ---------------------------------------------------------------------------


def _program_out(p: DRProgramModel, enrolled_count: int | None = None) -> dict:
    return {
        "id": p.id,
        "name": p.name,
        "description": p.description,
        "utility": p.utility,
        "incentive_per_event": p.incentive_per_event,
        "active": p.active,
        "enrolled_count": enrolled_count,
    }


@router.get(
    "/api/v1/programs", response_model=list[DRProgramResponse], response_model_exclude_none=True
)
async def list_all_programs(
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(_staff),
):
    counts = dict(
        (
            await session.execute(
                select(ProgramEnrollmentModel.program_id, func.count()).group_by(
                    ProgramEnrollmentModel.program_id
                )
            )
        ).all()
    )
    programs = (
        (await session.execute(select(DRProgramModel).order_by(DRProgramModel.name)))
        .scalars()
        .all()
    )
    return [_program_out(p, counts.get(p.id, 0)) for p in programs]


@router.post(
    "/api/v1/programs",
    response_model=DRProgramResponse,
    response_model_exclude_none=True,
    status_code=status.HTTP_201_CREATED,
)
async def create_program(
    body: DRProgramCreate,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(_admin),
):
    exists = (
        await session.execute(select(DRProgramModel.id).where(DRProgramModel.name == body.name))
    ).scalar_one_or_none()
    if exists is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT, detail="A program with this name already exists"
        )
    program = DRProgramModel(**body.model_dump())
    session.add(program)
    await session.flush()
    return _program_out(program, 0)


@router.patch(
    "/api/v1/programs/{program_id}",
    response_model=DRProgramResponse,
    response_model_exclude_none=True,
)
async def update_program(
    program_id: str,
    body: DRProgramUpdate,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(_admin),
):
    program = await session.get(DRProgramModel, program_id)
    if program is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Program not found")
    fields = body.model_dump(exclude_unset=True)
    for key in ("name", "description", "active"):
        if key in fields and fields[key] is None:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"{key} cannot be null"
            )
    if "name" in fields and fields["name"] != program.name:
        clash = (
            await session.execute(
                select(DRProgramModel.id).where(DRProgramModel.name == fields["name"])
            )
        ).scalar_one_or_none()
        if clash is not None:
            raise HTTPException(
                status.HTTP_409_CONFLICT, detail="A program with this name already exists"
            )
    for k, v in fields.items():
        setattr(program, k, v)
    await session.flush()
    return _program_out(program)
