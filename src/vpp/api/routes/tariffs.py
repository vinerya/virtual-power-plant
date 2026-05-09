"""Tariff CRUD + bill simulator + URDB import endpoints (Milestone 3).

API surface
-----------
- POST   /api/v1/tariffs              admin-only create
- GET    /api/v1/tariffs              list (filterable by ?utility=)
- GET    /api/v1/tariffs/{id}         read
- PUT    /api/v1/tariffs/{id}         admin-only partial update
- DELETE /api/v1/tariffs/{id}         admin-only soft-delete (deleted_at)
- POST   /api/v1/tariffs/{id}/simulate    authenticated bill simulation
- POST   /api/v1/tariffs/import-urdb      admin-only URDB import

Soft-delete vs hard-delete
--------------------------
DELETE marks the row's ``deleted_at`` column. Repository helpers and all
read endpoints filter out tombstoned rows. We picked soft-delete to keep
historical bills replayable. Pass ``hard=true`` query param for hard delete.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from vpp.auth.security import get_current_user, require_role
from vpp.db.engine import get_db
from vpp.db.models import UserModel
from vpp.db.repositories import TariffRepository
from vpp.events import Event, EventType, get_event_bus
from vpp.schemas.auth import UserRole
from vpp.schemas.tariffs import (
    BillLineItemDTO,
    BillResponse,
    BillSimulationRequest,
    TariffCreate,
    TariffRead,
    TariffUpdate,
    URDBImportRequest,
)
from vpp.tariffs import (
    BillingPeriod,
    MeterTrace,
    load_urdb_json,
)


router = APIRouter(prefix="/api/v1/tariffs", tags=["Tariffs"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_read(row) -> TariffRead:
    return TariffRead(
        id=row.id,
        name=row.name,
        utility=row.utility or "",
        urdb_label=row.urdb_label,
        urdb_json=json.loads(row.urdb_json) if row.urdb_json else {},
        effective_date=row.effective_date,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _meter_trace_from_dto(dto) -> MeterTrace:
    timestamps = []
    for ts in dto.timestamps:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        timestamps.append(ts)
    return MeterTrace(
        timestamps=timestamps,
        import_kwh=list(dto.import_kwh),
        export_kwh=list(dto.export_kwh) if dto.export_kwh else [0.0] * len(timestamps),
        interval_minutes=dto.interval_minutes,
        tz=timezone.utc,
    )


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


@router.post("", response_model=TariffRead, status_code=status.HTTP_201_CREATED)
@router.post("/", response_model=TariffRead, status_code=status.HTTP_201_CREATED, include_in_schema=False)
async def create_tariff(
    body: TariffCreate,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(require_role(UserRole.ADMIN)),
):
    """Create a new tariff (admin-only)."""
    row = await TariffRepository.create(
        session,
        name=body.name,
        utility=body.utility,
        urdb_json=body.urdb_json,
        effective_date=body.effective_date,
        urdb_label=body.urdb_label,
    )
    return _row_to_read(row)


@router.get("", response_model=list[TariffRead])
@router.get("/", response_model=list[TariffRead], include_in_schema=False)
async def list_tariffs(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    utility: Optional[str] = None,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """List tariffs (paginated, filterable by ``utility``)."""
    rows = await TariffRepository.list(session, skip=skip, limit=limit, utility=utility)
    return [_row_to_read(r) for r in rows]


@router.get("/{tariff_id}", response_model=TariffRead)
async def get_tariff(
    tariff_id: str,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    row = await TariffRepository.get(session, tariff_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tariff not found")
    return _row_to_read(row)


@router.put("/{tariff_id}", response_model=TariffRead)
async def update_tariff(
    tariff_id: str,
    body: TariffUpdate,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(require_role(UserRole.ADMIN)),
):
    fields = body.model_dump(exclude_none=True)
    row = await TariffRepository.update(session, tariff_id, **fields)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tariff not found")
    # Realize the response BEFORE publishing — publishing yields the loop
    # and would otherwise re-enter SQLAlchemy lazy-load on a closed session.
    response = _row_to_read(row)
    # Broadcast TariffUpdated for downstream invalidation (cached optimizer
    # params, web UI). Failure to publish is logged but does not fail the API.
    try:
        await get_event_bus().publish(
            Event(
                event_type=EventType.TARIFF_UPDATED,
                data={"tariff_id": tariff_id, "fields": list(fields.keys())},
                source="tariffs.update",
            )
        )
    except Exception:  # noqa: BLE001
        pass
    return response


@router.delete("/{tariff_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_tariff(
    tariff_id: str,
    hard: bool = Query(False, description="If true, hard-delete instead of soft-delete"),
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(require_role(UserRole.ADMIN)),
):
    """Soft-delete by default; pass ?hard=true to hard-delete."""
    ok = await TariffRepository.delete(session, tariff_id, soft=not hard)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tariff not found")
    try:
        await get_event_bus().publish(
            Event(
                event_type=EventType.TARIFF_DELETED,
                data={"tariff_id": tariff_id, "hard": hard},
                source="tariffs.delete",
            )
        )
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


@router.post("/{tariff_id}/simulate", response_model=BillResponse)
async def simulate_bill_from_id(
    tariff_id: str,
    body: BillSimulationRequest,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """Simulate a bill against a stored tariff.

    Example
    -------
    .. code-block:: bash

        curl -X POST http://localhost:8000/api/v1/tariffs/<id>/simulate \\
             -H "Authorization: Bearer $TOKEN" \\
             -H "Content-Type: application/json" \\
             -d '{
               "meter_trace": {"timestamps": ["2024-07-01T00:00:00Z", ...],
                               "import_kwh": [1.0, ...],
                               "interval_minutes": 60},
               "billing_period_start": "2024-07-01T00:00:00Z",
               "billing_period_end":   "2024-07-31T00:00:00Z"
             }'

    Response shape::

        {
          "total": 245.30,
          "tariff_name": "PG&E E-TOU-C",
          "line_items": [{"kind":"energy","label":"TOU period_2","quantity":...}],
          "period_start": "...", "period_end": "..."
        }
    """
    return await _do_simulate(session, body, force_tariff_id=tariff_id)


@router.post("/simulate", response_model=BillResponse)
async def simulate_bill_inline(
    body: BillSimulationRequest,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """Simulate a bill — body must carry tariff_id OR urdb_json."""
    return await _do_simulate(session, body)


async def _do_simulate(
    session: AsyncSession,
    body: BillSimulationRequest,
    *,
    force_tariff_id: str | None = None,
) -> BillResponse:
    if force_tariff_id is not None:
        tariff_id = force_tariff_id
    else:
        tariff_id = body.tariff_id

    if tariff_id is not None:
        row = await TariffRepository.get(session, tariff_id)
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tariff not found")
        urdb_json = json.loads(row.urdb_json) if row.urdb_json else {}
    else:
        urdb_json = body.urdb_json

    try:
        tariff = load_urdb_json(urdb_json)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid URDB JSON: {exc}",
        ) from exc

    trace = _meter_trace_from_dto(body.meter_trace)
    start = body.billing_period_start
    end = body.billing_period_end
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    period = BillingPeriod(start=start, end=end)
    bill = tariff.bill(trace, period)

    # ---- NEM-aware export credit (M4) -----------------------------------
    # When the meter trace includes export_kwh > 0, compute an export credit
    # at:
    #   * NEM 2.0  -> retail rate proxy (avg $/kWh of energy line items).
    #   * NEM 3.0  -> hourly avoided-cost vector (24 entries, repeats).
    #   * none     -> no credit.
    nem = (body.nem or "none").lower()
    line_items_out = list(bill.line_items)
    total = bill.total
    if nem in {"nem2", "nem3"}:
        total_export = sum(trace.export_kwh)
        if total_export > 0:
            credit_amount = 0.0
            if nem == "nem2":
                # Retail-rate proxy: total energy $ / total energy kWh.
                e_amt = sum(
                    li.amount for li in bill.line_items if li.kind in {"energy", "tier"}
                )
                e_kwh = sum(
                    li.quantity for li in bill.line_items if li.kind in {"energy", "tier"}
                )
                avg_rate = (e_amt / e_kwh) if e_kwh > 0 else 0.0
                credit_amount = round(total_export * avg_rate, 4)
            else:  # nem3
                acc = list(body.nem3_avoided_cost or [])
                if not acc:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="nem='nem3' requires nem3_avoided_cost",
                    )
                credit = 0.0
                for ts, exp in zip(trace.timestamps, trace.export_kwh):
                    if exp <= 0:
                        continue
                    if not (start <= ts < end):
                        continue
                    hour_idx = ts.astimezone(timezone.utc).hour
                    rate = float(acc[hour_idx % len(acc)])
                    credit += exp * rate
                credit_amount = round(credit, 4)
            if credit_amount > 0:
                from vpp.tariffs import BillLineItem  # local import OK
                line_items_out.append(
                    BillLineItem(
                        kind="credit",
                        label=f"{nem.upper()} export credit",
                        quantity=round(total_export, 4),
                        unit="kWh",
                        rate=round(credit_amount / total_export, 6),
                        amount=-credit_amount,
                        meta={"nem": nem},
                    )
                )
                total = round(total - credit_amount, 4)

    return BillResponse(
        total=total,
        tariff_name=bill.tariff_name,
        line_items=[
            BillLineItemDTO(
                kind=li.kind, label=li.label, quantity=li.quantity,
                unit=li.unit, rate=li.rate, amount=li.amount,
            )
            for li in line_items_out
        ],
        period_start=start,
        period_end=end,
    )

# ---------------------------------------------------------------------------
# URDB import
# ---------------------------------------------------------------------------


URDB_API_URL = "https://api.openei.org/utility_rates"


async def _fetch_urdb_record(label: str, api_key: str) -> dict[str, Any]:
    """Fetch a single URDB record by getpage id."""
    params = {
        "version": 8,
        "format": "json",
        "getpage": label,
        "api_key": api_key,
        "detail": "full",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(URDB_API_URL, params=params)
    if resp.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"OpenEI returned HTTP {resp.status_code}",
        )
    payload = resp.json()
    items = payload.get("items") or []
    if not items:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"URDB record '{label}' not found",
        )
    return items[0]


@router.post("/import-urdb", response_model=TariffRead, status_code=status.HTTP_201_CREATED)
async def import_urdb(
    body: URDBImportRequest,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(require_role(UserRole.ADMIN)),
):
    """Import a tariff from the OpenEI URDB by record id.

    Authentication
    --------------
    Admin role only.

    Environment
    -----------
    Requires the ``OPENEI_API_KEY`` environment variable to be set with
    a valid api.openei.org key. The server will reject the request with
    a 503 if the key is missing.

    Errors
    ------
    - 400 if the URDB JSON cannot be parsed by ``load_urdb_json``.
    - 404 if no record matches the given ``urdb_label``.
    - 409 if a tariff with the same ``urdb_label`` already exists.
    - 502 if OpenEI returns a non-2xx response.
    - 503 if ``OPENEI_API_KEY`` is not configured.
    """
    api_key = os.environ.get("OPENEI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OPENEI_API_KEY environment variable not set",
        )

    existing = await TariffRepository.get_by_urdb_label(session, body.urdb_label)
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Tariff with urdb_label '{body.urdb_label}' already exists",
        )

    record = await _fetch_urdb_record(body.urdb_label, api_key)

    # Validate by parsing
    try:
        parsed = load_urdb_json(record)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse URDB record: {exc}",
        ) from exc

    name = body.name_override or parsed.name or record.get("name", "URDB tariff")
    utility = parsed.utility or record.get("utility", "")
    eff = None
    sd = record.get("startdate")
    if sd:
        try:
            # URDB startdate is unix epoch seconds in some payloads, ISO in others.
            if isinstance(sd, (int, float)):
                eff = datetime.fromtimestamp(sd, tz=timezone.utc).date()
            else:
                eff = datetime.fromisoformat(str(sd).replace("Z", "+00:00")).date()
        except Exception:
            eff = None

    row = await TariffRepository.create(
        session,
        name=name,
        utility=utility,
        urdb_json=record,
        effective_date=eff,
        urdb_label=body.urdb_label,
    )
    return _row_to_read(row)
