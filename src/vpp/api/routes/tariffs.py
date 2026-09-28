"""Tariff CRUD + bill simulator + URDB import endpoints (Milestone 3).

API surface
-----------
- POST   /api/v1/tariffs                    admin-only create
- GET    /api/v1/tariffs                    list (filterable by ?utility=)
- GET    /api/v1/tariffs/presets            bundled URDB-shaped presets (summaries)
- GET    /api/v1/tariffs/presets/{id}       one preset incl. its URDB JSON
- GET    /api/v1/tariffs/import-urdb        whether URDB import is configured
- POST   /api/v1/tariffs/import-urdb        admin-only URDB import
- GET    /api/v1/tariffs/{id}               read
- PUT    /api/v1/tariffs/{id}               admin-only partial update
- DELETE /api/v1/tariffs/{id}               admin-only soft-delete (deleted_at)
- POST   /api/v1/tariffs/{id}/simulate      authenticated bill simulation
- POST   /api/v1/tariffs/simulate           same, with tariff_id or inline urdb_json

Read model
----------
``TariffRead`` carries the stored ``urdb_json`` (source of truth) plus
derived, read-only presentational fields -- ``components``, ``tou_heatmap``
(12x24 weekday $/kWh), ``tou_heatmap_weekend``, ``sector``, ``source``,
``nem_regime`` -- computed from the same parsed components the bill engine
uses (see :mod:`vpp.tariffs.view`).

Soft-delete vs hard-delete
--------------------------
DELETE marks the row's ``deleted_at`` column. Repository helpers and all
read endpoints filter out tombstoned rows. We picked soft-delete to keep
historical bills replayable. Pass ``hard=true`` query param for hard delete.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession

from vpp.auth.security import get_current_user, require_role
from vpp.db.engine import get_db
from vpp.db.models import UserModel
from vpp.db.repositories import TariffRepository
from vpp.events import Event, EventType, get_event_bus
from vpp.schemas.auth import UserRole
from vpp.schemas.tariffs import (
    BillCycleDTO,
    BillLineItemDTO,
    BillResponse,
    BillSimulationRequest,
    LoadSummary,
    SyntheticLoadSpec,
    TariffComponentView,
    TariffCreate,
    TariffPresetRead,
    TariffPresetSummary,
    TariffRead,
    TariffUpdate,
    URDBImportRequest,
    URDBImportStatus,
)
from vpp.tariffs import MeterTrace, Tariff, load_urdb_json
from vpp.tariffs.csv_trace import CSVTraceError, parse_csv_trace
from vpp.tariffs.nem import NEMConfigError, nem_config_from_urdb
from vpp.tariffs.preset_library import get_preset, list_presets
from vpp.tariffs.simulation import simulate_bill
from vpp.tariffs.synthetic_load import DEFAULT_AVG_KW, synthetic_trace
from vpp.tariffs.view import describe_tariff

router = APIRouter(prefix="/api/v1/tariffs", tags=["Tariffs"])
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_read(row) -> TariffRead:
    urdb_json = json.loads(row.urdb_json) if row.urdb_json else {}
    view = describe_tariff(urdb_json)
    return TariffRead(
        id=row.id,
        name=row.name,
        utility=row.utility or "",
        urdb_label=row.urdb_label,
        urdb_json=urdb_json,
        effective_date=row.effective_date,
        created_at=row.created_at,
        updated_at=row.updated_at,
        sector=view.sector,
        source="URDB" if row.urdb_label else view.source,
        description=view.description,
        components=[TariffComponentView(**vars(c)) for c in view.components],
        tou_heatmap=view.tou_heatmap,
        tou_heatmap_weekend=view.tou_heatmap_weekend,
        is_tou=view.is_tou,
        nem_regime=view.nem_regime,
        nem_source=view.nem_source,
        parse_error=view.parse_error,
    )


def _validate_urdb(urdb_json: dict[str, Any]) -> None:
    """Reject URDB JSON the bill engine cannot evaluate (422 now, not a 400 at bill time)."""
    try:
        load_urdb_json(urdb_json)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid URDB JSON: {exc}",
        ) from exc


async def _publish(event_type: EventType, data: dict[str, Any], source: str) -> None:
    # Failure to publish is logged but does not fail the API.
    try:
        await get_event_bus().publish(Event(event_type=event_type, data=data, source=source))
    except Exception:
        logger.debug("tariff event publish failed", exc_info=True)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


@router.post("", response_model=TariffRead, status_code=status.HTTP_201_CREATED)
@router.post(
    "/", response_model=TariffRead, status_code=status.HTTP_201_CREATED, include_in_schema=False
)
async def create_tariff(
    body: TariffCreate,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(require_role(UserRole.ADMIN)),
):
    """Create a new tariff (admin-only). The URDB JSON must be billable."""
    _validate_urdb(body.urdb_json)
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
    utility: str | None = None,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """List tariffs (paginated, filterable by ``utility``)."""
    rows = await TariffRepository.list(session, skip=skip, limit=limit, utility=utility)
    return [_row_to_read(r) for r in rows]


# Static paths are registered before ``/{tariff_id}`` so they are not
# swallowed by the id route.


def _preset_summary(preset_id: str, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": preset_id,
        "name": data.get("name", preset_id),
        "utility": data.get("utility"),
        "sector": data.get("sector"),
        "description": data.get("_comment"),
        "source_date": data.get("_source_date"),
        "illustrative": preset_id.startswith("illustrative"),
    }


@router.get("/presets", response_model=list[TariffPresetSummary])
async def list_tariff_presets(_user: UserModel = Depends(get_current_user)):
    """Bundled URDB-shaped tariffs to create from (no OpenEI key needed)."""
    return [_preset_summary(pid, data) for pid, data in list_presets()]


@router.get("/presets/{preset_id}", response_model=TariffPresetRead)
async def get_tariff_preset(preset_id: str, _user: UserModel = Depends(get_current_user)):
    data = get_preset(preset_id)
    if data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preset not found")
    return {**_preset_summary(preset_id, data), "urdb_json": data}


@router.get("/import-urdb", response_model=URDBImportStatus)
async def urdb_import_status(_user: UserModel = Depends(get_current_user)):
    """Whether ``POST /import-urdb`` can work (``OPENEI_API_KEY`` is set)."""
    if os.environ.get("OPENEI_API_KEY"):
        return URDBImportStatus(configured=True, detail="OpenEI URDB import is available")
    return URDBImportStatus(
        configured=False,
        detail="Set OPENEI_API_KEY on the API server to import tariffs from OpenEI URDB",
    )


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
    if "urdb_json" in fields:
        _validate_urdb(fields["urdb_json"])
    row = await TariffRepository.update(session, tariff_id, **fields)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tariff not found")
    # Realize the response BEFORE publishing — publishing yields the loop
    # and would otherwise re-enter SQLAlchemy lazy-load on a closed session.
    response = _row_to_read(row)
    # Broadcast TariffUpdated for downstream invalidation (cached optimizer
    # params, web UI).
    await _publish(
        EventType.TARIFF_UPDATED,
        {"tariff_id": tariff_id, "fields": list(fields.keys())},
        "tariffs.update",
    )
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
    await _publish(
        EventType.TARIFF_DELETED, {"tariff_id": tariff_id, "hard": hard}, "tariffs.delete"
    )


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

    Load comes from exactly one of ``meter_trace``, ``synthetic`` or ``csv``;
    see :class:`~vpp.schemas.tariffs.BillSimulationRequest`.

    Example
    -------
    .. code-block:: bash

        curl -X POST http://localhost:8000/api/v1/tariffs/<id>/simulate \\
             -H "Authorization: Bearer $TOKEN" \\
             -H "Content-Type: application/json" \\
             -d '{"synthetic": {"profile": "residential", "pv_kw": 5},
                  "period_days": 30, "timezone": "America/Los_Angeles",
                  "billing_period_start": "2024-07-01T00:00:00",
                  "compare_to": "<other tariff id>"}'

    Response shape::

        {
          "total": 245.30, "tariff_name": "PG&E E-TOU-C", "tariff_id": "...",
          "line_items": [{"kind": "energy", "label": "TOU period_1", ...}],
          "period_start": "...", "period_end": "...",
          "cycles": [...], "nem_regime": "nem2", "load_summary": {...},
          "comparison": {...same shape...} | null
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
    if body.tariff_id is None and body.urdb_json is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide exactly one of tariff_id or urdb_json",
        )
    return await _do_simulate(session, body)


def _parse_tariff(urdb_json: dict[str, Any]) -> Tariff:
    try:
        return load_urdb_json(urdb_json)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid URDB JSON: {exc}",
        ) from exc


def _aware(dt: datetime, tz: ZoneInfo) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=tz)


def _build_load(
    body: BillSimulationRequest, tariff: Tariff, tz: ZoneInfo
) -> tuple[MeterTrace, datetime, datetime, str, str | None]:
    """Return ``(trace, start, end, source, method)`` for the request's load source."""
    start = _aware(body.billing_period_start, tz) if body.billing_period_start else None
    end = _aware(body.billing_period_end, tz) if body.billing_period_end else None

    if body.meter_trace is not None:
        dto = body.meter_trace
        trace = MeterTrace(
            timestamps=[_aware(ts, tz) for ts in dto.timestamps],
            import_kwh=list(dto.import_kwh),
            export_kwh=list(dto.export_kwh) if dto.export_kwh else [0.0] * len(dto.timestamps),
            interval_minutes=dto.interval_minutes,
            tz=tz,
        )
        assert start is not None and end is not None  # enforced by the schema
        return trace, start, end, "meter_trace", None

    if body.csv is not None:
        try:
            trace = parse_csv_trace(body.csv, tz=tz)
        except CSVTraceError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid CSV: {exc}"
            ) from exc
        start = start or trace.timestamps[0]
        end = end or trace.timestamps[-1] + timedelta(minutes=trace.interval_minutes)
        if end <= start:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="billing_period_end must be after billing_period_start",
            )
        return trace, start, end, "csv", None

    spec = body.synthetic if isinstance(body.synthetic, SyntheticLoadSpec) else SyntheticLoadSpec()
    sector = (tariff.sector or "").lower()
    profile = spec.profile or (
        "commercial" if sector in {"commercial", "industrial"} else "residential"
    )
    if start is None:
        now_local = datetime.now(timezone.utc).astimezone(tz)
        start = now_local.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if end is not None:
        days = max(1, math.ceil((end - start).total_seconds() / 86400))
    else:
        days = body.period_days
        end = start + timedelta(days=days)
    trace = synthetic_trace(
        start=start,
        days=days,
        tz=tz,
        profile=profile,
        avg_kw=spec.avg_kw,
        pv_kw=spec.pv_kw,
        interval_minutes=spec.interval_minutes,
    )
    avg = spec.avg_kw if spec.avg_kw is not None else DEFAULT_AVG_KW[profile]
    method = (
        f"deterministic {profile} shape, {avg:g} kW average"
        + (f", {spec.pv_kw:g} kW PV" if spec.pv_kw else "")
        + " (illustrative, not metered)"
    )
    return trace, start, end, "synthetic", method


def _bill_one(
    tariff: Tariff,
    urdb_json: dict[str, Any],
    trace: MeterTrace,
    start: datetime,
    end: datetime,
    body: BillSimulationRequest,
    tariff_id: str | None,
) -> BillResponse:
    cfg = nem_config_from_urdb(urdb_json)
    regime = body.nem or cfg.regime
    avoided = body.nem3_avoided_cost or list(cfg.avoided_cost)
    try:
        result = simulate_bill(
            tariff,
            trace,
            start,
            end,
            nem_regime=regime,
            avoided_cost=avoided,
            cycle_mode=body.billing_cycle,
        )
    except NEMConfigError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return BillResponse(
        total=result.total,
        tariff_name=result.tariff_name,
        tariff_id=tariff_id,
        line_items=[
            BillLineItemDTO(
                kind=li.kind,
                label=li.label,
                quantity=li.quantity,
                unit=li.unit,
                rate=li.rate,
                amount=li.amount,
            )
            for li in result.line_items
        ],
        period_start=start,
        period_end=end,
        cycles=[
            BillCycleDTO(
                period_start=c.start,
                period_end=c.end,
                total=c.total,
                export_credit=c.credit.amount,
            )
            for c in result.cycles
        ],
        nem_regime=result.nem_regime,
        nem_source="request" if body.nem else cfg.source,
        export_kwh=result.export_kwh,
        export_credit=result.export_credit,
        notes=result.notes,
    )


async def _do_simulate(
    session: AsyncSession,
    body: BillSimulationRequest,
    *,
    force_tariff_id: str | None = None,
) -> BillResponse:
    tariff_id = force_tariff_id if force_tariff_id is not None else body.tariff_id

    if tariff_id is not None:
        row = await TariffRepository.get(session, tariff_id)
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tariff not found")
        urdb_json = json.loads(row.urdb_json) if row.urdb_json else {}
    else:
        urdb_json = body.urdb_json or {}
    tariff = _parse_tariff(urdb_json)

    compare: tuple[str, Tariff, dict[str, Any]] | None = None
    if body.compare_to:
        crow = await TariffRepository.get(session, body.compare_to)
        if crow is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Comparison tariff not found"
            )
        cjson = json.loads(crow.urdb_json) if crow.urdb_json else {}
        compare = (crow.id, _parse_tariff(cjson), cjson)

    tz = ZoneInfo(body.timezone)

    def _run() -> BillResponse:
        trace, start, end, source, method = _build_load(body, tariff, tz)
        response = _bill_one(tariff, urdb_json, trace, start, end, body, tariff_id)
        window = [i for i, ts in enumerate(trace.timestamps) if start <= ts < end]
        response.load_summary = LoadSummary(
            source=source,
            method=method,
            timezone=body.timezone,
            interval_minutes=trace.interval_minutes,
            intervals=len(window),
            import_kwh=round(sum(trace.import_kwh[i] for i in window), 3),
            export_kwh=round(sum(trace.export_kwh[i] for i in window), 3),
            peak_kw=round(
                max((trace.import_kwh[i] for i in window), default=0.0) / trace.interval_hours,
                3,
            ),
        )
        if compare is not None:
            cid, ctariff, cjson = compare
            response.comparison = _bill_one(ctariff, cjson, trace, start, end, body, cid)
        return response

    # CPU-bound (a year of 5-minute CSV data is ~100k intervals): keep it off the loop.
    return await run_in_threadpool(_run)


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
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(URDB_API_URL, params=params)
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not reach OpenEI: {exc.__class__.__name__}",
        ) from exc
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
    a 503 if the key is missing (``GET /import-urdb`` reports this up front).

    Errors
    ------
    - 400 if the URDB JSON cannot be parsed by ``load_urdb_json``.
    - 404 if no record matches the given ``urdb_label``.
    - 409 if a tariff with the same ``urdb_label`` already exists.
    - 502 if OpenEI is unreachable or returns a non-2xx response.
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
    except Exception as exc:
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
