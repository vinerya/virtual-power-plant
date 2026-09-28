"""CRUD routes for energy resources.

``POST`` validates the body against the type-specific model selected by
``resource_type`` (see :mod:`vpp.schemas.resources`) and persists every
field: ``capacity_kwh`` in ``nominal_energy_kwh``, ``chemistry`` and
``efficiency`` in their columns, the rest in ``config_json``. ``PUT`` is a
partial update that is re-validated against the same model after merging,
so both paths enforce the same rules. The optimizer's resource loader
(:mod:`vpp.api.optimization_support`) reads the same columns/keys.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import (
    AsyncSession,
)

from vpp.api.pagination import Page, page_params, paginate
from vpp.auth.security import get_current_user, require_role
from vpp.db.engine import get_db
from vpp.db.models import ResourceModel, UserModel
from vpp.db.repositories import ResourceRepository
from vpp.portal.sites import resource_capacity_kwh
from vpp.portal.telemetry import latest_soc
from vpp.schemas.auth import UserRole
from vpp.schemas.resources import (
    CREATE_MODELS,
    BatteryCreate,
    ResourceCreate,
    ResourceCreateRequest,
    ResourceResponse,
    ResourceUpdate,
    normalize_resource_type,
    resource_create_adapter,
)

router = APIRouter(prefix="/api/v1/resources", tags=["Resources"])

# Mutations change what the dispatcher controls: operators and admins only.
_writer = require_role(UserRole.ADMIN, UserRole.OPERATOR)

#: Type-specific fields that live in dedicated columns rather than config_json.
_COLUMN_FIELDS = frozenset({"capacity_kwh", "chemistry"})
#: Accepted on create for convenience, stored as ``state_of_charge``.
_DERIVED_FIELDS = frozenset({"current_charge_kwh"})
_COMMON_UPDATE_FIELDS = frozenset({"name", "rated_power", "metadata", "online", "efficiency"})


def _loads(raw: str | None) -> dict[str, Any]:
    try:
        val = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        return {}
    return val if isinstance(val, dict) else {}


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def configured_soc(row: ResourceModel, cfg: dict[str, Any] | None = None) -> float | None:
    """The SOC recorded on the resource itself (0-1), if any."""
    cfg = _loads(row.config_json) if cfg is None else cfg
    soc = _num(cfg.get("state_of_charge"))
    if soc is None:
        charge, cap = _num(cfg.get("current_charge_kwh")), resource_capacity_kwh(row)
        if charge is not None and cap:
            soc = charge / cap
    if soc is None:
        return None
    soc = soc / 100.0 if soc > 1.0 else soc
    return max(0.0, min(1.0, soc))


def resource_to_response(row: ResourceModel, telemetry_soc: float | None = None) -> dict:
    """Render a resource row in the :class:`ResourceResponse` shape."""
    cfg = _loads(row.config_json)
    out: dict[str, Any] = {
        "id": row.id,
        "name": row.name,
        "resource_type": row.resource_type,
        "rated_power": row.rated_power,
        "online": row.online,
        "current_power": row.current_power,
        "efficiency": row.efficiency,
        "site_id": row.site_id,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
        "metadata": _loads(row.metadata_json),
    }
    fields = ResourceResponse.model_fields
    for key, value in cfg.items():
        if key in fields and key not in out and key not in _DERIVED_FIELDS:
            out[key] = value
    if row.resource_type == "battery":
        cap = resource_capacity_kwh(row)
        out["capacity_kwh"] = cap
        out["chemistry"] = row.chemistry
        out["state_of_health"] = row.state_of_health
        soc: float | None
        source: str | None
        if telemetry_soc is not None:
            soc, source = max(0.0, min(1.0, telemetry_soc)), "telemetry"
        else:
            soc = configured_soc(row, cfg)
            source = "configured" if soc is not None else None
        out["state_of_charge"] = soc
        out["state_of_charge_source"] = source
        out["current_charge_kwh"] = soc * cap if soc is not None and cap else None
        throughput = row.cumulative_throughput_kwh or 0.0
        out["equivalent_full_cycles"] = throughput / (2.0 * cap) if cap else None
    return out


def _state_of(row: ResourceModel, model_cls: type[ResourceCreate]) -> dict[str, Any]:
    """Current row as a create-model input dict (for merge + re-validation)."""
    cfg = _loads(row.config_json)
    state: dict[str, Any] = {
        "name": row.name,
        "resource_type": row.resource_type,
        "rated_power": row.rated_power,
        "metadata": _loads(row.metadata_json),
    }
    if row.efficiency is not None and 0.0 < row.efficiency <= 1.0:
        state["efficiency"] = row.efficiency
    type_fields = model_cls.type_fields()
    for key, value in cfg.items():
        if key in type_fields and key not in _COLUMN_FIELDS | _DERIVED_FIELDS:
            state[key] = value
    if model_cls is BatteryCreate:
        cap = resource_capacity_kwh(row)
        if cap is not None:
            state["capacity_kwh"] = cap
        if row.chemistry:
            state["chemistry"] = row.chemistry
        soc = configured_soc(row, cfg)
        if soc is not None:
            state["state_of_charge"] = soc
    return state


def _apply(row: ResourceModel, model: ResourceCreate) -> None:
    """Write a validated create-model onto a row (columns + config_json)."""
    row.name = model.name
    row.resource_type = model.resource_type.value
    row.rated_power = model.rated_power
    row.metadata_json = json.dumps(model.metadata)
    row.efficiency = model.efficiency if model.efficiency is not None else 0.95

    type_fields = type(model).type_fields()
    # Keep keys other writers put in config_json (e.g. available_kw).
    cfg = {k: v for k, v in _loads(row.config_json).items() if k not in type_fields}
    for key in type_fields - _COLUMN_FIELDS - _DERIVED_FIELDS:
        value = getattr(model, key)
        if value is not None:
            cfg[key] = value
    if isinstance(model, BatteryCreate):
        soc = model.configured_soc()
        if soc is not None:
            cfg["state_of_charge"] = soc
        row.nominal_energy_kwh = model.capacity_kwh
        row.chemistry = model.chemistry
    row.config_json = json.dumps(cfg)


def _validation_error(exc: ValidationError) -> RequestValidationError:
    return RequestValidationError(
        [
            {**e, "loc": ("body", *e["loc"])}
            for e in exc.errors(include_url=False, include_context=False)
        ]
    )


async def _conflict_if_name_taken(session: AsyncSession, name: str) -> None:
    if await ResourceRepository.get_by_name(session, name) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Resource name already exists"
        )


@router.get("", response_model=list[ResourceResponse])
async def list_resources(
    response: Response,
    page: Page = Depends(page_params(default_limit=50, max_limit=200, legacy_skip=True)),
    resource_type: str | None = Query(None, description="battery | solar | wind_turbine (wind)"),
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """List registered energy resources, newest first.

    Paginated (``limit`` / ``offset``, ``skip`` is an alias); the total is in
    ``X-Total-Count``.
    """
    stmt = ResourceRepository.list_query(
        normalize_resource_type(resource_type) if resource_type else None
    )
    items = await paginate(session, response, stmt, page)
    socs = await latest_soc(session, [r.id for r in items if r.resource_type == "battery"])
    return [resource_to_response(r, socs.get(r.id)) for r in items]


@router.post("", response_model=ResourceResponse, status_code=status.HTTP_201_CREATED)
async def create_resource(
    body: ResourceCreateRequest,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(_writer),
):
    """Register a new energy resource.

    The body is validated against the model for its ``resource_type``
    (``battery``, ``solar``, ``wind_turbine``; ``wind`` is an alias); fields
    that do not belong to that type are rejected with 422.
    """
    await _conflict_if_name_taken(session, body.name)
    row = ResourceModel(config_json="{}", metadata_json="{}")
    _apply(row, body)
    session.add(row)
    await session.flush()
    await session.refresh(row)
    return resource_to_response(row)


@router.get("/{resource_id}", response_model=ResourceResponse)
async def get_resource(
    resource_id: str,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """Get a single resource by ID."""
    obj = await ResourceRepository.get_by_id(session, resource_id)
    if obj is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resource not found")
    socs = await latest_soc(session, [obj.id]) if obj.resource_type == "battery" else {}
    return resource_to_response(obj, socs.get(obj.id))


@router.put("/{resource_id}", response_model=ResourceResponse)
async def update_resource(
    resource_id: str,
    body: ResourceUpdate,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(_writer),
):
    """Partially update a resource (see :class:`~vpp.schemas.resources.ResourceUpdate`)."""
    row = await ResourceRepository.get_by_id(session, resource_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resource not found")
    updates = body.model_dump(exclude_unset=True)
    online = updates.pop("online", None)
    if "metadata" in updates and updates["metadata"] is None:
        updates["metadata"] = {}

    model_cls = CREATE_MODELS.get(normalize_resource_type(row.resource_type))
    allowed = _COMMON_UPDATE_FIELDS | (model_cls.type_fields() if model_cls else frozenset())
    foreign = sorted(set(updates) - allowed)
    if foreign:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Fields not applicable to resource_type '{row.resource_type}': {foreign}",
        )
    if "name" in updates and updates["name"] != row.name:
        await _conflict_if_name_taken(session, updates["name"])

    if model_cls is None:
        # A type this API does not model (created by another writer): only
        # the common columns can be changed.
        for key in ("name", "rated_power", "efficiency"):
            if updates.get(key) is not None:
                setattr(row, key, updates[key])
        if "metadata" in updates:
            row.metadata_json = json.dumps(updates["metadata"])
    elif updates:
        merged = _state_of(row, model_cls)
        if "current_charge_kwh" in updates:
            merged.pop("state_of_charge", None)
        merged.update(updates)
        try:
            model = resource_create_adapter.validate_python(merged)
        except ValidationError as exc:
            raise _validation_error(exc) from exc
        _apply(row, model)
    if online is not None:
        row.online = online
    await session.flush()
    # ``updated_at`` has a server-side onupdate, so the flush expires it;
    # reload now rather than lazy-loading outside the async context.
    await session.refresh(row)
    socs = await latest_soc(session, [row.id]) if row.resource_type == "battery" else {}
    return resource_to_response(row, socs.get(row.id))


@router.delete("/{resource_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_resource(
    resource_id: str,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(_writer),
):
    """Remove a resource."""
    deleted = await ResourceRepository.delete(session, resource_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resource not found")
