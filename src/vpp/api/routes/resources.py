"""CRUD routes for energy resources."""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from vpp.auth.security import get_current_user
from vpp.db.engine import get_db
from vpp.db.models import UserModel
from vpp.db.repositories import ResourceRepository
from vpp.schemas.resources import (
    ResourceCreate,
    ResourceResponse,
    ResourceUpdate,
    ResourceType,
)

router = APIRouter(prefix="/api/v1/resources", tags=["Resources"])


def _model_to_response(obj) -> dict:
    """Adapt a ResourceModel to the response schema dict."""
    cfg = json.loads(obj.config_json) if obj.config_json else {}
    meta = json.loads(obj.metadata_json) if obj.metadata_json else {}
    return {
        "id": obj.id,
        "name": obj.name,
        "resource_type": obj.resource_type,
        "rated_power": obj.rated_power,
        "online": obj.online,
        "current_power": obj.current_power,
        "efficiency": obj.efficiency,
        "created_at": obj.created_at,
        "updated_at": obj.updated_at,
        "metadata": meta,
        **cfg,
    }


@router.get("/", response_model=list[ResourceResponse])
async def list_resources(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    resource_type: Optional[str] = None,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """List all registered energy resources."""
    items = await ResourceRepository.list_all(session, skip=skip, limit=limit, resource_type=resource_type)
    return [_model_to_response(r) for r in items]


@router.post("/", response_model=ResourceResponse, status_code=status.HTTP_201_CREATED)
async def create_resource(
    body: ResourceCreate,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """Register a new energy resource."""
    existing = await ResourceRepository.get_by_name(session, body.name)
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Resource name already exists")

    extra_fields = body.model_dump(exclude={"name", "resource_type", "rated_power", "metadata"})
    obj = await ResourceRepository.create(
        session,
        name=body.name,
        resource_type=body.resource_type.value,
        rated_power=body.rated_power,
        config=extra_fields,
        metadata=body.metadata,
    )
    return _model_to_response(obj)


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
    return _model_to_response(obj)


@router.put("/{resource_id}", response_model=ResourceResponse)
async def update_resource(
    resource_id: str,
    body: ResourceUpdate,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """Update resource fields."""
    updates = body.model_dump(exclude_none=True)
    if "metadata" in updates:
        updates["metadata_json"] = json.dumps(updates.pop("metadata"))
    obj = await ResourceRepository.update(session, resource_id, **updates)
    if obj is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resource not found")
    return _model_to_response(obj)


@router.delete("/{resource_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_resource(
    resource_id: str,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """Remove a resource."""
    deleted = await ResourceRepository.delete(session, resource_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resource not found")
