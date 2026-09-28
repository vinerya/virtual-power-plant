"""Audit log: ``GET /api/v1/audit`` (admin only).

Rows are written by :mod:`vpp.audit`; this router only reads them. Newest
first, paginated with ``limit`` / ``offset`` and ``X-Total-Count``
(:mod:`vpp.api.pagination`).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from vpp.api.pagination import Page, page_params, paginate
from vpp.auth.security import require_role
from vpp.db.engine import get_db
from vpp.db.models import AuditLogModel, UserModel
from vpp.schemas.audit import AuditEntry
from vpp.schemas.auth import UserRole

router = APIRouter(prefix="/api/v1/audit", tags=["Audit"])

_page = page_params(default_limit=100, max_limit=500)


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _entry(row: AuditLogModel) -> AuditEntry:
    details: Any = {}
    if row.details_json:
        try:
            details = json.loads(row.details_json)
        except ValueError:
            details = {"raw": row.details_json}
    return AuditEntry(
        id=row.id,
        ts=_utc(row.ts),
        actor_id=row.actor_id,
        actor_username=row.actor_username,
        action=row.action,
        target_type=row.target_type,
        target_id=row.target_id,
        client_ip=row.client_ip,
        outcome=row.outcome,
        details=details if isinstance(details, dict) else {"value": details},
    )


@router.get("", response_model=list[AuditEntry])
async def list_audit(
    response: Response,
    actor: str | None = Query(None, max_length=64, description="Actor user id or username"),
    action: str | None = Query(
        None,
        max_length=64,
        description="Exact action (e.g. user.delete), or a prefix ending in '.' (e.g. user.)",
    ),
    outcome: str | None = Query(None, pattern="^(success|failure|denied)$"),
    target_type: str | None = Query(None, max_length=32),
    target_id: str | None = Query(None, max_length=128),
    since: datetime | None = Query(None, description="Only entries at/after (ISO-8601)"),
    until: datetime | None = Query(None, description="Only entries at/before (ISO-8601)"),
    page: Page = Depends(_page),
    session: AsyncSession = Depends(get_db),
    _admin: UserModel = Depends(require_role(UserRole.ADMIN)),
):
    """Security audit trail, newest first (admin only).

    The total number of matching entries is in ``X-Total-Count``.
    """
    stmt = select(AuditLogModel).order_by(AuditLogModel.ts.desc(), AuditLogModel.id.desc())
    if actor:
        stmt = stmt.where(
            or_(AuditLogModel.actor_id == actor, AuditLogModel.actor_username == actor)
        )
    if action:
        if action.endswith("."):
            stmt = stmt.where(AuditLogModel.action.startswith(action, autoescape=True))
        else:
            stmt = stmt.where(AuditLogModel.action == action)
    if outcome:
        stmt = stmt.where(AuditLogModel.outcome == outcome)
    if target_type:
        stmt = stmt.where(AuditLogModel.target_type == target_type)
    if target_id:
        stmt = stmt.where(AuditLogModel.target_id == target_id)
    if since is not None:
        stmt = stmt.where(AuditLogModel.ts >= _utc(since))
    if until is not None:
        stmt = stmt.where(AuditLogModel.ts <= _utc(until))
    rows = await paginate(session, response, stmt, page)
    return [_entry(r) for r in rows]
