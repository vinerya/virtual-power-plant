"""Alerts + alert-rule endpoints (consumed by the operator console).

API surface
-----------
- GET    /api/v1/alerts                     list (``since``, ``until``, ``severity``,
                                            ``status``, ``source``, ``limit``)
- GET    /api/v1/alerts/{id}                read one alert
- POST   /api/v1/alerts/{id}/ack            acknowledge          (operator/admin)
- POST   /api/v1/alerts/{id}/snooze         snooze until/for     (operator/admin)
- POST   /api/v1/alerts/{id}/resolve        resolve              (operator/admin)
- GET    /api/v1/alerts/rules               list rules
- POST   /api/v1/alerts/rules               create rule          (admin)
- GET    /api/v1/alerts/rules/{id}          read rule
- PATCH  /api/v1/alerts/rules/{id}          partial update       (admin)
- DELETE /api/v1/alerts/rules/{id}          delete               (admin)

Reads require any authenticated user.  ``status`` accepts ``active``,
``acknowledged``, ``snoozed``, ``resolved``, ``open`` (anything not
resolved) or ``all``; an expired snooze counts as ``active``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from vpp.alert_service import (
    AlertRepository,
    as_utc,
    get_alert_service,
    serialize_alert,
    snooze_until_from,
    utcnow,
)
from vpp.auth.security import get_current_user, require_role
from vpp.db.engine import get_db
from vpp.db.models import AlertRuleModel, UserModel
from vpp.schemas.alerts import (
    MAX_SNOOZE_MS,
    AlertRead,
    AlertRuleCreate,
    AlertRuleRead,
    AlertRuleUpdate,
    SnoozeRequest,
)
from vpp.schemas.auth import UserRole

router = APIRouter(prefix="/api/v1/alerts", tags=["Alerts"])

_operator = require_role(UserRole.ADMIN, UserRole.OPERATOR)
_admin = require_role(UserRole.ADMIN)

StatusFilter = Literal["active", "acknowledged", "snoozed", "resolved", "open", "all"]
SeverityFilter = Literal["info", "warning", "critical", "all"]


def _rule_read(row: AlertRuleModel) -> AlertRuleRead:
    return AlertRuleRead.model_validate(row, from_attributes=True)


async def _reload_rules() -> None:
    svc = get_alert_service()
    if svc is not None:
        await svc.reload_rules()


# ---------------------------------------------------------------------------
# Rules (declared before /{alert_id} so "rules" is not taken as an id)
# ---------------------------------------------------------------------------


@router.get("/rules", response_model=list[AlertRuleRead])
async def list_rules(
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    return [_rule_read(r) for r in await AlertRepository.list_rules(session)]


@router.post("/rules", response_model=AlertRuleRead, status_code=status.HTTP_201_CREATED)
async def create_rule(
    body: AlertRuleCreate,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(_admin),
):
    if await AlertRepository.get_rule_by_name(session, body.name) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Alert rule '{body.name}' already exists")
    row = await AlertRepository.create_rule(session, **body.model_dump())
    await session.commit()
    await _reload_rules()
    return _rule_read(row)


@router.get("/rules/{rule_id}", response_model=AlertRuleRead)
async def get_rule(
    rule_id: str,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    row = await AlertRepository.get_rule(session, rule_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Alert rule not found")
    return _rule_read(row)


@router.patch("/rules/{rule_id}", response_model=AlertRuleRead)
@router.put("/rules/{rule_id}", response_model=AlertRuleRead, include_in_schema=False)
async def update_rule(
    rule_id: str,
    body: AlertRuleUpdate,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(_admin),
):
    row = await AlertRepository.get_rule(session, rule_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Alert rule not found")
    changes = body.model_dump(exclude_unset=True)
    if "name" in changes and changes["name"] != row.name:
        clash = await AlertRepository.get_rule_by_name(session, changes["name"])
        if clash is not None:
            raise HTTPException(
                status.HTTP_409_CONFLICT, f"Alert rule '{changes['name']}' already exists"
            )
    for key, value in changes.items():
        setattr(row, key, value)
    await session.flush()
    await session.refresh(row)
    await session.commit()
    await _reload_rules()
    return _rule_read(row)


@router.delete("/rules/{rule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_rule(
    rule_id: str,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(_admin),
):
    row = await AlertRepository.get_rule(session, rule_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Alert rule not found")
    await session.delete(row)
    await session.commit()
    await _reload_rules()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------


@router.get("", response_model=list[AlertRead])
@router.get("/", response_model=list[AlertRead], include_in_schema=False)
async def list_alerts(
    since: datetime | None = Query(None, description="Only alerts fired at/after (ISO-8601)"),
    until: datetime | None = Query(None, description="Only alerts fired at/before (ISO-8601)"),
    severity: SeverityFilter = Query("all"),
    status_: StatusFilter = Query("all", alias="status"),
    source: str | None = Query(None, max_length=255),
    limit: int = Query(200, ge=1, le=1000),
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    """List alerts, newest first."""
    now = utcnow()
    rows = await AlertRepository.list_alerts(
        session,
        since=as_utc(since),
        until=as_utc(until),
        severity=None if severity == "all" else severity,
        status=status_,
        source=source,
        limit=limit,
        now=now,
    )
    return [serialize_alert(r, now) for r in rows]


async def _get_or_404(session: AsyncSession, alert_id: str):
    row = await AlertRepository.get(session, alert_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Alert not found")
    return row


@router.get("/{alert_id}", response_model=AlertRead)
async def get_alert(
    alert_id: str,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(get_current_user),
):
    return serialize_alert(await _get_or_404(session, alert_id))


@router.post("/{alert_id}/ack", response_model=AlertRead)
async def acknowledge_alert(
    alert_id: str,
    session: AsyncSession = Depends(get_db),
    user: UserModel = Depends(_operator),
):
    """Acknowledge an alert (idempotent; a resolved alert stays resolved)."""
    row = await _get_or_404(session, alert_id)
    AlertRepository.acknowledge(row, by=user.username)
    await session.commit()
    return serialize_alert(row)


@router.post("/{alert_id}/snooze", response_model=AlertRead)
async def snooze_alert(
    alert_id: str,
    body: SnoozeRequest,
    session: AsyncSession = Depends(get_db),
    _user: UserModel = Depends(_operator),
):
    """Snooze an alert; it reads as ``active`` again once ``until`` passes."""
    row = await _get_or_404(session, alert_id)
    if row.status == "resolved":
        raise HTTPException(status.HTTP_409_CONFLICT, "Cannot snooze a resolved alert")
    now = utcnow()
    until = snooze_until_from(body.until, body.duration_ms, now)
    if until <= now:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "'until' must be in the future")
    if (until - now).total_seconds() * 1000 > MAX_SNOOZE_MS:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Snooze longer than 7 days")
    AlertRepository.snooze(row, until)
    await session.commit()
    return serialize_alert(row, now)


@router.post("/{alert_id}/resolve", response_model=AlertRead)
async def resolve_alert(
    alert_id: str,
    session: AsyncSession = Depends(get_db),
    user: UserModel = Depends(_operator),
):
    row = await _get_or_404(session, alert_id)
    AlertRepository.resolve(row, by=user.username)
    await session.commit()
    return serialize_alert(row)
