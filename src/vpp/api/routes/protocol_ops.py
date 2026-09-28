"""Protocol data (read-only) and operator actions for OCPP / OpenADR / IEEE 2030.5.

Reads are open to every operator-side user (``get_current_user``: customer
accounts are refused); actions (remote start/stop, opt-in/out override)
require the ``admin`` or ``operator`` role. Every response carries the
adapter's ``mode`` so a simulated adapter is never mistaken for real
hardware / a real utility connection.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vpp import audit
from vpp.api.pagination import Page, page_params, paginate
from vpp.api.routes.protocols import get_registry
from vpp.auth.security import get_current_user, require_role
from vpp.db.engine import get_db
from vpp.db.models import DREventResponseModel
from vpp.dr.orchestrator import DROrchestrator, get_dr_orchestrator, response_to_dict
from vpp.dr.translate import DRPolicy, current_signal_value
from vpp.protocols.base import ProtocolRegistry
from vpp.protocols.ieee2030_5 import IEEE2030_5Adapter
from vpp.protocols.ocpp import ChargePoint, OCPPAdapter
from vpp.protocols.openadr import DREventStatus, OpenADRAdapter
from vpp.settings import get_settings

router = APIRouter(prefix="/api/v1/protocols", tags=["protocols"])
dr_router = APIRouter(prefix="/api/v1/dr", tags=["demand-response"])


def _adapter(registry: ProtocolRegistry, name: str, cls: type) -> Any:
    adapter = registry.get(name)
    if not isinstance(adapter, cls):
        raise HTTPException(status_code=404, detail=f"Protocol '{name}' is not running")
    return adapter


def _header(adapter: Any) -> dict[str, Any]:
    return {
        "protocol": adapter.name,
        "status": adapter.status.value,
        "mode": adapter.mode.value,
        "simulated": adapter.mode.value == "simulated",
    }


# ---------------------------------------------------------------------------
# OCPP
# ---------------------------------------------------------------------------


def _cp_dict(adapter: OCPPAdapter, cp: ChargePoint) -> dict[str, Any]:
    connector_ids = sorted(set(cp.connector_status) | set(cp.connector_meter))
    out = cp.to_dict()
    out.update(
        {
            "serial_number": cp.serial_number,
            "firmware_version": cp.firmware_version,
            "connected": adapter.is_charge_point_connected(cp.charge_point_id),
            "connectors": [
                {
                    "connector_id": cid,
                    "status": cp.connector_status[cid].value
                    if cid in cp.connector_status
                    else None,
                    "meter": cp.connector_meter.get(cid),
                }
                for cid in connector_ids
            ],
            "active_profile": cp.active_profile.to_dict() if cp.active_profile else None,
            "last_error": cp.metadata.get("last_error"),
            "last_session_kwh": cp.metadata.get("last_session_kwh"),
            "transactions": [
                tx
                for tx in adapter.list_transactions()
                if tx.get("charge_point_id") == cp.charge_point_id
            ],
        }
    )
    return out


@router.get("/ocpp/charge-points")
async def list_charge_points(
    _user=Depends(get_current_user),
    registry: ProtocolRegistry = Depends(get_registry),
):
    """Charge points known to the Central System, with connector state."""
    adapter = _adapter(registry, "ocpp", OCPPAdapter)
    return {
        **_header(adapter),
        "charge_points": [_cp_dict(adapter, cp) for cp in adapter.list_charge_points()],
    }


@router.get("/ocpp/charge-points/{cp_id}")
async def get_charge_point(
    cp_id: str,
    _user=Depends(get_current_user),
    registry: ProtocolRegistry = Depends(get_registry),
):
    adapter = _adapter(registry, "ocpp", OCPPAdapter)
    cp = adapter.get_charge_point(cp_id)
    if cp is None:
        raise HTTPException(status_code=404, detail=f"Charge point {cp_id} not found")
    return {**_header(adapter), **_cp_dict(adapter, cp)}


@router.get("/ocpp/transactions")
async def list_ocpp_transactions(
    _user=Depends(get_current_user),
    registry: ProtocolRegistry = Depends(get_registry),
):
    """Open transactions (StartTransaction seen, StopTransaction not yet)."""
    adapter = _adapter(registry, "ocpp", OCPPAdapter)
    return {**_header(adapter), "transactions": adapter.list_transactions()}


class RemoteStartRequest(BaseModel):
    connector_id: int = Field(1, ge=1, le=100)
    id_tag: str | None = Field(None, min_length=1, max_length=20)


class RemoteStopRequest(BaseModel):
    transaction_id: int | None = None


def _require_cp(adapter: OCPPAdapter, cp_id: str) -> None:
    if adapter.get_charge_point(cp_id) is None and not adapter.is_charge_point_connected(cp_id):
        raise HTTPException(status_code=404, detail=f"Charge point {cp_id} not found")


@router.post("/ocpp/charge-points/{cp_id}/remote-start")
async def remote_start(
    cp_id: str,
    request: Request,
    body: RemoteStartRequest | None = None,
    _user=Depends(require_role("admin", "operator")),
    registry: ProtocolRegistry = Depends(get_registry),
    session: AsyncSession = Depends(get_db),
):
    """Send RemoteStartTransaction; ``accepted`` is the charger's answer (live)."""
    adapter = _adapter(registry, "ocpp", OCPPAdapter)
    _require_cp(adapter, cp_id)
    body = body or RemoteStartRequest()
    accepted = await adapter.remote_start(cp_id, body.connector_id, id_tag=body.id_tag)
    audit.record(
        session,
        request,
        "control.ocpp_remote_start",
        actor=_user,
        target_type="charge_point",
        target_id=cp_id,
        outcome="success" if accepted else "failure",
        details={**body.model_dump(), "accepted": bool(accepted), "mode": adapter.mode.value},
    )
    return {
        **_header(adapter),
        "charge_point_id": cp_id,
        "action": "RemoteStartTransaction",
        "accepted": accepted,
        "live": adapter.is_charge_point_connected(cp_id),
    }


@router.post("/ocpp/charge-points/{cp_id}/remote-stop")
async def remote_stop(
    cp_id: str,
    request: Request,
    body: RemoteStopRequest | None = None,
    _user=Depends(require_role("admin", "operator")),
    registry: ProtocolRegistry = Depends(get_registry),
    session: AsyncSession = Depends(get_db),
):
    """Send RemoteStopTransaction (the given, or the charge point's latest, transaction)."""
    adapter = _adapter(registry, "ocpp", OCPPAdapter)
    _require_cp(adapter, cp_id)
    body = body or RemoteStopRequest()
    accepted = await adapter.remote_stop(cp_id, body.transaction_id)
    audit.record(
        session,
        request,
        "control.ocpp_remote_stop",
        actor=_user,
        target_type="charge_point",
        target_id=cp_id,
        outcome="success" if accepted else "failure",
        details={**body.model_dump(), "accepted": bool(accepted), "mode": adapter.mode.value},
    )
    return {
        **_header(adapter),
        "charge_point_id": cp_id,
        "action": "RemoteStopTransaction",
        "accepted": accepted,
        "live": adapter.is_charge_point_connected(cp_id),
    }


# ---------------------------------------------------------------------------
# OpenADR
# ---------------------------------------------------------------------------


def _event_dict(adapter: OpenADRAdapter, event: Any, now: float) -> dict[str, Any]:
    meta = event.metadata or {}
    response = adapter.get_response(event.event_id)
    out: dict[str, Any] = event.to_dict()
    out.update(
        {
            "end_time": event.end_time,
            "active": event.start_time <= now < event.end_time
            and event.status not in (DREventStatus.CANCELLED, DREventStatus.COMPLETED),
            "current_value": current_signal_value(event, max(now, event.start_time)),
            "opt_type": response.opt_type if response else None,
            "opted_at": response.created_at if response else None,
            "source": meta.get("source", "local"),
            "modification_number": meta.get("modification_number"),
            "priority": meta.get("priority"),
            "test_event": bool(meta.get("test_event")),
            "response_required": meta.get("response_required"),
            "vtn_status": meta.get("vtn_status"),
            "signals": meta.get("signals", []),
        }
    )
    return out


@router.get("/openadr/events")
async def list_openadr_events(
    _user=Depends(get_current_user),
    registry: ProtocolRegistry = Depends(get_registry),
):
    """DR events known to the VEN with their signals and opt state."""
    adapter = _adapter(registry, "openadr", OpenADRAdapter)
    now = time.time()
    return {
        **_header(adapter),
        "ven_id": adapter.ven_id,
        "vtn_id": adapter.vtn_id,
        "registered": adapter.registered,
        "events": [_event_dict(adapter, e, now) for e in adapter.list_events()],
    }


@router.get("/openadr/events/{event_id}")
async def get_openadr_event(
    event_id: str,
    _user=Depends(get_current_user),
    registry: ProtocolRegistry = Depends(get_registry),
):
    adapter = _adapter(registry, "openadr", OpenADRAdapter)
    event = adapter.get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail=f"DR event {event_id} not found")
    return {**_header(adapter), **_event_dict(adapter, event, time.time())}


class OptRequest(BaseModel):
    opt_type: Literal["optIn", "optOut"]
    reason: str = Field("", max_length=500)


@router.post("/openadr/events/{event_id}/opt")
async def override_opt(
    event_id: str,
    body: OptRequest,
    request: Request,
    user=Depends(require_role("admin", "operator")),
    registry: ProtocolRegistry = Depends(get_registry),
    session: AsyncSession = Depends(get_db),
):
    """Operator opt-in/out override.

    VTN events are answered over the wire (``oadrCreatedEvent``) when the VEN
    is live (``sent_to_vtn``); opting out also stops an ongoing automatic
    dispatch for the event.
    """
    adapter = _adapter(registry, "openadr", OpenADRAdapter)
    if adapter.get_event(event_id) is None:
        raise HTTPException(status_code=404, detail=f"DR event {event_id} not found")
    orchestrator = get_dr_orchestrator() or DROrchestrator(
        DRPolicy.from_settings(get_settings()), registry
    )
    try:
        result = await orchestrator.override_opt(
            event_id, body.opt_type, user=getattr(user, "username", None), reason=body.reason
        )
    except Exception as exc:  # VTN unreachable / rejected the response
        audit.record(
            session,
            request,
            "control.dr_opt",
            actor=user,
            target_type="dr_event",
            target_id=event_id,
            outcome="failure",
            details={"opt_type": body.opt_type, "reason": body.reason, "error": str(exc)},
            always=True,
        )
        raise HTTPException(status_code=502, detail=f"opt response failed: {exc}") from exc
    audit.record(
        session,
        request,
        "control.dr_opt",
        actor=user,
        target_type="dr_event",
        target_id=event_id,
        details={"opt_type": body.opt_type, "reason": body.reason},
    )
    return {**_header(adapter), **result}


# ---------------------------------------------------------------------------
# IEEE 2030.5
# ---------------------------------------------------------------------------


@router.get("/ieee2030_5/controls")
async def list_ieee2030_5_controls(
    _user=Depends(get_current_user),
    registry: ProtocolRegistry = Depends(get_registry),
):
    """Active DER controls (highest priority first), programs and defaults."""
    adapter = _adapter(registry, "ieee2030_5", IEEE2030_5Adapter)
    return {
        **_header(adapter),
        "end_device": adapter.end_device_href,
        "server_time": adapter.server_now(),
        "active_controls": [c.to_dict() for c in adapter.get_active_controls()],
        "default_controls": [c.to_dict() for c in adapter.get_default_controls()],
        "programs": [
            {
                "program_id": p.program_id,
                "description": p.description,
                "primacy": p.primacy,
                "href": p.href,
                "control_count": len(p.active_controls),
            }
            for p in adapter.list_programs()
        ],
    }


# ---------------------------------------------------------------------------
# DR orchestrator
# ---------------------------------------------------------------------------


@dr_router.get("/status")
async def dr_status(_user=Depends(get_current_user)):
    """Whether automatic DR response is running, its rules and the active dispatch."""
    orchestrator = get_dr_orchestrator()
    if orchestrator is None:
        policy = DRPolicy.from_settings(get_settings())
        return {
            "running": False,
            "auto_response_enabled": policy.auto_response,
            "policy": policy.to_dict(),
            "active": None,
            "detail": "no OpenADR / IEEE 2030.5 adapter enabled",
        }
    return orchestrator.status()


@dr_router.get("/responses")
async def dr_responses(
    response: Response,
    protocol: str | None = None,
    source_id: str | None = None,
    action: str | None = None,
    page: Page = Depends(page_params(default_limit=100, max_limit=1000)),
    _user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
):
    """Log of DR decisions (received/opt/dispatch/release), newest first.

    Paginated (``limit`` / ``offset``); the total is in ``X-Total-Count``.
    """
    stmt = select(DREventResponseModel).order_by(
        DREventResponseModel.created_at.desc(), DREventResponseModel.id
    )
    if protocol:
        stmt = stmt.where(DREventResponseModel.protocol == protocol)
    if source_id:
        stmt = stmt.where(DREventResponseModel.source_id == source_id)
    if action:
        stmt = stmt.where(DREventResponseModel.action == action)
    rows = await paginate(session, response, stmt, page)
    return [response_to_dict(r) for r in rows]
