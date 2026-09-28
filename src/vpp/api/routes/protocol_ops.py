"""Protocol data (read-only) and operator actions for OCPP / OpenADR / IEEE 2030.5.

Reads are open to every operator-side user (``get_current_user``: customer
accounts are refused); actions (remote start/stop, opt-in/out override)
require the ``admin`` or ``operator`` role. Every response carries the
adapter's ``mode`` so a simulated adapter is never mistaken for real
hardware / a real utility connection.

The OpenADR / IEEE 2030.5 / DR-orchestrator views read state held by the
``protocol-adapters`` lease holder; other API workers forward them there
(see ``PROTOCOL_READ_TIMEOUT_S``).
"""

from __future__ import annotations

import time
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vpp.api.routes.protocols import PROTOCOL_READ_TIMEOUT_S, get_registry
from vpp.auth.security import get_current_user, require_role
from vpp.cluster.rpc import on_leader, register_handler
from vpp.cluster.topology import LEASE_PROTOCOLS
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
    body: RemoteStartRequest | None = None,
    _user=Depends(require_role("admin", "operator")),
    registry: ProtocolRegistry = Depends(get_registry),
):
    """Send RemoteStartTransaction; ``accepted`` is the charger's answer (live)."""
    adapter = _adapter(registry, "ocpp", OCPPAdapter)
    _require_cp(adapter, cp_id)
    body = body or RemoteStartRequest()
    accepted = await adapter.remote_start(cp_id, body.connector_id, id_tag=body.id_tag)
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
    body: RemoteStopRequest | None = None,
    _user=Depends(require_role("admin", "operator")),
    registry: ProtocolRegistry = Depends(get_registry),
):
    """Send RemoteStopTransaction (the given, or the charge point's latest, transaction)."""
    adapter = _adapter(registry, "ocpp", OCPPAdapter)
    _require_cp(adapter, cp_id)
    body = body or RemoteStopRequest()
    accepted = await adapter.remote_stop(cp_id, body.transaction_id)
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


# The OpenADR VEN, the IEEE 2030.5 client and the DR orchestrator run on the
# ``protocol-adapters`` lease holder (one process per deployment). With
# several API workers, a request that lands elsewhere is forwarded to the
# holder through ``cluster_calls`` (``on_leader``); in a single process (or
# on the holder) the handler runs directly, with no database round trip.
# Reads fail fast: if the holder does not answer within
# ``PROTOCOL_READ_TIMEOUT_S`` (capped by VPP_CLUSTER_CALL_TIMEOUT_SECONDS) the
# client gets ``503 leader_unavailable``.


async def _openadr_events_local(registry: ProtocolRegistry) -> dict[str, Any]:
    adapter = _adapter(registry, "openadr", OpenADRAdapter)
    now = time.time()
    return {
        **_header(adapter),
        "ven_id": adapter.ven_id,
        "vtn_id": adapter.vtn_id,
        "registered": adapter.registered,
        "events": [_event_dict(adapter, e, now) for e in adapter.list_events()],
    }


async def _openadr_event_local(registry: ProtocolRegistry, event_id: str) -> dict[str, Any]:
    adapter = _adapter(registry, "openadr", OpenADRAdapter)
    event = adapter.get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail=f"DR event {event_id} not found")
    return {**_header(adapter), **_event_dict(adapter, event, time.time())}


@router.get("/openadr/events")
async def list_openadr_events(
    _user=Depends(get_current_user),
    registry: ProtocolRegistry = Depends(get_registry),
):
    """DR events known to the VEN with their signals and opt state."""
    return await on_leader(
        LEASE_PROTOCOLS,
        "openadr_events",
        {},
        lambda: _openadr_events_local(registry),
        timeout_s=PROTOCOL_READ_TIMEOUT_S,
    )


@router.get("/openadr/events/{event_id}")
async def get_openadr_event(
    event_id: str,
    _user=Depends(get_current_user),
    registry: ProtocolRegistry = Depends(get_registry),
):
    return await on_leader(
        LEASE_PROTOCOLS,
        "openadr_event",
        {"event_id": event_id},
        lambda: _openadr_event_local(registry, event_id),
        timeout_s=PROTOCOL_READ_TIMEOUT_S,
    )


class OptRequest(BaseModel):
    opt_type: Literal["optIn", "optOut"]
    reason: str = Field("", max_length=500)


async def _override_opt_local(
    registry: ProtocolRegistry, event_id: str, body: OptRequest, username: str | None
) -> dict[str, Any]:
    adapter = _adapter(registry, "openadr", OpenADRAdapter)
    if adapter.get_event(event_id) is None:
        raise HTTPException(status_code=404, detail=f"DR event {event_id} not found")
    orchestrator = get_dr_orchestrator() or DROrchestrator(
        DRPolicy.from_settings(get_settings()), registry
    )
    try:
        result = await orchestrator.override_opt(
            event_id, body.opt_type, user=username, reason=body.reason
        )
    except Exception as exc:  # VTN unreachable / rejected the response
        raise HTTPException(status_code=502, detail=f"opt response failed: {exc}") from exc
    return {**_header(adapter), **result}


@router.post("/openadr/events/{event_id}/opt")
async def override_opt(
    event_id: str,
    body: OptRequest,
    user=Depends(require_role("admin", "operator")),
    registry: ProtocolRegistry = Depends(get_registry),
):
    """Operator opt-in/out override.

    VTN events are answered over the wire (``oadrCreatedEvent``) when the VEN
    is live (``sent_to_vtn``); opting out also stops an ongoing automatic
    dispatch for the event.
    """
    username = getattr(user, "username", None)
    return await on_leader(
        LEASE_PROTOCOLS,
        "openadr_opt",
        {"event_id": event_id, "body": body.model_dump(mode="json"), "username": username},
        lambda: _override_opt_local(registry, event_id, body, username),
    )


# ---------------------------------------------------------------------------
# IEEE 2030.5
# ---------------------------------------------------------------------------


async def _ieee2030_5_controls_local(registry: ProtocolRegistry) -> dict[str, Any]:
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


@router.get("/ieee2030_5/controls")
async def list_ieee2030_5_controls(
    _user=Depends(get_current_user),
    registry: ProtocolRegistry = Depends(get_registry),
):
    """Active DER controls (highest priority first), programs and defaults."""
    return await on_leader(
        LEASE_PROTOCOLS,
        "ieee2030_5_controls",
        {},
        lambda: _ieee2030_5_controls_local(registry),
        timeout_s=PROTOCOL_READ_TIMEOUT_S,
    )


# ---------------------------------------------------------------------------
# DR orchestrator
# ---------------------------------------------------------------------------


async def _dr_status_local() -> dict[str, Any]:
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


@dr_router.get("/status")
async def dr_status(_user=Depends(get_current_user)):
    """Whether automatic DR response is running, its rules and the active dispatch."""
    return await on_leader(
        LEASE_PROTOCOLS,
        "dr_status",
        {},
        _dr_status_local,
        timeout_s=PROTOCOL_READ_TIMEOUT_S,
    )


# Handlers the protocol-adapters holder runs for calls forwarded by other workers.


async def _h_openadr_events(payload: dict[str, Any], session: AsyncSession) -> Any:
    return await _openadr_events_local(get_registry())


async def _h_openadr_event(payload: dict[str, Any], session: AsyncSession) -> Any:
    return await _openadr_event_local(get_registry(), str(payload["event_id"]))


async def _h_openadr_opt(payload: dict[str, Any], session: AsyncSession) -> Any:
    body = OptRequest.model_validate(payload["body"])
    return await _override_opt_local(
        get_registry(), str(payload["event_id"]), body, payload.get("username")
    )


async def _h_ieee2030_5_controls(payload: dict[str, Any], session: AsyncSession) -> Any:
    return await _ieee2030_5_controls_local(get_registry())


async def _h_dr_status(payload: dict[str, Any], session: AsyncSession) -> Any:
    return await _dr_status_local()


for _method, _handler in (
    ("openadr_events", _h_openadr_events),
    ("openadr_event", _h_openadr_event),
    ("openadr_opt", _h_openadr_opt),
    ("ieee2030_5_controls", _h_ieee2030_5_controls),
    ("dr_status", _h_dr_status),
):
    register_handler(LEASE_PROTOCOLS, _method, _handler)


@dr_router.get("/responses")
async def dr_responses(
    protocol: str | None = None,
    source_id: str | None = None,
    action: str | None = None,
    limit: int = Query(100, ge=1, le=1000),
    _user=Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
):
    """Audit log of DR decisions (received/opt/dispatch/release), newest first."""
    stmt = select(DREventResponseModel).order_by(DREventResponseModel.created_at.desc())
    if protocol:
        stmt = stmt.where(DREventResponseModel.protocol == protocol)
    if source_id:
        stmt = stmt.where(DREventResponseModel.source_id == source_id)
    if action:
        stmt = stmt.where(DREventResponseModel.action == action)
    rows = (await session.execute(stmt.limit(limit))).scalars().all()
    return [response_to_dict(r) for r in rows]
