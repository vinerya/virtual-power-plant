"""Protocol management API endpoints.

OCPP / OpenADR / IEEE 2030.5 adapters run on the ``protocol-adapters`` lease
holder. With several API workers, a request for one of them that lands on
another worker is forwarded to the holder (``cluster_calls``), and the
adapter list merges the holder's adapters in; in a single process everything
is answered locally, as before.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from vpp import audit
from vpp.auth.security import get_current_user, require_role
from vpp.cluster.lease import is_local
from vpp.cluster.rpc import on_leader, register_handler
from vpp.cluster.topology import LEASE_PROTOCOLS
from vpp.db.engine import get_db
from vpp.protocols.base import ProtocolMode, ProtocolRegistry

router = APIRouter(prefix="/api/v1/protocols", tags=["protocols"])

# Process-wide registry, created at import time.  The app lifespan's MQTT and
# Modbus ingestion loops register their adapters into it via get_registry();
# set_registry() lets tests swap in an isolated instance.
_registry = ProtocolRegistry()


def get_registry() -> ProtocolRegistry:
    return _registry


def set_registry(registry: ProtocolRegistry) -> None:
    global _registry
    _registry = registry


# Adapters started by the protocol-adapters lease holder (vpp.protocols.bootstrap).
LEASE_BOUND_ADAPTERS = frozenset({"ocpp", "openadr", "ieee2030_5"})
# Reads forwarded to the holder give up after this long (503 leader_unavailable).
PROTOCOL_READ_TIMEOUT_S = 3.0


def _forwards(name: str, registry: ProtocolRegistry) -> bool:
    """Whether a request for adapter *name* must go to the protocol-adapters holder."""
    return (
        name in LEASE_BOUND_ADAPTERS
        and registry.get(name) is None
        and not is_local(LEASE_PROTOCOLS)
    )


# -- Schemas -----------------------------------------------------------------


class ProtocolInfo(BaseModel):
    name: str
    version: str
    status: str
    # "live" = talks to a real endpoint; "simulated" = in-memory only, no
    # external traffic (status is then "simulated", never "connected").
    mode: str = "live"
    simulated: bool = False
    messages_sent: int = 0
    messages_received: int = 0
    errors: int = 0
    uptime_seconds: float = 0.0


class ConnectRequest(BaseModel):
    config: dict = {}


class ConnectResponse(BaseModel):
    name: str
    status: str
    message: str = ""


# -- Endpoints ---------------------------------------------------------------


def _info(a: Any) -> ProtocolInfo:
    return ProtocolInfo(
        name=a.name,
        version=a.version,
        status=a.status.value,
        mode=a.mode.value,
        simulated=a.mode == ProtocolMode.SIMULATED,
        messages_sent=a.metrics.messages_sent,
        messages_received=a.metrics.messages_received,
        errors=a.metrics.errors,
        uptime_seconds=a.metrics.uptime_seconds,
    )


@router.get("/", response_model=list[ProtocolInfo])
async def list_protocols(
    _user=Depends(get_current_user),
    registry: ProtocolRegistry = Depends(get_registry),
):
    """List all registered protocol adapters and their status.

    On a worker that does not hold the protocol-adapters lease, the holder's
    OCPP / OpenADR / IEEE 2030.5 adapters are included (forwarded read).
    """
    infos = [_info(a) for a in registry.list_adapters()]
    if not is_local(LEASE_PROTOCOLS):
        remote = await on_leader(
            LEASE_PROTOCOLS,
            "list_adapters",
            {},
            _list_lease_bound_local,
            timeout_s=PROTOCOL_READ_TIMEOUT_S,
        )
        seen = {i.name for i in infos}
        infos.extend(ProtocolInfo.model_validate(r) for r in remote if r["name"] not in seen)
    return infos


async def _list_lease_bound_local() -> list[dict[str, Any]]:
    return [
        _info(a).model_dump()
        for a in get_registry().list_adapters()
        if a.name in LEASE_BOUND_ADAPTERS
    ]


@router.post("/{name}/connect", response_model=ConnectResponse)
async def connect_protocol(
    name: str,
    request: Request,
    body: ConnectRequest | None = None,
    _user=Depends(require_role("admin", "operator")),
    registry: ProtocolRegistry = Depends(get_registry),
    session: AsyncSession = Depends(get_db),
):
    """Connect a protocol adapter."""
    config = body.config if body else {}
    # Only the option names: values may hold endpoint credentials.
    details: dict[str, Any] = {"config_keys": sorted(config)}
    try:
        if _forwards(name, registry):
            result = await on_leader(
                LEASE_PROTOCOLS,
                "connect",
                {"name": name, "config": config},
                lambda: _connect_local(registry, name, config),
            )
        else:
            result = await _connect_local(registry, name, config)
    except HTTPException as exc:
        if exc.status_code == status.HTTP_502_BAD_GATEWAY:
            audit.record(
                session,
                request,
                "control.protocol_connect",
                actor=_user,
                target_type="protocol",
                target_id=name,
                outcome="failure",
                details={**details, "error": str(exc.detail)},
                always=True,
            )
        raise
    audit.record(
        session,
        request,
        "control.protocol_connect",
        actor=_user,
        target_type="protocol",
        target_id=name,
        details={**details, "status": result.get("status")},
    )
    return result


async def _connect_local(
    registry: ProtocolRegistry, name: str, config: dict[str, Any]
) -> dict[str, Any]:
    adapter = registry.get(name)
    if adapter is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Protocol '{name}' not found"
        )

    if adapter.is_operational:
        return ConnectResponse(
            name=name,
            status=adapter.status.value,
            message="Already running (simulated)" if adapter.is_simulated else "Already connected",
        ).model_dump()

    if config:
        adapter.configure(**config)

    try:
        await adapter.connect()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Connection failed: {exc}",
        ) from exc

    if adapter.is_simulated:
        return ConnectResponse(
            name=name,
            status=adapter.status.value,
            message="Running in simulated mode: no real endpoint configured",
        ).model_dump()
    return ConnectResponse(
        name=name, status=adapter.status.value, message="Connected"
    ).model_dump()


@router.post("/{name}/disconnect", response_model=ConnectResponse)
async def disconnect_protocol(
    name: str,
    request: Request,
    _user=Depends(require_role("admin", "operator")),
    registry: ProtocolRegistry = Depends(get_registry),
    session: AsyncSession = Depends(get_db),
):
    """Disconnect a protocol adapter."""
    if _forwards(name, registry):
        result = await on_leader(
            LEASE_PROTOCOLS,
            "disconnect",
            {"name": name},
            lambda: _disconnect_local(registry, name),
        )
    else:
        result = await _disconnect_local(registry, name)
    audit.record(
        session,
        request,
        "control.protocol_disconnect",
        actor=_user,
        target_type="protocol",
        target_id=name,
    )
    return result


async def _disconnect_local(registry: ProtocolRegistry, name: str) -> dict[str, Any]:
    adapter = registry.get(name)
    if adapter is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Protocol '{name}' not found"
        )

    await adapter.disconnect()
    return ConnectResponse(
        name=name, status=adapter.status.value, message="Disconnected"
    ).model_dump()


@router.get("/{name}/metrics")
async def protocol_metrics(
    name: str,
    _user=Depends(get_current_user),
    registry: ProtocolRegistry = Depends(get_registry),
):
    """Get detailed metrics for a protocol adapter."""
    if _forwards(name, registry):
        return await on_leader(
            LEASE_PROTOCOLS,
            "metrics",
            {"name": name},
            lambda: _metrics_local(registry, name),
            timeout_s=PROTOCOL_READ_TIMEOUT_S,
        )
    return await _metrics_local(registry, name)


async def _metrics_local(registry: ProtocolRegistry, name: str) -> dict[str, Any]:
    adapter = registry.get(name)
    if adapter is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Protocol '{name}' not found"
        )

    m = adapter.metrics
    return {
        "name": adapter.name,
        "version": adapter.version,
        "status": adapter.status.value,
        "mode": adapter.mode.value,
        "messages_sent": m.messages_sent,
        "messages_received": m.messages_received,
        "errors": m.errors,
        "uptime_seconds": round(m.uptime_seconds, 1),
        "last_message_at": m.last_message_at,
        "reconnect_count": m.reconnect_count,
    }


# Handlers the protocol-adapters holder runs for calls forwarded by other workers.


async def _h_list_adapters(payload: dict[str, Any], session: Any) -> Any:
    return await _list_lease_bound_local()


async def _h_connect(payload: dict[str, Any], session: Any) -> Any:
    return await _connect_local(
        get_registry(), str(payload["name"]), dict(payload.get("config") or {})
    )


async def _h_disconnect(payload: dict[str, Any], session: Any) -> Any:
    return await _disconnect_local(get_registry(), str(payload["name"]))


async def _h_metrics(payload: dict[str, Any], session: Any) -> Any:
    return await _metrics_local(get_registry(), str(payload["name"]))


for _method, _handler in (
    ("list_adapters", _h_list_adapters),
    ("connect", _h_connect),
    ("disconnect", _h_disconnect),
    ("metrics", _h_metrics),
):
    register_handler(LEASE_PROTOCOLS, _method, _handler)
