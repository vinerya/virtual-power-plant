"""Protocol management API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from vpp.auth.security import get_current_user, require_role
from vpp.db.engine import get_db
from vpp.protocols.base import ProtocolRegistry, ProtocolStatus

router = APIRouter(prefix="/api/v1/protocols", tags=["protocols"])

# Module-level registry instance (wired up in app.py lifespan)
_registry = ProtocolRegistry()


def get_registry() -> ProtocolRegistry:
    return _registry


def set_registry(registry: ProtocolRegistry) -> None:
    global _registry
    _registry = registry


# -- Schemas -----------------------------------------------------------------

class ProtocolInfo(BaseModel):
    name: str
    version: str
    status: str
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

@router.get("/", response_model=list[ProtocolInfo])
async def list_protocols(
    _user=Depends(get_current_user),
    registry: ProtocolRegistry = Depends(get_registry),
):
    """List all registered protocol adapters and their status."""
    return [
        ProtocolInfo(
            name=a.name,
            version=a.version,
            status=a.status.value,
            messages_sent=a.metrics.messages_sent,
            messages_received=a.metrics.messages_received,
            errors=a.metrics.errors,
            uptime_seconds=a.metrics.uptime_seconds,
        )
        for a in registry.list_adapters()
    ]


@router.post("/{name}/connect", response_model=ConnectResponse)
async def connect_protocol(
    name: str,
    body: ConnectRequest | None = None,
    _user=Depends(require_role("admin", "operator")),
    registry: ProtocolRegistry = Depends(get_registry),
):
    """Connect a protocol adapter."""
    adapter = registry.get(name)
    if adapter is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Protocol '{name}' not found")

    if adapter.is_connected:
        return ConnectResponse(name=name, status=adapter.status.value, message="Already connected")

    if body and body.config:
        adapter.configure(**body.config)

    try:
        await adapter.connect()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Connection failed: {exc}",
        )

    return ConnectResponse(name=name, status=adapter.status.value, message="Connected")


@router.post("/{name}/disconnect", response_model=ConnectResponse)
async def disconnect_protocol(
    name: str,
    _user=Depends(require_role("admin", "operator")),
    registry: ProtocolRegistry = Depends(get_registry),
):
    """Disconnect a protocol adapter."""
    adapter = registry.get(name)
    if adapter is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Protocol '{name}' not found")

    await adapter.disconnect()
    return ConnectResponse(name=name, status=adapter.status.value, message="Disconnected")


@router.get("/{name}/metrics")
async def protocol_metrics(
    name: str,
    _user=Depends(get_current_user),
    registry: ProtocolRegistry = Depends(get_registry),
):
    """Get detailed metrics for a protocol adapter."""
    adapter = registry.get(name)
    if adapter is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Protocol '{name}' not found")

    m = adapter.metrics
    return {
        "name": adapter.name,
        "version": adapter.version,
        "status": adapter.status.value,
        "messages_sent": m.messages_sent,
        "messages_received": m.messages_received,
        "errors": m.errors,
        "uptime_seconds": round(m.uptime_seconds, 1),
        "last_message_at": m.last_message_at,
        "reconnect_count": m.reconnect_count,
    }
