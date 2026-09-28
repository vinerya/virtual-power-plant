"""WebSocket endpoint for real-time event streaming.

Served at both ``/api/v1/ws`` (canonical) and ``/ws`` (legacy alias). The
wire protocol is identical on both paths — see :func:`websocket_endpoint`.

Authentication
--------------
Browsers cannot set an ``Authorization`` header on a WebSocket upgrade, so a
JWT is accepted from (first match wins):

1. the ``token`` query parameter,
2. the ``Sec-WebSocket-Protocol`` header as the pair ``bearer, <jwt>`` — the
   server then selects the ``bearer`` subprotocol, as the spec requires,
3. an ``Authorization: Bearer <jwt>`` header (non-browser clients).

Browser clients should first mint a short-lived token from
``POST /api/v1/ws/token`` (authenticated like any other API call) rather than
exposing the long-lived session token to JavaScript.

When ``VPP_WS_AUTH_REQUIRED`` is true (the default) an unauthenticated or
invalid handshake is refused with close code 1008 (policy violation), which
ASGI servers surface as an HTTP 403 on the upgrade request.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from jose import jwt
from pydantic import BaseModel

from vpp.auth.security import decode_access_token, get_current_user
from vpp.db.models import UserModel
from vpp.events.bus import Event, EventBus, EventType
from vpp.schemas.auth import TokenPayload, UserRole
from vpp.settings import get_settings

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections and channel subscriptions."""

    def __init__(self) -> None:
        self._connections: dict[WebSocket, set[str]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket, subprotocol: str | None = None) -> None:
        if subprotocol is not None:
            await ws.accept(subprotocol=subprotocol)
        else:
            await ws.accept()
        async with self._lock:
            self._connections[ws] = set()

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._connections.pop(ws, None)

    async def subscribe(self, ws: WebSocket, channel: str) -> None:
        async with self._lock:
            if ws in self._connections:
                self._connections[ws].add(channel)

    async def unsubscribe(self, ws: WebSocket, channel: str) -> None:
        async with self._lock:
            if ws in self._connections:
                self._connections[ws].discard(channel)

    async def broadcast(self, channel: str, data: dict[str, Any]) -> None:
        """Send a message to all subscribers of *channel*."""
        message = json.dumps(
            {
                "channel": channel,
                "data": data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        async with self._lock:
            targets = [
                ws
                for ws, channels in self._connections.items()
                if channel in channels or "*" in channels
            ]
        for ws in targets:
            try:
                await ws.send_text(message)
            except Exception:
                await self.disconnect(ws)

    @property
    def active_count(self) -> int:
        return len(self._connections)


# Singleton
manager = ConnectionManager()

VALID_CHANNELS = {
    "resource_updates",
    "optimization_events",
    "market_data",
    "alerts",
    "*",
}


# Maps EventBus event types to the WebSocket channel operators subscribe to.
# Anything not listed here falls through to "alerts" so it's never silently
# dropped — see event_to_channel().
_EVENT_CHANNEL_MAP: dict[EventType, str] = {
    EventType.RESOURCE_ADDED: "resource_updates",
    EventType.RESOURCE_REMOVED: "resource_updates",
    EventType.RESOURCE_UPDATED: "resource_updates",
    EventType.RESOURCE_FAULT: "resource_updates",
    EventType.OPTIMIZATION_STARTED: "optimization_events",
    EventType.OPTIMIZATION_COMPLETED: "optimization_events",
    EventType.OPTIMIZATION_FAILED: "optimization_events",
    EventType.DISPATCH_EXECUTED: "optimization_events",
    EventType.ORDER_SUBMITTED: "market_data",
    EventType.ORDER_FILLED: "market_data",
    EventType.ORDER_CANCELLED: "market_data",
    EventType.ORDER_REJECTED: "market_data",
    EventType.TRADE_EXECUTED: "market_data",
    EventType.MARKET_DATA: "market_data",
}


def event_to_channel(event_type: EventType) -> str:
    """Map an EventBus event type to a WebSocket broadcast channel."""
    return _EVENT_CHANNEL_MAP.get(event_type, "alerts")


def subscribe_event_bus_to_websocket(bus: EventBus, mgr: ConnectionManager) -> str:
    """Bridge EventBus publishes to WebSocket broadcasts.

    Without this bridge, events published to the EventBus never reach
    connected WebSocket clients — the two pub/sub systems are otherwise
    entirely disconnected. Returns the EventBus subscription id so callers
    can unsubscribe on shutdown.
    """

    async def _forward(event: Event) -> None:
        await mgr.broadcast(event_to_channel(event.event_type), event.to_dict())

    return bus.subscribe(_forward)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

WS_POLICY_VIOLATION = 1008
WS_TOKEN_TYPE = "ws"
_BEARER_SUBPROTOCOL = "bearer"


def _extract_token(ws: WebSocket) -> tuple[str | None, str | None]:
    """Return ``(token, subprotocol_to_select)`` from the handshake."""
    protocols = [
        p.strip() for p in ws.headers.get("sec-websocket-protocol", "").split(",") if p.strip()
    ]
    selected = _BEARER_SUBPROTOCOL if _BEARER_SUBPROTOCOL in protocols else None

    query_token = ws.query_params.get("token")
    if query_token:
        return query_token, selected

    if selected is not None:
        idx = protocols.index(_BEARER_SUBPROTOCOL)
        if idx + 1 < len(protocols):
            return protocols[idx + 1], selected

    auth = ws.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None, selected
    return None, selected


async def _load_active_user(user_id: str) -> UserModel | None:
    """Look up an active user by id. Returns None if missing/inactive/no DB."""
    from vpp.db.engine import get_session_factory
    from vpp.db.repositories import UserRepository

    try:
        factory = get_session_factory()
    except RuntimeError:
        logger.warning("WebSocket auth: database not initialised; refusing socket")
        return None
    async with factory() as session:
        user = await UserRepository.get_by_id(session, user_id)
    if user is None or not user.is_active:
        return None
    return user


async def authenticate_websocket(token: str) -> TokenPayload | None:
    """Validate a JWT for a WebSocket handshake; None if it is not acceptable."""
    try:
        payload = decode_access_token(token)
    except HTTPException:
        return None
    user = await _load_active_user(payload.sub)
    if user is None:
        return None
    # Every channel carries fleet-wide data, so customer accounts (member
    # portal) are refused, mirroring get_current_user on the HTTP API.
    if user.role == UserRole.CUSTOMER.value:
        return None
    return payload


def create_ws_token(user: UserModel) -> tuple[str, int]:
    """Mint a short-lived JWT that is only accepted by the WebSocket endpoint."""
    settings = get_settings()
    ttl = max(1, settings.ws_token_expire_seconds)
    expire = datetime.now(timezone.utc) + timedelta(seconds=ttl)
    token = jwt.encode(
        {
            "sub": user.id,
            "username": user.username,
            "role": user.role,
            "typ": WS_TOKEN_TYPE,
            "exp": expire,
        },
        settings.secret_key,
        algorithm=settings.jwt_algorithm,
    )
    return token, ttl


class WsTokenResponse(BaseModel):
    token: str
    expires_in: int
    channels: list[str]


router = APIRouter(prefix="/api/v1/ws", tags=["WebSocket"])


@router.post("/token", response_model=WsTokenResponse)
async def issue_ws_token(user: UserModel = Depends(get_current_user)) -> WsTokenResponse:
    """Exchange the caller's credentials for a short-lived WebSocket token.

    The token is only valid for opening a socket (the HTTP API rejects it)
    and expires after ``VPP_WS_TOKEN_EXPIRE_SECONDS`` (default 60s). Expiry
    only gates the handshake; an already-open socket is not closed.
    """
    token, ttl = create_ws_token(user)
    return WsTokenResponse(token=token, expires_in=ttl, channels=sorted(VALID_CHANNELS))


def _parse_channels(raw: str | None) -> tuple[list[str], list[str]]:
    """Split a ``channels`` query value into (valid, invalid) lists."""
    if not raw:
        return [], []
    valid: list[str] = []
    invalid: list[str] = []
    for part in raw.split(","):
        ch = part.strip()
        if not ch:
            continue
        (valid if ch in VALID_CHANNELS else invalid).append(ch)
    return valid, invalid


async def websocket_endpoint(ws: WebSocket) -> None:
    """Handle a WebSocket connection (served at ``/api/v1/ws`` and ``/ws``).

    Handshake query parameters (all optional):
        ``token``    — JWT (see module docstring for alternatives)
        ``channels`` — comma-separated initial subscriptions, e.g.
                       ``resource_updates,alerts``

    Protocol (JSON messages):
        Client → Server:
            {"action": "subscribe", "channel": "resource_updates"}
            {"action": "unsubscribe", "channel": "resource_updates"}
            {"action": "ping"}
        Server → Client:
            {"channel": "...", "data": {...}, "timestamp": "..."}   # broadcast
            {"ack": "subscribed:<channel>"} / {"ack": "unsubscribed:<channel>"}
            {"pong": "<iso timestamp>"}
            {"error": "..."}

    For EventBus-originated broadcasts ``data`` is ``Event.to_dict()``:
    ``{"event_id", "event_type", "data", "source", "severity", "timestamp"}``.
    """
    settings = get_settings()
    token, subprotocol = _extract_token(ws)

    if token is not None:
        if await authenticate_websocket(token) is None:
            await ws.close(code=WS_POLICY_VIOLATION, reason="Invalid or expired token")
            return
    elif settings.ws_auth_required:
        await ws.close(code=WS_POLICY_VIOLATION, reason="Authentication required")
        return

    await manager.connect(ws, subprotocol=subprotocol)
    try:
        initial, rejected = _parse_channels(ws.query_params.get("channels"))
        for channel in initial:
            await manager.subscribe(ws, channel)
            await ws.send_text(json.dumps({"ack": f"subscribed:{channel}"}))
        for channel in rejected:
            await ws.send_text(json.dumps({"error": f"Unknown channel: {channel}"}))

        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_text(json.dumps({"error": "Invalid JSON"}))
                continue
            if not isinstance(msg, dict):
                await ws.send_text(json.dumps({"error": "Expected a JSON object"}))
                continue

            action = msg.get("action")
            channel = msg.get("channel", "")

            if action == "ping":
                await ws.send_text(json.dumps({"pong": datetime.now(timezone.utc).isoformat()}))
            elif action == "subscribe" and channel in VALID_CHANNELS:
                await manager.subscribe(ws, channel)
                await ws.send_text(json.dumps({"ack": f"subscribed:{channel}"}))
            elif action == "unsubscribe" and channel in VALID_CHANNELS:
                await manager.unsubscribe(ws, channel)
                await ws.send_text(json.dumps({"ack": f"unsubscribed:{channel}"}))
            else:
                await ws.send_text(
                    json.dumps({"error": f"Unknown action or channel: {action}/{channel}"})
                )
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(ws)
