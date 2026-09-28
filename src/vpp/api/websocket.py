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

Credential expiry
-----------------
An authenticated socket does not outlive the credential behind it: it is
closed with code **4001** (reason ``"Token expired"``) when

* the session it was minted from expires, for tokens from
  ``POST /api/v1/ws/token`` (the ``sexp`` claim -- the session JWT's ``exp``,
  or ``now + VPP_JWT_EXPIRE_MINUTES`` when minted with an API key); the
  token's own short ``exp`` only bounds the handshake;
* the JWT itself expires, for regular access tokens used directly.

Clients should reconnect with a fresh token (the console does so
automatically); if the session is gone, minting a new token fails with 401.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import jwt
from pydantic import BaseModel

from vpp.auth.security import decode_access_token, get_current_user
from vpp.db.models import UserModel  # noqa: TC001 -- runtime annotation (FastAPI)
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
    "grid_events",
    "system",
    "*",
}


# Maps EventBus event types to the WebSocket channel operators subscribe to.
# "alerts" is reserved for alert-shaped payloads (AlertService broadcasts
# those directly and the console toasts every message on it), so anything
# not listed here falls through to "system" rather than being dropped.
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
    EventType.PROTOCOL_CONNECTED: "grid_events",
    EventType.PROTOCOL_DISCONNECTED: "grid_events",
    EventType.PROTOCOL_ERROR: "grid_events",
    EventType.DR_EVENT_RECEIVED: "grid_events",
    EventType.DR_RESPONSE_SENT: "grid_events",
    EventType.EV_CONNECTED: "grid_events",
    EventType.EV_DISCONNECTED: "grid_events",
    EventType.V2G_DISPATCH: "grid_events",
    EventType.V2G_SCHEDULE_CREATED: "grid_events",
    EventType.ISLAND_DETECTED: "grid_events",
    EventType.ISLAND_ENTERED: "grid_events",
    EventType.GRID_RECONNECTED: "grid_events",
    EventType.LOAD_SHED: "grid_events",
    EventType.ALERT_TRIGGERED: "alerts",
}


def event_to_channel(event_type: EventType) -> str:
    """Map an EventBus event type to a WebSocket broadcast channel."""
    return _EVENT_CHANNEL_MAP.get(event_type, "system")


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
#: Application close code: the socket's credential expired (reconnect).
WS_TOKEN_EXPIRED = 4001
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


def create_ws_token(
    user: UserModel, session_expires_at: datetime | None = None
) -> tuple[str, int]:
    """Mint a short-lived JWT that is only accepted by the WebSocket endpoint.

    ``session_expires_at`` is the expiry of the credential the caller used;
    it is embedded as ``sexp`` (sockets opened with the token are closed at
    that time) and caps the token's own ``exp``. Defaults to a fresh
    session lifetime (``VPP_JWT_EXPIRE_MINUTES``), e.g. for API keys.
    """
    settings = get_settings()
    now = datetime.now(timezone.utc)
    if session_expires_at is None:
        session_expires_at = now + timedelta(minutes=settings.jwt_expire_minutes)
    ttl = max(1, settings.ws_token_expire_seconds)
    expire = min(now + timedelta(seconds=ttl), session_expires_at)
    ttl = max(1, int((expire - now).total_seconds()))
    token = jwt.encode(
        {
            "sub": user.id,
            "username": user.username,
            "role": user.role,
            "typ": WS_TOKEN_TYPE,
            "exp": expire,
            "sexp": int(session_expires_at.timestamp()),
        },
        settings.secret_key,
        algorithm=settings.jwt_algorithm,
    )
    return token, ttl


def socket_expires_at(payload: TokenPayload) -> float:
    """Epoch seconds at which a socket opened with ``payload`` must close."""
    if payload.typ == WS_TOKEN_TYPE and payload.sexp is not None:
        return float(payload.sexp)
    return float(payload.exp)


class WsTokenResponse(BaseModel):
    token: str
    expires_in: int
    channels: list[str]


router = APIRouter(prefix="/api/v1/ws", tags=["WebSocket"])


_optional_bearer = HTTPBearer(auto_error=False)


@router.post("/token", response_model=WsTokenResponse)
async def issue_ws_token(
    user: UserModel = Depends(get_current_user),
    bearer: HTTPAuthorizationCredentials | None = Depends(_optional_bearer),
) -> WsTokenResponse:
    """Exchange the caller's credentials for a short-lived WebSocket token.

    The token is only valid for opening a socket (the HTTP API rejects it)
    and must be used within ``VPP_WS_TOKEN_EXPIRE_SECONDS`` (default 60s).
    A socket opened with it stays open until the caller's *session* expires
    (the bearer JWT's ``exp``; for API keys, ``VPP_JWT_EXPIRE_MINUTES`` from
    now) and is then closed with code 4001.
    """
    session_exp: datetime | None = None
    if bearer is not None:
        # Already validated by get_current_user (bearer takes precedence).
        exp = decode_access_token(bearer.credentials).exp
        session_exp = datetime.fromtimestamp(exp, tz=timezone.utc)
    token, ttl = create_ws_token(user, session_exp)
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

    expires_at: float | None = None
    if token is not None:
        payload = await authenticate_websocket(token)
        if payload is None:
            await ws.close(code=WS_POLICY_VIOLATION, reason="Invalid or expired token")
            return
        expires_at = socket_expires_at(payload)
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
            if expires_at is None:
                raw = await ws.receive_text()
            else:
                remaining = expires_at - datetime.now(timezone.utc).timestamp()
                try:
                    if remaining <= 0:
                        raise TimeoutError
                    raw = await asyncio.wait_for(ws.receive_text(), timeout=remaining)
                except TimeoutError:
                    await manager.disconnect(ws)
                    await ws.close(code=WS_TOKEN_EXPIRED, reason="Token expired")
                    return
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
