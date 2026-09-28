"""Tests for WebSocket authentication, paths and the initial-subscription handshake.

The handshake tests drive a minimal FastAPI app through starlette's
TestClient (which runs its own event loop), so the user lookup is stubbed;
the DB-backed pieces (token minting, real user lookup) are covered separately
through the shared async ``client`` fixture.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from jose import jwt
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from vpp.api import websocket as ws_module
from vpp.auth.security import create_access_token
from vpp.settings import get_settings


class _User:
    def __init__(self, uid: str = "u-1") -> None:
        self.id = uid
        self.username = "alice"
        self.role = "operator"
        self.is_active = True


@pytest.fixture
def ws_app(monkeypatch) -> FastAPI:
    known = {"u-1": _User("u-1")}

    async def _fake_load(user_id: str):
        return known.get(user_id)

    monkeypatch.setattr(ws_module, "_load_active_user", _fake_load)
    app = FastAPI()
    app.add_api_websocket_route("/api/v1/ws", ws_module.websocket_endpoint)
    app.add_api_websocket_route("/ws", ws_module.websocket_endpoint)
    return app


def _token(sub: str = "u-1") -> str:
    return create_access_token({"sub": sub, "username": "alice", "role": "operator"})


@pytest.mark.parametrize("path", ["/api/v1/ws", "/ws"])
def test_unauthenticated_socket_is_rejected(ws_app, path):
    with TestClient(ws_app) as tc, pytest.raises(WebSocketDisconnect) as exc:
        with tc.websocket_connect(path) as ws:
            ws.receive_text()
    assert exc.value.code == ws_module.WS_POLICY_VIOLATION


def test_invalid_token_is_rejected(ws_app):
    with TestClient(ws_app) as tc, pytest.raises(WebSocketDisconnect) as exc:
        with tc.websocket_connect("/api/v1/ws?token=not.a.jwt") as ws:
            ws.receive_text()
    assert exc.value.code == ws_module.WS_POLICY_VIOLATION


def test_token_for_unknown_user_is_rejected(ws_app):
    with TestClient(ws_app) as tc, pytest.raises(WebSocketDisconnect):
        with tc.websocket_connect(f"/api/v1/ws?token={_token('ghost')}") as ws:
            ws.receive_text()


@pytest.mark.parametrize("path", ["/api/v1/ws", "/ws"])
def test_query_token_and_initial_channels(ws_app, path):
    url = f"{path}?token={_token()}&channels=resource_updates,alerts,bogus"
    with TestClient(ws_app) as tc, tc.websocket_connect(url) as ws:
        assert json.loads(ws.receive_text()) == {"ack": "subscribed:resource_updates"}
        assert json.loads(ws.receive_text()) == {"ack": "subscribed:alerts"}
        assert "bogus" in json.loads(ws.receive_text())["error"]
        # Message protocol still works after the handshake.
        ws.send_text(json.dumps({"action": "subscribe", "channel": "market_data"}))
        assert json.loads(ws.receive_text()) == {"ack": "subscribed:market_data"}
        ws.send_text(json.dumps({"action": "ping"}))
        assert "pong" in json.loads(ws.receive_text())


def test_subprotocol_token_selects_bearer(ws_app):
    with (
        TestClient(ws_app) as tc,
        tc.websocket_connect(
            "/api/v1/ws?channels=alerts", subprotocols=["bearer", _token()]
        ) as ws,
    ):
        assert ws.accepted_subprotocol == "bearer"
        assert json.loads(ws.receive_text()) == {"ack": "subscribed:alerts"}


def test_authorization_header_is_accepted(ws_app):
    with (
        TestClient(ws_app) as tc,
        tc.websocket_connect("/api/v1/ws", headers={"Authorization": f"Bearer {_token()}"}) as ws,
    ):
        ws.send_text(json.dumps({"action": "subscribe", "channel": "alerts"}))
        assert json.loads(ws.receive_text()) == {"ack": "subscribed:alerts"}


def test_auth_can_be_disabled(ws_app, monkeypatch):
    from vpp.settings import get_settings

    monkeypatch.setattr(get_settings(), "ws_auth_required", False)
    with TestClient(ws_app) as tc, tc.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"action": "subscribe", "channel": "alerts"}))
        assert json.loads(ws.receive_text()) == {"ack": "subscribed:alerts"}


def test_broadcast_reaches_socket_subscribed_via_query(ws_app):
    with TestClient(ws_app) as tc:
        with tc.websocket_connect(f"/api/v1/ws?token={_token()}&channels=alerts") as ws:
            assert json.loads(ws.receive_text()) == {"ack": "subscribed:alerts"}
            tc.portal.call(ws_module.manager.broadcast, "alerts", {"title": "hi"})
            msg = json.loads(ws.receive_text())
            assert msg["channel"] == "alerts"
            assert msg["data"] == {"title": "hi"}
            assert "timestamp" in msg


# ---------------------------------------------------------------------------
# DB-backed: token minting + real user lookup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ws_token_endpoint_requires_auth(client: AsyncClient):
    resp = await client.post("/api/v1/ws/token")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_ws_token_is_valid_for_socket_but_not_http(client: AsyncClient, auth_headers):
    resp = await client.post("/api/v1/ws/token", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert 0 < body["expires_in"] <= 300
    assert "alerts" in body["channels"]

    payload = await ws_module.authenticate_websocket(body["token"])
    assert payload is not None and payload.typ == ws_module.WS_TOKEN_TYPE

    # The short-lived socket token must not work as an HTTP bearer token.
    http = await client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {body['token']}"}
    )
    assert http.status_code == 401


@pytest.mark.asyncio
async def test_regular_access_token_authenticates_socket(client: AsyncClient, admin_user):
    token = create_access_token(
        {"sub": admin_user.id, "username": admin_user.username, "role": admin_user.role}
    )
    assert await ws_module.authenticate_websocket(token) is not None
    assert await ws_module.authenticate_websocket("garbage") is None


def test_customer_token_is_rejected(monkeypatch):
    """Channels carry fleet-wide data, so member-portal accounts can't subscribe."""
    customer = _User("c-1")
    customer.role = "customer"

    async def _fake_load(user_id: str):
        return customer if user_id == "c-1" else None

    monkeypatch.setattr(ws_module, "_load_active_user", _fake_load)
    app = FastAPI()
    app.add_api_websocket_route("/api/v1/ws", ws_module.websocket_endpoint)
    token = create_access_token({"sub": "c-1", "username": "carol", "role": "customer"})
    with (
        TestClient(app) as tc,
        pytest.raises(WebSocketDisconnect),
        tc.websocket_connect(f"/api/v1/ws?token={token}") as ws,
    ):
        ws.receive_text()


# ---------------------------------------------------------------------------
# Credential expiry closes open sockets (code 4001)
# ---------------------------------------------------------------------------


def _short_lived_access_token(seconds: int) -> str:
    settings = get_settings()
    return jwt.encode(
        {
            "sub": "u-1",
            "username": "alice",
            "role": "operator",
            "exp": datetime.now(timezone.utc) + timedelta(seconds=seconds),
        },
        settings.secret_key,
        algorithm=settings.jwt_algorithm,
    )


def _assert_closed_on_expiry(tc: TestClient, url: str, **kwargs) -> None:
    baseline = ws_module.manager.active_count
    start = time.monotonic()
    with tc.websocket_connect(url, **kwargs) as ws:
        assert json.loads(ws.receive_text()) == {"ack": "subscribed:alerts"}
        ws.send_text(json.dumps({"action": "ping"}))
        assert "pong" in json.loads(ws.receive_text())  # usable before expiry
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
    assert exc.value.code == ws_module.WS_TOKEN_EXPIRED
    assert exc.value.reason == "Token expired"
    assert time.monotonic() - start < 10
    assert ws_module.manager.active_count == baseline


def test_socket_closed_when_access_token_expires(ws_app):
    token = _short_lived_access_token(2)
    with TestClient(ws_app) as tc:
        _assert_closed_on_expiry(tc, f"/api/v1/ws?token={token}&channels=alerts")


def test_ws_token_socket_closed_at_session_expiry(ws_app):
    session_exp = datetime.now(timezone.utc) + timedelta(seconds=2)
    token, ttl = ws_module.create_ws_token(_User("u-1"), session_exp)
    assert ttl <= 2  # never outlives the session it was minted from
    with TestClient(ws_app) as tc:
        _assert_closed_on_expiry(tc, "/api/v1/ws?channels=alerts", subprotocols=["bearer", token])


def test_ws_token_socket_outlives_handshake_ttl(ws_app, monkeypatch):
    """The short ws-token ``exp`` only gates the handshake, not the socket."""
    monkeypatch.setattr(get_settings(), "ws_token_expire_seconds", 1)
    session_exp = datetime.now(timezone.utc) + timedelta(hours=1)
    token, ttl = ws_module.create_ws_token(_User("u-1"), session_exp)
    assert ttl == 1
    payload = ws_module.decode_access_token(token)
    assert ws_module.socket_expires_at(payload) == int(session_exp.timestamp())
    with TestClient(ws_app) as tc, tc.websocket_connect(f"/api/v1/ws?token={token}") as ws:
        time.sleep(2.2)  # past the token's own exp
        ws.send_text(json.dumps({"action": "ping"}))
        assert "pong" in json.loads(ws.receive_text())


@pytest.mark.asyncio
async def test_ws_token_carries_session_expiry(client: AsyncClient, auth_headers):
    session = ws_module.decode_access_token(auth_headers["Authorization"].split()[1])
    body = (await client.post("/api/v1/ws/token", headers=auth_headers)).json()
    payload = ws_module.decode_access_token(body["token"])
    assert payload.sexp == session.exp
    assert payload.exp <= session.exp
