"""User and credential management: admin user routes, self-service password /
API keys, session revocation (JWT ``ver`` claim, WebSocket), login hardening,
password policy, first-admin bootstrap and the ``vpp users`` CLI."""

from __future__ import annotations

import json
import logging
import time
import uuid

import pytest
import pytest_asyncio
from click.testing import CliRunner
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from vpp.api import websocket as ws_module
from vpp.auth import security
from vpp.auth.bootstrap import BootstrapError, bootstrap_admin_from_settings
from vpp.auth.passwords import PasswordPolicyError, validate_password
from vpp.auth.security import create_access_token, get_password_hash, issue_access_token
from vpp.auth.throttle import LoginThrottle, login_throttle
from vpp.cli.main import cli
from vpp.db.base import Base
from vpp.db.models import APIKeyModel, UserModel
from vpp.db.repositories import UserRepository
from vpp.settings import Settings, get_settings

STRONG = "Grid-Battery-4217"
STRONG2 = "Turbine-Harbor-9051"


def uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.fixture(autouse=True)
def _clean_throttle():
    login_throttle.reset()
    yield
    login_throttle.reset()


@pytest_asyncio.fixture
async def make_user(db_session: AsyncSession):
    async def _make(role: str = "viewer", password: str = STRONG) -> UserModel:
        user = await UserRepository.create_user(
            db_session,
            username=uid(role),
            hashed_password=get_password_hash(password),
            role=role,
        )
        await db_session.commit()
        return user

    return _make


def bearer(user: UserModel) -> dict[str, str]:
    return {"Authorization": f"Bearer {issue_access_token(user)}"}


async def fresh(db_session: AsyncSession, user_id: str) -> UserModel:
    db_session.expire_all()
    user = await db_session.get(UserModel, user_id)
    assert user is not None
    return user


# ---------------------------------------------------------------------------
# Password policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("password", "reason"),
    [
        ("short-1A", "at least 12"),
        ("aaaaaaaaaaaaaaa", "different characters"),
        ("121212121212", "different characters"),
        ("Password123!", "too common"),
        ("letmein2024!!", "too common"),
        ("CHANGEME-NOW-1", "too common"),
        ("x" * 70 + "éé", "72 bytes"),
        ("my-alice-rocks-99", "username"),
    ],
)
def test_password_policy_rejects(password, reason):
    with pytest.raises(PasswordPolicyError, match=reason):
        validate_password(password, username="alice")


def test_password_policy_accepts_passphrase():
    validate_password(STRONG, username="alice")
    validate_password("correct horse battery staple", username="alice")


def test_password_min_length_is_configurable(monkeypatch):
    monkeypatch.setattr(get_settings(), "password_min_length", 20)
    with pytest.raises(PasswordPolicyError, match="at least 20"):
        validate_password(STRONG)


# ---------------------------------------------------------------------------
# Login hardening
# ---------------------------------------------------------------------------


def test_throttle_locks_after_max_failures_and_expires(monkeypatch):
    monkeypatch.setattr(get_settings(), "login_max_failures", 3)
    monkeypatch.setattr(get_settings(), "login_lockout_seconds", 60)
    now = [1000.0]
    t = LoginThrottle(clock=lambda: now[0])
    for _ in range(2):
        t.record_failure("Bob")
    assert t.retry_after("bob") is None
    t.record_failure("BOB ")
    assert t.retry_after("bob") == 60
    now[0] += 30
    assert t.retry_after("bob") == 30
    now[0] += 31
    assert t.retry_after("bob") is None
    t.record_failure("bob")
    t.record_success("bob")
    t.record_failure("bob")
    t.record_failure("bob")
    assert t.retry_after("bob") is None  # success cleared the earlier failure


def test_throttle_disabled_with_zero(monkeypatch):
    monkeypatch.setattr(get_settings(), "login_max_failures", 0)
    t = LoginThrottle()
    for _ in range(50):
        t.record_failure("bob")
    assert t.retry_after("bob") is None


@pytest.mark.asyncio
async def test_login_throttles_per_username(client: AsyncClient, make_user, monkeypatch):
    monkeypatch.setattr(get_settings(), "login_max_failures", 3)
    user = await make_user()
    for _ in range(3):
        r = await client.post(
            "/api/v1/auth/token", data={"username": user.username, "password": "wrong-password"}
        )
        assert r.status_code == 401
    # Even the right password is refused while locked out.
    locked = await client.post(
        "/api/v1/auth/token", data={"username": user.username, "password": STRONG}
    )
    assert locked.status_code == 429
    assert int(locked.headers["Retry-After"]) > 0
    # Unknown usernames are throttled identically (no enumeration via 429).
    ghost = uid("ghost")
    codes = [
        (
            await client.post("/api/v1/auth/token", data={"username": ghost, "password": "x" * 12})
        ).status_code
        for _ in range(4)
    ]
    assert codes == [401, 401, 401, 429]
    # Other usernames are unaffected.
    other = await make_user()
    ok = await client.post(
        "/api/v1/auth/token", data={"username": other.username, "password": STRONG}
    )
    assert ok.status_code == 200


@pytest.mark.asyncio
async def test_unknown_user_still_runs_bcrypt(client: AsyncClient, make_user, monkeypatch):
    calls: list[str] = []
    real = security.verify_password

    def _spy(plain: str, hashed: str) -> bool:
        calls.append(hashed)
        return real(plain, hashed)

    monkeypatch.setattr(security, "verify_password", _spy)
    r = await client.post(
        "/api/v1/auth/token", data={"username": uid("nobody"), "password": "whatever-123"}
    )
    assert r.status_code == 401
    assert len(calls) == 1 and calls[0].startswith("$2")
    # Inactive users are verified too, and get the same answer.
    user = await make_user()
    user.is_active = False
    calls.clear()
    await _set_active(user.id, False)
    r2 = await client.post(
        "/api/v1/auth/token", data={"username": user.username, "password": STRONG}
    )
    assert r2.status_code == 401 and r2.json() == r.json()
    assert len(calls) == 1


async def _set_active(user_id: str, active: bool) -> None:
    from vpp.db.engine import get_session_factory

    async with get_session_factory()() as s:
        await s.execute(update(UserModel).where(UserModel.id == user_id).values(is_active=active))
        await s.commit()


@pytest.mark.asyncio
async def test_overlong_password_is_401_not_500(client: AsyncClient, make_user):
    user = await make_user()
    r = await client.post(
        "/api/v1/auth/token", data={"username": user.username, "password": "p" * 100}
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_login_records_last_login(client: AsyncClient, make_user, db_session):
    user = await make_user()
    r = await client.post(
        "/api/v1/auth/token", data={"username": user.username, "password": STRONG}
    )
    assert r.status_code == 200
    assert (await fresh(db_session, user.id)).last_login_at is not None


# ---------------------------------------------------------------------------
# Session revocation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_login_token_carries_version(client: AsyncClient, make_user):
    from jose import jwt

    user = await make_user()
    r = await client.post(
        "/api/v1/auth/token", data={"username": user.username, "password": STRONG}
    )
    claims = jwt.get_unverified_claims(r.json()["access_token"])
    assert claims["ver"] == 0


@pytest.mark.asyncio
async def test_legacy_token_valid_only_until_first_revocation(client: AsyncClient, make_user):
    user = await make_user()
    legacy = {
        "Authorization": "Bearer "
        + create_access_token({"sub": user.id, "username": user.username, "role": user.role})
    }
    assert (await client.get("/api/v1/auth/me", headers=legacy)).status_code == 200
    assert (await client.post("/api/v1/auth/logout-all", headers=legacy)).status_code == 204
    r = await client.get("/api/v1/auth/me", headers=legacy)
    assert r.status_code == 401
    assert r.json()["detail"] == "Session has been revoked"


@pytest.mark.asyncio
async def test_logout_all_revokes_every_token_but_not_api_keys(client: AsyncClient, make_user):
    user = await make_user("operator")
    a, b = bearer(user), bearer(user)
    key = (
        await client.post(
            "/api/v1/auth/api-key", json={"name": "k", "role": "operator"}, headers=a
        )
    ).json()["key"]
    assert (await client.post("/api/v1/auth/logout-all", headers=a)).status_code == 204
    assert (await client.get("/api/v1/auth/me", headers=a)).status_code == 401
    assert (await client.get("/api/v1/auth/me", headers=b)).status_code == 401
    assert (await client.get("/api/v1/auth/me", headers={"X-API-Key": key})).status_code == 200
    # A fresh login works again.
    r = await client.post(
        "/api/v1/auth/token", data={"username": user.username, "password": STRONG}
    )
    headers = {"Authorization": f"Bearer {r.json()['access_token']}"}
    assert (await client.get("/api/v1/auth/me", headers=headers)).status_code == 200


@pytest.mark.asyncio
async def test_revoked_session_cannot_mint_or_use_ws_tokens(client: AsyncClient, make_user):
    user = await make_user("operator")
    headers = bearer(user)
    ws_token = (await client.post("/api/v1/ws/token", headers=headers)).json()["token"]
    session_token = headers["Authorization"][7:]
    assert await ws_module.authenticate_websocket(ws_token) is not None
    assert await ws_module.authenticate_websocket(session_token) is not None

    assert (await client.post("/api/v1/auth/logout-all", headers=headers)).status_code == 204
    assert await ws_module.authenticate_websocket(ws_token) is None
    assert await ws_module.authenticate_websocket(session_token) is None
    assert (await client.post("/api/v1/ws/token", headers=headers)).status_code == 401


def test_open_socket_closed_when_session_revoked(monkeypatch):
    class _U:
        id = "u-rev"
        username = "rev"
        role = "operator"
        is_active = True
        token_version = 0

    user = _U()

    async def _load(user_id: str):
        return user if user_id == user.id and user.is_active else None

    monkeypatch.setattr(ws_module, "_load_active_user", _load)
    monkeypatch.setattr(ws_module, "WS_REVALIDATE_SECONDS", 0.2)
    app = FastAPI()
    app.add_api_websocket_route("/api/v1/ws", ws_module.websocket_endpoint)
    token = create_access_token({"sub": user.id, "username": "rev", "role": "operator", "ver": 0})
    with TestClient(app) as tc, tc.websocket_connect(f"/api/v1/ws?token={token}") as ws:
        ws.send_text(json.dumps({"action": "ping"}))
        assert "pong" in json.loads(ws.receive_text())
        time.sleep(0.3)  # survives a revalidation while still valid
        ws.send_text(json.dumps({"action": "ping"}))
        assert "pong" in json.loads(ws.receive_text())
        user.token_version = 1  # e.g. password changed in another request
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
    assert exc.value.code == ws_module.WS_TOKEN_EXPIRED
    assert exc.value.reason == "Session revoked"


# ---------------------------------------------------------------------------
# Self-service password change
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_change_password_flow(client: AsyncClient, make_user):
    user = await make_user("customer")
    old = bearer(user)
    url = "/api/v1/auth/password"

    wrong = await client.post(
        url, json={"current_password": "nope-nope-nope", "new_password": STRONG2}, headers=old
    )
    assert wrong.status_code == 400
    weak = await client.post(
        url, json={"current_password": STRONG, "new_password": "password1234"}, headers=old
    )
    assert weak.status_code == 422 and "common" in weak.json()["detail"]
    same = await client.post(
        url, json={"current_password": STRONG, "new_password": STRONG}, headers=old
    )
    assert same.status_code == 422

    ok = await client.post(
        url, json={"current_password": STRONG, "new_password": STRONG2}, headers=old
    )
    assert ok.status_code == 200
    new = {"Authorization": f"Bearer {ok.json()['access_token']}"}
    assert (await client.get("/api/v1/auth/me", headers=old)).status_code == 401
    assert (await client.get("/api/v1/auth/me", headers=new)).status_code == 200
    login_old = await client.post(
        "/api/v1/auth/token", data={"username": user.username, "password": STRONG}
    )
    assert login_old.status_code == 401
    login_new = await client.post(
        "/api/v1/auth/token", data={"username": user.username, "password": STRONG2}
    )
    assert login_new.status_code == 200


@pytest.mark.asyncio
async def test_change_password_wrong_current_is_throttled(
    client: AsyncClient, make_user, monkeypatch
):
    monkeypatch.setattr(get_settings(), "login_max_failures", 2)
    user = await make_user()
    headers = bearer(user)
    body = {"current_password": "guess-guess-1", "new_password": STRONG2}
    assert (
        await client.post("/api/v1/auth/password", json=body, headers=headers)
    ).status_code == 400
    assert (
        await client.post("/api/v1/auth/password", json=body, headers=headers)
    ).status_code == 400
    assert (
        await client.post("/api/v1/auth/password", json=body, headers=headers)
    ).status_code == 429


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_key_listing_last_used_and_revocation(client: AsyncClient, make_user):
    owner = await make_user("operator")
    other = await make_user("operator")
    admin = await make_user("admin")
    h = bearer(owner)
    created = await client.post(
        "/api/v1/auth/api-keys", json={"name": "ci", "role": "operator"}, headers=h
    )
    assert created.status_code == 201
    key = created.json()
    assert key["key"].startswith(key["key_prefix"]) and len(key["key_prefix"]) == 12

    listed = (await client.get("/api/v1/auth/api-keys", headers=h)).json()
    assert [k["id"] for k in listed] == [key["id"]]
    assert "key" not in listed[0] and "hashed_key" not in listed[0]
    assert listed[0]["last_used_at"] is None and listed[0]["username"] == owner.username

    assert (
        await client.get("/api/v1/auth/me", headers={"X-API-Key": key["key"]})
    ).status_code == 200
    listed = (await client.get("/api/v1/auth/api-keys", headers=h)).json()
    assert listed[0]["last_used_at"] is not None

    # Others cannot see or revoke it; they get 404, not 403.
    assert (await client.get("/api/v1/auth/api-keys", headers=bearer(other))).json() == []
    assert (
        await client.delete(f"/api/v1/auth/api-keys/{key['id']}", headers=bearer(other))
    ).status_code == 404
    assert (
        await client.get("/api/v1/auth/api-keys?all=true", headers=bearer(other))
    ).status_code == 403

    # Admin sees every key.
    all_keys = (await client.get("/api/v1/auth/api-keys?all=true", headers=bearer(admin))).json()
    assert key["id"] in {k["id"] for k in all_keys}

    # Owner revokes: key stops working, stays listed with include_revoked.
    assert (
        await client.delete(f"/api/v1/auth/api-keys/{key['id']}", headers=h)
    ).status_code == 204
    assert (
        await client.delete(f"/api/v1/auth/api-keys/{key['id']}", headers=h)
    ).status_code == 204
    assert (
        await client.get("/api/v1/auth/me", headers={"X-API-Key": key["key"]})
    ).status_code == 401
    assert (await client.get("/api/v1/auth/api-keys", headers=h)).json() == []
    revoked = (await client.get("/api/v1/auth/api-keys?include_revoked=true", headers=h)).json()
    assert revoked[0]["is_active"] is False


@pytest.mark.asyncio
async def test_admin_can_revoke_any_key(client: AsyncClient, make_user):
    owner = await make_user("operator")
    admin = await make_user("admin")
    key = (
        await client.post(
            "/api/v1/auth/api-key", json={"name": "x", "role": "operator"}, headers=bearer(owner)
        )
    ).json()
    r = await client.delete(f"/api/v1/auth/api-keys/{key['id']}", headers=bearer(admin))
    assert r.status_code == 204
    assert (
        await client.get("/api/v1/auth/me", headers={"X-API-Key": key["key"]})
    ).status_code == 401
    per_user = (
        await client.get(
            f"/api/v1/users/{owner.id}/api-keys?include_revoked=true", headers=bearer(admin)
        )
    ).json()
    assert [k["is_active"] for k in per_user] == [False]


# ---------------------------------------------------------------------------
# Admin user management
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_routes_require_admin(client: AsyncClient, make_user):
    viewer = await make_user("viewer")
    h = bearer(viewer)
    assert (await client.get("/api/v1/users", headers=h)).status_code == 403
    assert (await client.get(f"/api/v1/users/{viewer.id}", headers=h)).status_code == 403
    assert (
        await client.patch(f"/api/v1/users/{viewer.id}", json={"role": "admin"}, headers=h)
    ).status_code == 403
    assert (await client.get("/api/v1/users")).status_code == 401


@pytest.mark.asyncio
async def test_admin_lists_creates_and_gets_users(client: AsyncClient, make_user):
    admin = await make_user("admin")
    h = bearer(admin)
    name = uid("new")
    weak = await client.post(
        "/api/v1/users",
        json={"username": name, "password": "password1", "role": "operator"},
        headers=h,
    )
    assert weak.status_code == 422
    created = await client.post(
        "/api/v1/users", json={"username": name, "password": STRONG, "role": "operator"}, headers=h
    )
    assert created.status_code == 201, created.text
    new_id = created.json()["id"]
    dup = await client.post(
        "/api/v1/users", json={"username": name, "password": STRONG, "role": "operator"}, headers=h
    )
    assert dup.status_code == 409

    got = (await client.get(f"/api/v1/users/{new_id}", headers=h)).json()
    assert got["username"] == name and got["role"] == "operator" and got["api_key_count"] == 0
    assert "hashed_password" not in got and "token_version" not in got
    listed = (await client.get("/api/v1/users?role=operator", headers=h)).json()
    assert name in {u["username"] for u in listed}
    assert all(u["role"] == "operator" for u in listed)
    assert (await client.get(f"/api/v1/users/{uuid.uuid4()}", headers=h)).status_code == 404


@pytest.mark.asyncio
async def test_register_enforces_password_policy(client: AsyncClient, auth_headers):
    r = await client.post(
        "/api/v1/auth/register",
        json={"username": uid("reg"), "password": "qwerty123456", "role": "viewer"},
        headers=auth_headers,
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_role_change_revokes_sessions(client: AsyncClient, make_user):
    admin = await make_user("admin")
    target = await make_user("operator")
    th = bearer(target)
    r = await client.patch(
        f"/api/v1/users/{target.id}", json={"role": "viewer"}, headers=bearer(admin)
    )
    assert r.status_code == 200 and r.json()["role"] == "viewer"
    assert (await client.get("/api/v1/auth/me", headers=th)).status_code == 401
    # No-op patch does not revoke.
    th2 = await _login(client, target.username)
    r = await client.patch(
        f"/api/v1/users/{target.id}", json={"role": "viewer"}, headers=bearer(admin)
    )
    assert (await client.get("/api/v1/auth/me", headers=th2)).status_code == 200
    bad = await client.patch(
        f"/api/v1/users/{target.id}", json={"password": "x"}, headers=bearer(admin)
    )
    assert bad.status_code == 422


async def _login(client: AsyncClient, username: str, password: str = STRONG) -> dict[str, str]:
    r = await client.post("/api/v1/auth/token", data={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.mark.asyncio
async def test_deactivation_revokes_sessions_and_api_keys(client: AsyncClient, make_user):
    admin = await make_user("admin")
    target = await make_user("operator")
    th = bearer(target)
    key = (
        await client.post(
            "/api/v1/auth/api-key", json={"name": "k", "role": "operator"}, headers=th
        )
    ).json()
    r = await client.patch(
        f"/api/v1/users/{target.id}", json={"is_active": False}, headers=bearer(admin)
    )
    assert r.status_code == 200
    assert r.json()["is_active"] is False and r.json()["api_key_count"] == 0
    assert (await client.get("/api/v1/auth/me", headers=th)).status_code == 401
    assert (
        await client.get("/api/v1/auth/me", headers={"X-API-Key": key["key"]})
    ).status_code == 401
    login = await client.post(
        "/api/v1/auth/token", data={"username": target.username, "password": STRONG}
    )
    assert login.status_code == 401

    # Re-activation restores login, but not the old API keys or tokens.
    r = await client.patch(
        f"/api/v1/users/{target.id}", json={"is_active": True}, headers=bearer(admin)
    )
    assert r.status_code == 200 and r.json()["is_active"] is True
    await _login(client, target.username)
    assert (
        await client.get("/api/v1/auth/me", headers={"X-API-Key": key["key"]})
    ).status_code == 401
    assert (await client.get("/api/v1/auth/me", headers=th)).status_code == 401


@pytest.mark.asyncio
async def test_admin_cannot_lock_themselves_out(client: AsyncClient, make_user):
    admin = await make_user("admin")
    h = bearer(admin)
    for body in ({"is_active": False}, {"role": "operator"}):
        r = await client.patch(f"/api/v1/users/{admin.id}", json=body, headers=h)
        assert r.status_code == 409, body
    # Still an active admin.
    me = (await client.get(f"/api/v1/users/{admin.id}", headers=h)).json()
    assert me["role"] == "admin" and me["is_active"] is True
    # Re-asserting the current state is fine.
    assert (
        await client.patch(f"/api/v1/users/{admin.id}", json={"role": "admin"}, headers=h)
    ).status_code == 200


@pytest.mark.asyncio
async def test_admin_can_demote_another_admin(client: AsyncClient, make_user):
    a = await make_user("admin")
    b = await make_user("admin")
    r = await client.patch(f"/api/v1/users/{b.id}", json={"role": "operator"}, headers=bearer(a))
    assert r.status_code == 200 and r.json()["role"] == "operator"


@pytest.mark.asyncio
async def test_last_active_admin_is_protected(db_session: AsyncSession, make_user):
    """Guards the race where two admins demote each other concurrently: the
    second request (whose caller is no longer an active admin) must fail."""
    from fastapi import HTTPException

    from vpp.api.routes.users import update_user
    from vpp.schemas.auth import UserUpdate

    last = await make_user("admin")
    last_id = last.id
    others = list(
        (
            await db_session.execute(
                select(UserModel.id).where(
                    UserModel.role == "admin",
                    UserModel.is_active.is_(True),
                    UserModel.id != last_id,
                )
            )
        ).scalars()
    )
    await db_session.execute(
        update(UserModel).where(UserModel.id.in_(others)).values(is_active=False)
    )
    await db_session.commit()
    try:
        caller = UserModel(id=str(uuid.uuid4()), username="demoted-meanwhile", role="admin")
        for body in (UserUpdate(role="viewer"), UserUpdate(is_active=False)):
            with pytest.raises(HTTPException) as exc:
                await update_user(last_id, body, session=db_session, admin=caller)
            assert exc.value.status_code == 409
            assert "last active admin" in exc.value.detail
        await db_session.rollback()
        assert (await fresh(db_session, last_id)).role == "admin"
    finally:
        await db_session.execute(
            update(UserModel).where(UserModel.id.in_(others)).values(is_active=True)
        )
        await db_session.commit()


@pytest.mark.asyncio
async def test_admin_password_reset(client: AsyncClient, make_user):
    admin = await make_user("admin")
    target = await make_user("viewer")
    th = bearer(target)
    url = f"/api/v1/users/{target.id}/password"
    ah = bearer(admin)
    assert (await client.post(url, json={"new_password": "short"}, headers=ah)).status_code == 422
    assert (await client.post(url, json={"new_password": STRONG2}, headers=th)).status_code == 403
    assert (await client.post(url, json={"new_password": STRONG2}, headers=ah)).status_code == 204
    assert (await client.get("/api/v1/auth/me", headers=th)).status_code == 401
    await _login(client, target.username, STRONG2)
    # Own password goes through the self-service route.
    own = await client.post(
        f"/api/v1/users/{admin.id}/password", json={"new_password": STRONG2}, headers=ah
    )
    assert own.status_code == 409


@pytest.mark.asyncio
async def test_admin_revoke_sessions(client: AsyncClient, make_user):
    admin = await make_user("admin")
    target = await make_user("viewer")
    th = bearer(target)
    r = await client.post(f"/api/v1/users/{target.id}/revoke-sessions", headers=bearer(admin))
    assert r.status_code == 204
    assert (await client.get("/api/v1/auth/me", headers=th)).status_code == 401


# ---------------------------------------------------------------------------
# First-boot bootstrap
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def empty_db(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'boot.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


def _boot_settings(tmp_path, password: str | None = STRONG, **extra) -> Settings:
    path = None
    if password is not None:
        path = tmp_path / "admin_password"
        path.write_text(password + "\n")
    return Settings(
        bootstrap_admin_username="root-admin",
        bootstrap_admin_password_file=str(path) if path else None,
        **extra,
    )


@pytest.mark.asyncio
async def test_bootstrap_creates_admin_once(empty_db, tmp_path, caplog):
    settings = _boot_settings(tmp_path)
    with caplog.at_level(logging.DEBUG):
        assert await bootstrap_admin_from_settings(empty_db, settings) is True
        assert await bootstrap_admin_from_settings(empty_db, settings) is False
    assert STRONG not in caplog.text
    async with empty_db() as s:
        users = list((await s.execute(select(UserModel))).scalars())
    assert [(u.username, u.role, u.is_active) for u in users] == [("root-admin", "admin", True)]
    assert security.verify_password(STRONG, users[0].hashed_password)  # newline stripped


@pytest.mark.asyncio
async def test_bootstrap_ignored_when_users_exist_even_if_file_missing(empty_db, tmp_path):
    async with empty_db() as s:
        await UserRepository.create_user(s, username="someone", hashed_password="x")
        await s.commit()
    settings = Settings(
        bootstrap_admin_username="root-admin",
        bootstrap_admin_password_file=str(tmp_path / "does-not-exist"),
    )
    assert await bootstrap_admin_from_settings(empty_db, settings) is False


@pytest.mark.asyncio
async def test_bootstrap_misconfiguration_fails_loudly(empty_db, tmp_path):
    assert await bootstrap_admin_from_settings(empty_db, Settings()) is False  # not configured
    with pytest.raises(BootstrapError, match="PASSWORD_FILE"):
        await bootstrap_admin_from_settings(empty_db, _boot_settings(tmp_path, password=None))
    with pytest.raises(BootstrapError, match="too common") as exc:
        await bootstrap_admin_from_settings(empty_db, _boot_settings(tmp_path, "changeme123!"))
    assert "changeme123!" not in str(exc.value)
    missing = Settings(
        bootstrap_admin_username="root-admin",
        bootstrap_admin_password_file=str(tmp_path / "nope"),
    )
    with pytest.raises(BootstrapError, match="Cannot read"):
        await bootstrap_admin_from_settings(empty_db, missing)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_db(tmp_path, monkeypatch):
    db = tmp_path / "cli_users.db"
    monkeypatch.setenv("VPP_DATABASE_URL", f"sqlite+aiosqlite:///{db}")
    monkeypatch.delenv("VPP_ADMIN_PASSWORD", raising=False)
    get_settings.cache_clear()
    yield db
    get_settings.cache_clear()


def _cli_users(db) -> list[tuple[str, str, bool, int]]:
    from sqlalchemy import create_engine

    engine = create_engine(f"sqlite:///{db}")
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                select(
                    UserModel.username,
                    UserModel.role,
                    UserModel.is_active,
                    UserModel.token_version,
                )
            ).all()
        return [tuple(r) for r in rows]
    finally:
        engine.dispose()


def test_cli_create_admin_from_stdin(cli_db):
    runner = CliRunner()
    weak = runner.invoke(
        cli, ["users", "create-admin", "ops", "--password-stdin"], input="admin123\n"
    )
    assert weak.exit_code != 0 and "at least 12" in weak.output
    assert "admin123" not in weak.output

    ok = runner.invoke(
        cli, ["users", "create-admin", "ops", "--password-stdin"], input=STRONG + "\n"
    )
    assert ok.exit_code == 0, ok.output
    assert STRONG not in ok.output
    assert _cli_users(cli_db) == [("ops", "admin", True, 0)]

    again = runner.invoke(
        cli, ["users", "create-admin", "ops", "--password-stdin"], input=STRONG + "\n"
    )
    assert again.exit_code != 0 and "already exists" in again.output

    listed = runner.invoke(cli, ["users", "list"])
    assert listed.exit_code == 0 and "ops" in listed.output and "admin" in listed.output


def test_cli_create_admin_from_env_and_file(cli_db, tmp_path, monkeypatch):
    runner = CliRunner()
    monkeypatch.setenv("VPP_ADMIN_PASSWORD", STRONG)
    assert runner.invoke(cli, ["users", "create-admin", "env-admin"]).exit_code == 0
    monkeypatch.delenv("VPP_ADMIN_PASSWORD")
    pw = tmp_path / "pw"
    pw.write_text(STRONG2 + "\n")
    r = runner.invoke(cli, ["users", "create-admin", "file-admin", "--password-file", str(pw)])
    assert r.exit_code == 0, r.output
    # No password source and no TTY: refuse instead of hanging.
    r = runner.invoke(cli, ["users", "create-admin", "nopw"])
    assert r.exit_code != 0 and "No password given" in r.output
    bad = runner.invoke(cli, ["users", "create-admin", "a b", "--password-stdin"], input=STRONG)
    assert bad.exit_code != 0 and "Username" in bad.output
    assert {u[0] for u in _cli_users(cli_db)} == {"env-admin", "file-admin"}


def test_cli_set_password_revokes_sessions_and_reactivates(cli_db):
    runner = CliRunner()
    runner.invoke(cli, ["users", "create-admin", "ops", "--password-stdin"], input=STRONG + "\n")
    from sqlalchemy import create_engine

    engine = create_engine(f"sqlite:///{cli_db}")
    with engine.begin() as conn:
        conn.execute(update(UserModel).values(is_active=False))
    engine.dispose()

    r = runner.invoke(
        cli, ["users", "set-password", "ops", "--password-stdin"], input=STRONG2 + "\n"
    )
    assert r.exit_code == 0, r.output
    assert _cli_users(cli_db) == [("ops", "admin", True, 1)]
    missing = runner.invoke(
        cli, ["users", "set-password", "ghost", "--password-stdin"], input=STRONG2
    )
    assert missing.exit_code != 0 and "No user named" in missing.output


def test_api_key_model_has_last_used_and_prefix_columns():
    cols = APIKeyModel.__table__.columns
    assert "last_used_at" in cols and "key_prefix" in cols
    assert "token_version" in UserModel.__table__.columns
