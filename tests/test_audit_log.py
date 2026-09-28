"""Audit log (``/api/v1/audit``), user deletion and list pagination."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from vpp import audit
from vpp.auth.security import generate_api_key, get_password_hash, hash_api_key
from vpp.auth.throttle import login_throttle
from vpp.db.models import (
    APIKeyModel,
    AuditLogModel,
    OrderModel,
    SiteModel,
    TradeModel,
    UserModel,
)
from vpp.db.repositories import UserRepository
from vpp.trading.service import TradingServiceConfig, reset_trading_service

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
    from vpp.auth.security import issue_access_token

    return {"Authorization": f"Bearer {issue_access_token(user)}"}


@pytest_asyncio.fixture
async def admin_h(make_user) -> dict[str, str]:
    return bearer(await make_user("admin"))


async def entries(client: AsyncClient, headers: dict[str, str], **params: Any) -> list[dict]:
    r = await client.get("/api/v1/audit", params=params, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


async def login(client: AsyncClient, username: str, password: str) -> str:
    r = await client.post("/api/v1/auth/token", data={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


# ---------------------------------------------------------------------------
# Read endpoint: RBAC, filters, pagination
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["operator", "viewer", "researcher", "customer"])
async def test_audit_is_admin_only(client: AsyncClient, make_user, role):
    user = await make_user(role)
    assert (await client.get("/api/v1/audit", headers=bearer(user))).status_code == 403


@pytest.mark.asyncio
async def test_audit_requires_auth(client: AsyncClient):
    assert (await client.get("/api/v1/audit")).status_code == 401


@pytest.mark.asyncio
async def test_audit_filters_and_pagination(client: AsyncClient, make_user, admin_h):
    user = await make_user("viewer")
    for _ in range(3):
        await client.post(
            "/api/v1/auth/token", data={"username": user.username, "password": "wrong-password"}
        )
    await login(client, user.username, STRONG)

    all_rows = await entries(client, admin_h, actor=user.username, action="auth.login")
    assert [r["outcome"] for r in all_rows] == ["success", "failure", "failure", "failure"]

    r = await client.get(
        "/api/v1/audit",
        params={"actor": user.id, "action": "auth.", "limit": 2, "offset": 1},
        headers=admin_h,
    )
    assert r.status_code == 200
    assert r.headers["X-Total-Count"] == "4"
    assert [x["id"] for x in r.json()] == [x["id"] for x in all_rows[1:3]]

    failures = await entries(client, admin_h, actor=user.username, outcome="failure")
    assert len(failures) == 3
    newest = all_rows[0]["ts"]
    assert len(await entries(client, admin_h, actor=user.username, since=newest)) == 1
    assert len(await entries(client, admin_h, actor=user.username, until=newest)) == 4
    assert await entries(client, admin_h, actor=user.username, action="user.delete") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 501}, {"offset": -1}])
async def test_audit_pagination_bounds(client: AsyncClient, admin_h, params):
    r = await client.get("/api/v1/audit", params=params, headers=admin_h)
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# What gets recorded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_login_success_and_failure_are_audited_without_secrets(
    client: AsyncClient, make_user, admin_h
):
    user = await make_user("operator")
    bad = await client.post(
        "/api/v1/auth/token",
        data={"username": user.username, "password": "Secret-Wrong-Pass-1"},
        headers={"X-Forwarded-For": "203.0.113.9"},
    )
    assert bad.status_code == 401
    token = await login(client, user.username, STRONG)
    ghost = uid("ghost")
    r = await client.post("/api/v1/auth/token", data={"username": ghost, "password": "x-y-z-1"})
    assert r.status_code == 401

    rows = await entries(client, admin_h, actor=user.username, action="auth.login")
    assert [(r["outcome"], r["details"].get("reason")) for r in rows] == [
        ("success", None),
        ("failure", "bad_password"),
    ]
    assert all(r["actor_id"] == user.id for r in rows)
    # X-Forwarded-For is ignored unless the peer is a trusted proxy.
    assert rows[1]["client_ip"] == "127.0.0.1"
    unknown = await entries(client, admin_h, actor=ghost)
    assert len(unknown) == 1
    assert unknown[0]["actor_id"] is None and unknown[0]["details"]["reason"] == "unknown_user"

    blob = json.dumps(rows + unknown)
    assert "Secret-Wrong-Pass-1" not in blob and STRONG not in blob and token not in blob


@pytest.mark.asyncio
async def test_inactive_and_throttled_logins_are_audited(
    client: AsyncClient, make_user, admin_h, monkeypatch
):
    from vpp.settings import get_settings

    monkeypatch.setattr(get_settings(), "login_max_failures", 1)
    user = await make_user("viewer")
    await client.post("/api/v1/auth/token", data={"username": user.username, "password": "nope"})
    r = await client.post(
        "/api/v1/auth/token", data={"username": user.username, "password": STRONG}
    )
    assert r.status_code == 429
    rows = await entries(client, admin_h, actor=user.username)
    assert [(r["outcome"], r["details"]["reason"]) for r in rows] == [
        ("denied", "throttled"),
        ("failure", "bad_password"),
    ]


@pytest.mark.asyncio
async def test_logout_password_and_session_actions_are_audited(
    client: AsyncClient, make_user, admin_h
):
    user = await make_user("viewer")
    h = bearer(user)
    assert (await client.post("/api/v1/auth/logout", headers=h)).status_code == 204
    wrong = await client.post(
        "/api/v1/auth/password",
        json={"current_password": "not-it", "new_password": STRONG2},
        headers=h,
    )
    assert wrong.status_code == 400
    ok = await client.post(
        "/api/v1/auth/password",
        json={"current_password": STRONG, "new_password": STRONG2},
        headers=h,
    )
    assert ok.status_code == 200
    h = {"Authorization": f"Bearer {ok.json()['access_token']}"}
    assert (await client.post("/api/v1/auth/logout-all", headers=h)).status_code == 204

    rows = await entries(client, admin_h, actor=user.id)
    assert [(r["action"], r["outcome"]) for r in rows] == [
        ("auth.logout_all", "success"),
        ("auth.password_change", "success"),
        ("auth.password_change", "failure"),
        ("auth.logout", "success"),
    ]
    assert STRONG2 not in json.dumps(rows) and "not-it" not in json.dumps(rows)
    assert (await client.post("/api/v1/auth/logout")).status_code == 401


@pytest.mark.asyncio
async def test_api_key_create_and_revoke_are_audited(client: AsyncClient, make_user, admin_h):
    user = await make_user("operator")
    h = bearer(user)
    created = await client.post(
        "/api/v1/auth/api-keys", json={"name": "ci", "role": "operator"}, headers=h
    )
    assert created.status_code == 201
    raw_key, key_id = created.json()["key"], created.json()["id"]
    denied = await client.post(
        "/api/v1/auth/api-keys", json={"name": "esc", "role": "admin"}, headers=h
    )
    assert denied.status_code == 403
    assert (await client.delete(f"/api/v1/auth/api-keys/{key_id}", headers=h)).status_code == 204

    rows = await entries(client, admin_h, actor=user.id, action="api_key.")
    assert [(r["action"], r["outcome"]) for r in rows] == [
        ("api_key.revoke", "success"),
        ("api_key.create", "denied"),
        ("api_key.create", "success"),
    ]
    assert rows[2]["target_id"] == key_id and rows[2]["details"]["prefix"] == raw_key[:12]
    assert raw_key not in json.dumps(rows)
    assert hash_api_key(raw_key) not in json.dumps(rows)


@pytest.mark.asyncio
async def test_user_administration_is_audited(client: AsyncClient, make_user):
    admin = await make_user("admin")
    h = bearer(admin)
    name = uid("audited")
    created = await client.post(
        "/api/v1/users", json={"username": name, "password": STRONG, "role": "viewer"}, headers=h
    )
    assert created.status_code == 201
    target = created.json()["id"]
    base = f"/api/v1/users/{target}"
    assert (await client.patch(base, json={"role": "operator"}, headers=h)).status_code == 200
    assert (await client.patch(base, json={"is_active": False}, headers=h)).status_code == 200
    assert (await client.patch(base, json={"is_active": True}, headers=h)).status_code == 200
    r = await client.post(f"{base}/password", json={"new_password": STRONG2}, headers=h)
    assert r.status_code == 204
    assert (await client.post(f"{base}/revoke-sessions", headers=h)).status_code == 204
    assert (await client.delete(base, headers=h)).status_code == 204
    self_delete = await client.delete(f"/api/v1/users/{admin.id}", headers=h)
    assert self_delete.status_code == 409

    rows = await entries(client, h, actor=admin.id)
    assert [(r["action"], r["outcome"], r["target_id"]) for r in rows] == [
        ("user.delete", "denied", admin.id),
        ("user.delete", "success", target),
        ("user.sessions_revoke", "success", target),
        ("user.password_reset", "success", target),
        ("user.activate", "success", target),
        ("user.deactivate", "success", target),
        ("user.role_change", "success", target),
        ("user.create", "success", target),
    ]
    assert rows[6]["details"]["role"] == {"from": "viewer", "to": "operator"}
    assert rows[1]["details"]["username"] == name
    assert STRONG not in json.dumps(rows) and STRONG2 not in json.dumps(rows)


@pytest.mark.asyncio
async def test_register_and_customer_onboarding_are_audited(client: AsyncClient, make_user):
    admin = await make_user("admin")
    h = bearer(admin)
    name = uid("reg")
    r = await client.post(
        "/api/v1/auth/register",
        json={"username": name, "password": STRONG, "role": "viewer"},
        headers=h,
    )
    assert r.status_code == 201
    rows = await entries(client, h, actor=admin.id, action="user.create")
    assert rows[0]["details"] == {"username": name, "role": "viewer"}


@pytest.mark.asyncio
async def test_control_actions_are_audited(client: AsyncClient, make_user, db_session):
    await db_session.execute(delete(TradeModel))
    await db_session.execute(delete(OrderModel))
    await db_session.commit()
    reset_trading_service(TradingServiceConfig(seed=7, base_volume=20.0))
    admin_h = bearer(await make_user("admin"))
    operator = await make_user("operator")
    viewer = await make_user("viewer")
    oh = bearer(operator)
    try:
        order = await client.post(
            "/api/v1/trading/orders",
            json={
                "order_type": "limit",
                "market": "day_ahead",
                "side": "buy",
                "quantity": 10.0,
                "price": 1.0,
            },
            headers=oh,
        )
        assert order.status_code == 201, order.text
        order_id = order.json()["id"]
        assert (
            await client.delete(f"/api/v1/trading/orders/{order_id}", headers=oh)
        ).status_code == 200
        again = await client.delete(f"/api/v1/trading/orders/{order_id}", headers=oh)
        assert again.status_code == 409
    finally:
        await db_session.execute(delete(TradeModel))
        await db_session.execute(delete(OrderModel))
        await db_session.commit()
        reset_trading_service(TradingServiceConfig(seed=7, base_volume=20.0))

    rows = await entries(client, admin_h, actor=operator.id, action="market.")
    assert [(r["action"], r["outcome"]) for r in rows] == [
        ("market.order_cancel", "failure"),
        ("market.order_cancel", "success"),
        ("market.order_submit", "success"),
    ]
    assert rows[2]["target_id"] == order_id
    assert rows[2]["details"]["market"] == "day_ahead"

    # Applying setpoints to devices needs admin/operator: the refusal is audited.
    denied = await client.post(
        "/api/v1/optimization/dispatch",
        json={"target_power_kw": 5.0, "apply": True},
        headers=bearer(viewer),
    )
    assert denied.status_code == 403
    rows = await entries(client, admin_h, actor=viewer.id)
    assert [(r["action"], r["outcome"]) for r in rows] == [("control.dispatch", "denied")]


# ---------------------------------------------------------------------------
# Robustness and sanitisation
# ---------------------------------------------------------------------------


def test_sanitize_details_drops_secrets_and_caps_size():
    raw = audit.sanitize_details(
        {
            "password": "p",
            "new_password": "p2",
            "nested": {"access_token": "t", "Authorization": "Bearer x", "ok": 1},
            "key": "vpp_raw",
            "hashed_key": "h",
            "client_secret": "s",
            "name": "ci",
        }
    )
    assert raw is not None
    data = json.loads(raw)
    assert data == {"nested": {"ok": 1}, "name": "ci"}
    big = audit.sanitize_details({"blob": ["x" * 200] * 40})
    assert big is not None and len(big) <= 2000 and json.loads(big)["truncated"] is True
    assert audit.sanitize_details(None) is None and audit.sanitize_details({}) is None


@pytest.mark.asyncio
async def test_failed_audit_insert_does_not_break_the_request(
    client: AsyncClient, make_user, monkeypatch, caplog
):
    user = await make_user("viewer")
    real_build = audit.build_entry

    def broken(*args, **kwargs):
        row = real_build(*args, **kwargs)
        row.action = None  # NOT NULL violation at insert time
        return row

    monkeypatch.setattr(audit, "build_entry", broken)
    caplog.set_level(logging.WARNING, logger="vpp.db.engine")
    r = await client.post(
        "/api/v1/auth/password",
        json={"current_password": STRONG, "new_password": STRONG2},
        headers=bearer(user),
    )
    assert r.status_code == 200
    assert any("audit" in rec.getMessage() for rec in caplog.records)
    # The request's own change was committed despite the audit failure.
    token = await login(client, user.username, STRONG2)
    assert (
        await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    ).status_code == 200


@pytest.mark.asyncio
async def test_forbidden_delete_writes_no_user_delete_entry(
    client: AsyncClient, make_user, db_session: AsyncSession
):
    """RBAC refusals (403 from the role dependency) happen before any audit call."""
    user = await make_user("viewer")
    # Non-admin: 403 before anything is recorded.
    r = await client.delete(f"/api/v1/users/{user.id}", headers=bearer(user))
    assert r.status_code == 403
    rows = (
        await db_session.execute(select(AuditLogModel).where(AuditLogModel.target_id == user.id))
    ).all()
    assert rows == []


# ---------------------------------------------------------------------------
# User deletion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_user_removes_credentials_and_keeps_history(
    client: AsyncClient, make_user, db_session: AsyncSession
):
    admin = await make_user("admin")
    h = bearer(admin)
    victim = await make_user("customer")
    victim_id, victim_name = victim.id, victim.username
    raw = generate_api_key()
    db_session.add(
        APIKeyModel(user_id=victim.id, name="k", hashed_key=hash_api_key(raw), role="customer")
    )
    site = SiteModel(name=uid("site"), lat=37.0, lon=-122.0, owner_id=victim.id)
    db_session.add(site)
    await db_session.commit()
    site_id = site.id
    victim_h = bearer(victim)
    assert (await client.get("/api/v1/auth/me", headers=victim_h)).status_code == 200
    assert (await client.get("/api/v1/auth/me", headers={"X-API-Key": raw})).status_code == 200

    assert (await client.delete(f"/api/v1/users/{victim_id}", headers=h)).status_code == 204
    assert (await client.get(f"/api/v1/users/{victim_id}", headers=h)).status_code == 404
    assert (await client.delete(f"/api/v1/users/{victim_id}", headers=h)).status_code == 404
    assert (await client.get("/api/v1/auth/me", headers=victim_h)).status_code == 401
    assert (await client.get("/api/v1/auth/me", headers={"X-API-Key": raw})).status_code == 401

    db_session.expire_all()
    keys = (
        await db_session.execute(select(APIKeyModel).where(APIKeyModel.user_id == victim_id))
    ).all()
    assert keys == []
    owner = (
        await db_session.execute(select(SiteModel.owner_id).where(SiteModel.id == site_id))
    ).one()
    assert owner == (None,)
    rows = await entries(client, h, target_id=victim_id, action="user.delete")
    assert rows[0]["details"]["username"] == victim_name
    assert rows[0]["details"]["api_keys_deleted"] == 1
    await db_session.execute(delete(SiteModel).where(SiteModel.id == site_id))
    await db_session.commit()


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["operator", "viewer", "customer"])
async def test_non_admin_cannot_delete_users(client: AsyncClient, make_user, role):
    caller = await make_user(role)
    target = await make_user("viewer")
    r = await client.delete(f"/api/v1/users/{target.id}", headers=bearer(caller))
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_admin_cannot_delete_self(client: AsyncClient, make_user):
    admin = await make_user("admin")
    r = await client.delete(f"/api/v1/users/{admin.id}", headers=bearer(admin))
    assert r.status_code == 409 and "own account" in r.json()["detail"]


@pytest.mark.asyncio
async def test_admin_can_delete_another_admin(client: AsyncClient, make_user):
    a = await make_user("admin")
    b = await make_user("admin")
    assert (await client.delete(f"/api/v1/users/{b.id}", headers=bearer(a))).status_code == 204


@pytest.mark.asyncio
async def test_last_active_admin_cannot_be_deleted(db_session: AsyncSession, make_user):
    from fastapi import HTTPException

    from vpp.api.routes.users import delete_user

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
        # The caller was demoted concurrently: only ``last`` is still an active admin.
        caller = UserModel(id=str(uuid.uuid4()), username="demoted-meanwhile", role="admin")
        with pytest.raises(HTTPException) as exc:
            await delete_user(last_id, request=None, session=db_session, admin=caller)
        assert exc.value.status_code == 409
        assert "last active admin" in exc.value.detail
        await db_session.rollback()
        assert await db_session.get(UserModel, last_id) is not None
    finally:
        await db_session.execute(
            update(UserModel).where(UserModel.id.in_(others)).values(is_active=True)
        )
        await db_session.commit()


# ---------------------------------------------------------------------------
# Pagination of list endpoints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_users_list_is_paginated(client: AsyncClient, make_user, admin_h):
    for _ in range(3):
        await make_user("researcher")
    full = await client.get("/api/v1/users", params={"role": "researcher"}, headers=admin_h)
    total = int(full.headers["X-Total-Count"])
    assert total == len(full.json()) >= 3
    page = await client.get(
        "/api/v1/users", params={"role": "researcher", "limit": 2, "offset": 1}, headers=admin_h
    )
    assert page.headers["X-Total-Count"] == str(total)
    assert [u["id"] for u in page.json()] == [u["id"] for u in full.json()[1:3]]
    for bad in ({"limit": 0}, {"limit": 501}, {"offset": -1}):
        assert (await client.get("/api/v1/users", params=bad, headers=admin_h)).status_code == 422


@pytest.mark.asyncio
async def test_api_key_lists_are_paginated(client: AsyncClient, make_user, admin_h):
    user = await make_user("operator")
    h = bearer(user)
    for i in range(3):
        r = await client.post(
            "/api/v1/auth/api-keys", json={"name": f"k{i}", "role": "operator"}, headers=h
        )
        assert r.status_code == 201
    own = await client.get("/api/v1/auth/api-keys", params={"limit": 2}, headers=h)
    assert own.headers["X-Total-Count"] == "3" and len(own.json()) == 2
    admin_view = await client.get(
        f"/api/v1/users/{user.id}/api-keys", params={"offset": 2}, headers=admin_h
    )
    assert admin_view.headers["X-Total-Count"] == "3" and len(admin_view.json()) == 1
    too_big = await client.get("/api/v1/auth/api-keys", params={"limit": 10_000}, headers=h)
    assert too_big.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "max_limit"),
    [
        ("/api/v1/resources", 200),
        ("/api/v1/alerts", 1000),
        ("/api/v1/trading/orders", 200),
        ("/api/v1/trading/trades", 200),
        ("/api/v1/optimization/runs", 200),
        ("/api/v1/dispatches", 200),
        ("/api/v1/tariffs", 200),
        ("/api/v1/customers", 500),
        ("/api/v1/dr/responses", 1000),
        ("/api/v1/optimization/setpoints", 500),
    ],
)
async def test_list_endpoints_report_totals_and_cap_limit(
    client: AsyncClient, admin_h, path, max_limit
):
    r = await client.get(path, params={"limit": 1, "offset": 0}, headers=admin_h)
    assert r.status_code == 200, r.text
    total = int(r.headers["X-Total-Count"])
    body = r.json()
    items = body["recent"] if isinstance(body, dict) else body
    assert len(items) == min(1, total)
    over = await client.get(path, params={"limit": max_limit + 1}, headers=admin_h)
    assert over.status_code == 422
    neg = await client.get(path, params={"offset": -1}, headers=admin_h)
    assert neg.status_code == 422


@pytest.mark.asyncio
async def test_resources_offset_and_legacy_skip(client: AsyncClient, auth_headers):
    ids = []
    for _ in range(3):
        r = await client.post(
            "/api/v1/resources",
            json={"name": uid("pg"), "resource_type": "solar", "rated_power": 5.0},
            headers=auth_headers,
        )
        assert r.status_code == 201, r.text
        ids.append(r.json()["id"])
    first = await client.get("/api/v1/resources", params={"limit": 2}, headers=auth_headers)
    total = int(first.headers["X-Total-Count"])
    assert total >= 3
    by_offset = await client.get(
        "/api/v1/resources", params={"limit": 1, "offset": 1}, headers=auth_headers
    )
    by_skip = await client.get(
        "/api/v1/resources", params={"limit": 1, "skip": 1}, headers=auth_headers
    )
    assert by_offset.json() == by_skip.json() == first.json()[1:2]
    assert by_offset.headers["X-Total-Count"] == str(total)
    for rid in ids:
        await client.delete(f"/api/v1/resources/{rid}", headers=auth_headers)


@pytest.mark.asyncio
async def test_cors_exposes_total_count(client: AsyncClient, admin_h):
    from vpp.settings import get_settings

    origins = get_settings().cors_origins
    if not origins or origins == ["*"]:
        pytest.skip("no explicit CORS origin configured")
    r = await client.get(
        "/api/v1/users", params={"limit": 1}, headers={**admin_h, "Origin": origins[0]}
    )
    assert "x-total-count" in r.headers.get("access-control-expose-headers", "").lower()
