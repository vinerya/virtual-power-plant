"""Alerts API, persistent AlertService evaluation and lifespan wiring."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from vpp.alert_service import (
    DEFAULT_RULES,
    AlertRepository,
    AlertService,
    effective_status,
    get_alert_service,
    serialize_alert,
)
from vpp.alerts import AlertManager, AlertRule, RuleType
from vpp.api import app as app_module
from vpp.api.websocket import manager as ws_manager
from vpp.db import engine as db_engine
from vpp.db.engine import get_session_factory
from vpp.db.models import AlertModel, AlertRuleModel, ResourceModel
from vpp.events import Event, EventType, get_event_bus, reset_event_bus
from vpp.events.bus import EventBus
from vpp.metrics import REGISTRY
from vpp.settings import Settings


def _uid() -> str:
    return uuid.uuid4().hex[:10]


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _make_alert(db_session, **overrides) -> AlertModel:
    now = _now()
    fields = {
        "rule_name": "Test rule",
        "severity": "warning",
        "status": "active",
        "title": "Test alert",
        "message": "something happened",
        "source": f"res-{_uid()}",
        "source_kind": "resource",
        "metric": "soc",
        "value": 0.05,
        "threshold": 0.1,
        "fired_at": now,
        "last_fired_at": now,
    }
    fields.update(overrides)
    row = AlertModel(**fields)
    db_session.add(row)
    await db_session.commit()
    return row


class FakeBroadcaster:
    def __init__(self) -> None:
        self.sent: list[tuple[str, dict]] = []

    async def broadcast(self, channel: str, data: dict) -> None:
        self.sent.append((channel, data))


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def accept(self) -> None:
        pass

    async def send_text(self, data: str) -> None:
        self.sent.append(data)


# ---------------------------------------------------------------------------
# REST: alerts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_requires_auth(client: AsyncClient):
    assert (await client.get("/api/v1/alerts")).status_code == 401


@pytest.mark.asyncio
async def test_list_alerts_matches_console_shape(client, auth_headers, db_session):
    row = await _make_alert(db_session, severity="critical")
    resp = await client.get(f"/api/v1/alerts?source={row.source}", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    [item] = resp.json()
    # Fields read by web/lib/api/types.ts::Alert
    for key in (
        "id",
        "timestamp",
        "severity",
        "source",
        "source_kind",
        "source_link",
        "title",
        "message",
        "status",
        "snoozed_until",
        "acknowledged_at",
    ):
        assert key in item
    assert item["id"] == row.id
    assert item["severity"] == "critical"
    assert item["status"] == "active"
    assert item["source_link"] == f"/assets/{row.source}"
    datetime.fromisoformat(item["timestamp"])


@pytest.mark.asyncio
async def test_list_filters(client, auth_headers, db_session):
    src = f"res-{_uid()}"
    old = await _make_alert(db_session, source=src, fired_at=_now() - timedelta(days=3))
    warn = await _make_alert(db_session, source=src, severity="warning")
    crit = await _make_alert(db_session, source=src, severity="error")  # engine 'error'
    acked = await _make_alert(db_session, source=src, status="acknowledged")
    resolved = await _make_alert(db_session, source=src, status="resolved")

    async def ids(qs: str) -> set[str]:
        r = await client.get(f"/api/v1/alerts?source={src}&{qs}", headers=auth_headers)
        assert r.status_code == 200, r.text
        return {a["id"] for a in r.json()}

    since = (_now() - timedelta(days=1)).isoformat().replace("+00:00", "Z")
    assert old.id not in await ids(f"since={since}")
    assert warn.id in await ids(f"since={since}")
    assert await ids("severity=critical") == {crit.id}
    assert await ids("status=acknowledged") == {acked.id}
    assert await ids("status=resolved") == {resolved.id}
    assert resolved.id not in await ids("status=open")
    assert await ids("status=active") == {old.id, warn.id, crit.id}
    assert len(await ids("limit=2")) == 2
    r = await client.get("/api/v1/alerts?status=bogus", headers=auth_headers)
    assert r.status_code == 422

    # 'error' is reported as 'critical' to the console
    r = await client.get(f"/api/v1/alerts/{crit.id}", headers=auth_headers)
    assert r.json()["severity"] == "critical"


@pytest.mark.asyncio
async def test_get_unknown_alert_404(client, auth_headers):
    assert (await client.get("/api/v1/alerts/nope", headers=auth_headers)).status_code == 404
    assert (await client.post("/api/v1/alerts/nope/ack", headers=auth_headers)).status_code == 404


@pytest.mark.asyncio
async def test_acknowledge(client, auth_headers, viewer_headers, db_session):
    row = await _make_alert(db_session)
    denied = await client.post(f"/api/v1/alerts/{row.id}/ack", json={}, headers=viewer_headers)
    assert denied.status_code == 403

    resp = await client.post(f"/api/v1/alerts/{row.id}/ack", json={}, headers=auth_headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "acknowledged"
    assert body["acknowledged_at"] is not None
    assert body["acknowledged_by"] == "testadmin"


@pytest.mark.asyncio
async def test_snooze_and_expiry(client, auth_headers, db_session):
    row = await _make_alert(db_session)
    until = _now() + timedelta(minutes=30)
    resp = await client.post(
        f"/api/v1/alerts/{row.id}/snooze",
        json={"until": until.isoformat(), "duration_ms": 30 * 60 * 1000},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "snoozed"
    assert datetime.fromisoformat(body["snoozed_until"]) == until

    snoozed = await client.get(
        f"/api/v1/alerts?source={row.source}&status=snoozed", headers=auth_headers
    )
    assert [a["id"] for a in snoozed.json()] == [row.id]

    # Snooze via duration only
    resp = await client.post(
        f"/api/v1/alerts/{row.id}/snooze",
        json={"duration_ms": 1000},
        headers=auth_headers,
    )
    assert resp.status_code == 200

    # An expired snooze reads as active again.
    await db_session.refresh(row)
    row.snoozed_until = _now() - timedelta(seconds=1)
    await db_session.commit()
    active = await client.get(
        f"/api/v1/alerts?source={row.source}&status=active", headers=auth_headers
    )
    assert [a["status"] for a in active.json()] == ["active"]


@pytest.mark.asyncio
async def test_snooze_validation(client, auth_headers, db_session):
    row = await _make_alert(db_session)
    url = f"/api/v1/alerts/{row.id}/snooze"
    assert (await client.post(url, json={}, headers=auth_headers)).status_code == 422
    past = (_now() - timedelta(minutes=1)).isoformat()
    assert (await client.post(url, json={"until": past}, headers=auth_headers)).status_code == 422
    far = (_now() + timedelta(days=30)).isoformat()
    assert (await client.post(url, json={"until": far}, headers=auth_headers)).status_code == 422
    too_long = {"duration_ms": 8 * 24 * 3600 * 1000}
    assert (await client.post(url, json=too_long, headers=auth_headers)).status_code == 422


@pytest.mark.asyncio
async def test_resolve(client, auth_headers, db_session):
    row = await _make_alert(db_session)
    resp = await client.post(f"/api/v1/alerts/{row.id}/resolve", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "resolved"
    assert resp.json()["resolved_at"] is not None
    # Resolved alerts cannot be snoozed; ack is a no-op.
    snooze = await client.post(
        f"/api/v1/alerts/{row.id}/snooze",
        json={"duration_ms": 1000},
        headers=auth_headers,
    )
    assert snooze.status_code == 409
    ack = await client.post(f"/api/v1/alerts/{row.id}/ack", headers=auth_headers)
    assert ack.json()["status"] == "resolved"


# ---------------------------------------------------------------------------
# REST: rules
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rules_crud(client, auth_headers, viewer_headers):
    name = f"High temp {_uid()}"
    body = {
        "name": name,
        "metric": "temperature",
        "comparison": ">",
        "threshold": 45,
        "severity": "critical",
    }
    assert (
        await client.post("/api/v1/alerts/rules", json=body, headers=viewer_headers)
    ).status_code == 403

    created = await client.post("/api/v1/alerts/rules", json=body, headers=auth_headers)
    assert created.status_code == 201, created.text
    rule = created.json()
    assert rule["rule_type"] == "threshold" and rule["enabled"] is True
    rid = rule["id"]

    dup = await client.post("/api/v1/alerts/rules", json=body, headers=auth_headers)
    assert dup.status_code == 409

    listed = await client.get("/api/v1/alerts/rules", headers=viewer_headers)
    assert rid in {r["id"] for r in listed.json()}
    assert (await client.get(f"/api/v1/alerts/rules/{rid}", headers=viewer_headers)).json()[
        "name"
    ] == name

    patched = await client.patch(
        f"/api/v1/alerts/rules/{rid}",
        json={"threshold": 55, "enabled": False},
        headers=auth_headers,
    )
    assert patched.status_code == 200
    assert patched.json()["threshold"] == 55 and patched.json()["enabled"] is False
    assert patched.json()["metric"] == "temperature"

    bad = await client.post(
        "/api/v1/alerts/rules",
        json={**body, "name": "x" + name, "severity": "fatal"},
        headers=auth_headers,
    )
    assert bad.status_code == 422

    assert (
        await client.delete(f"/api/v1/alerts/rules/{rid}", headers=auth_headers)
    ).status_code == 204
    assert (
        await client.get(f"/api/v1/alerts/rules/{rid}", headers=auth_headers)
    ).status_code == 404
    assert (
        await client.delete(f"/api/v1/alerts/rules/{rid}", headers=auth_headers)
    ).status_code == 404


# ---------------------------------------------------------------------------
# AlertManager per-source scoping
# ---------------------------------------------------------------------------


def test_manager_scopes_cooldown_per_source():
    mgr = AlertManager()
    mgr.add_rule(
        AlertRule(
            name="r",
            rule_type=RuleType.THRESHOLD,
            metric_name="soc",
            threshold=0.1,
            comparison="<",
            cooldown_s=3600,
        )
    )
    assert len(mgr.check("soc", 0.05, source="a")) == 1
    assert mgr.check("soc", 0.05, source="a") == []  # cooldown for a
    [alert] = mgr.check("soc", 0.05, source="b")  # independent for b
    assert alert.source == "b" and alert.metadata["metric"] == "soc"


def test_manager_respects_rule_resource_scope():
    mgr = AlertManager()
    mgr.add_rule(
        AlertRule(
            name="r",
            rule_type=RuleType.THRESHOLD,
            metric_name="p",
            threshold=1,
            cooldown_s=0,
            resource_id="only-me",
        )
    )
    assert mgr.check("p", 5, source="other") == []
    assert len(mgr.check("p", 5, source="only-me")) == 1


def test_effective_status_and_serialize_naive_datetimes():
    row = AlertModel(
        id="x",
        status="snoozed",
        severity="info",
        source="system",
        source_kind="system",
        title="t",
        message="m",
        fired_at=datetime(2026, 1, 1, 12, 0),  # naive, as from SQLite
        last_fired_at=datetime(2026, 1, 1, 12, 0),
        snoozed_until=datetime(2026, 1, 1, 13, 0),
        occurrences=1,
    )
    at = datetime(2026, 1, 1, 12, 30, tzinfo=timezone.utc)
    assert effective_status(row, at) == "snoozed"
    assert effective_status(row, at + timedelta(hours=1)) == "active"
    out = serialize_alert(row, at)
    assert out["timestamp"] == "2026-01-01T12:00:00+00:00"
    assert out["source_link"] is None


# ---------------------------------------------------------------------------
# AlertService: telemetry -> persisted + broadcast alerts
# ---------------------------------------------------------------------------


async def _add_rule(**fields) -> AlertRuleModel:
    factory = get_session_factory()
    async with factory() as s:
        row = await AlertRepository.create_rule(s, **fields)
        await s.commit()
        return row


async def _alerts_for(source: str) -> list[AlertModel]:
    async with get_session_factory()() as s:
        return list(
            (
                await s.execute(
                    select(AlertModel)
                    .where(AlertModel.source == source)
                    .order_by(AlertModel.fired_at)
                )
            )
            .scalars()
            .all()
        )


@pytest.mark.asyncio
async def test_service_fires_dedupes_and_auto_resolves(app, db_session):
    metric = f"m_{_uid()}"
    rule = await _add_rule(
        name=f"Low {metric}",
        metric=metric,
        comparison="<",
        threshold=10.0,
        severity="critical",
        cooldown_s=0,
    )
    resource = ResourceModel(name=f"bat-{_uid()}", resource_type="battery", rated_power=5.0)
    db_session.add(resource)
    await db_session.commit()

    bus = EventBus()
    fake = FakeBroadcaster()
    svc = AlertService(get_session_factory(), broadcaster=fake)
    await svc.start(bus)
    fired_before = (
        REGISTRY.get_sample_value(
            "vpp_alerts_fired_total",
            {"severity": "critical", "rule": rule.name},
        )
        or 0.0
    )
    try:

        async def publish(value: float) -> None:
            await bus.publish(
                Event(
                    event_type=EventType.RESOURCE_UPDATED,
                    data={"resource_id": resource.id, metric: value},
                )
            )
            await svc.drain()

        await publish(50.0)  # in range
        assert await _alerts_for(resource.id) == []

        await publish(5.0)  # fires
        [row] = await _alerts_for(resource.id)
        assert row.status == "active" and row.severity == "critical"
        assert row.rule_id == rule.id and row.title == rule.name
        assert resource.name in row.message
        assert len(fake.sent) == 1
        channel, payload = fake.sent[0]
        assert (
            channel == "alerts" and payload["id"] == row.id and payload["severity"] == "critical"
        )
        assert (
            REGISTRY.get_sample_value(
                "vpp_alerts_fired_total",
                {"severity": "critical", "rule": rule.name},
            )
            == fired_before + 1
        )

        await publish(4.0)  # still low: de-duplicated onto the same alert
        [row] = await _alerts_for(resource.id)
        assert row.occurrences == 2 and row.value == 4.0
        assert len(fake.sent) == 1

        await publish(20.0)  # back in range: auto-resolved
        [row] = await _alerts_for(resource.id)
        assert row.status == "resolved" and row.resolved_by == "auto"

        await publish(1.0)  # fires again as a new alert
        rows = await _alerts_for(resource.id)
        assert [r.status for r in rows] == ["resolved", "active"]
        assert len(fake.sent) == 2
    finally:
        await svc.stop()
    assert bus.subscriber_count == 0


@pytest.mark.asyncio
async def test_service_normalises_soc_percent_and_ignores_other_events(app):
    metric = "soc"
    rid = f"res-{_uid()}"
    rule = await _add_rule(
        name=f"soc low {_uid()}",
        metric=metric,
        comparison="<",
        threshold=0.1,
        cooldown_s=0,
        resource_id=rid,
    )
    bus = EventBus()
    svc = AlertService(get_session_factory(), broadcaster=FakeBroadcaster())
    await svc.start(bus)
    try:
        await bus.publish(
            Event(event_type=EventType.RESOURCE_UPDATED, data={"resource_id": rid, "soc": 50.0})
        )  # 50% -> 0.5
        await bus.publish(
            Event(event_type=EventType.RESOURCE_FAULT, data={"resource_id": rid, "soc": 1.0})
        )  # not telemetry
        await svc.drain()
        assert await _alerts_for(rid) == []
        await bus.publish(
            Event(event_type=EventType.RESOURCE_UPDATED, data={"resource_id": rid, "soc": 5.0})
        )  # 5% -> 0.05
        await svc.drain()
        [row] = await _alerts_for(rid)
        assert row.value == pytest.approx(0.05)
        assert row.source_kind == "resource"
    finally:
        await svc.stop()
        async with get_session_factory()() as s:
            await s.delete(await s.get(AlertRuleModel, rule.id))
            await s.commit()


@pytest.mark.asyncio
async def test_service_reload_picks_up_rule_changes(app):
    metric = f"m_{_uid()}"
    bus = EventBus()
    svc = AlertService(get_session_factory())
    await svc.start(bus)
    try:
        assert metric not in svc.manager.metric_names()
        rule = await _add_rule(name=f"r {metric}", metric=metric, threshold=1.0, cooldown_s=0)
        await svc.reload_rules()
        assert metric in svc.manager.metric_names()

        async with get_session_factory()() as s:
            row = await s.get(AlertRuleModel, rule.id)
            row.enabled = False
            await s.commit()
        await svc.reload_rules()
        assert metric not in svc.manager.metric_names()
    finally:
        await svc.stop()


@pytest.mark.asyncio
async def test_service_drops_when_queue_full(app):
    metric = f"m_{_uid()}"
    await _add_rule(name=f"r {metric}", metric=metric, threshold=1.0)
    svc = AlertService(get_session_factory(), queue_size=1)
    await svc.reload_rules()  # not started: nothing drains the queue
    ev = Event(event_type=EventType.RESOURCE_UPDATED, data={"resource_id": "x", metric: 5})
    await svc._on_event(ev)
    await svc._on_event(ev)
    assert svc.dropped_events == 1


@pytest.mark.asyncio
async def test_rule_crud_reloads_running_service(client, auth_headers, monkeypatch):
    from vpp import alert_service as alert_service_module

    svc = AlertService(get_session_factory())
    await svc.start(EventBus())
    monkeypatch.setattr(alert_service_module, "_service", svc)
    try:
        metric = f"m_{_uid()}"
        resp = await client.post(
            "/api/v1/alerts/rules",
            json={"name": f"r {metric}", "metric": metric, "threshold": 1},
            headers=auth_headers,
        )
        assert resp.status_code == 201
        assert metric in svc.manager.metric_names()
        await client.delete(f"/api/v1/alerts/rules/{resp.json()['id']}", headers=auth_headers)
        assert metric not in svc.manager.metric_names()
    finally:
        await svc.stop()


# ---------------------------------------------------------------------------
# Lifespan: the real app evaluates telemetry and pushes on the alerts channel
# ---------------------------------------------------------------------------


@pytest.fixture
def preserve_db_globals():
    saved_engine = db_engine._engine
    saved_factory = db_engine._session_factory
    yield
    db_engine._engine = saved_engine
    db_engine._session_factory = saved_factory


@pytest.mark.asyncio
async def test_lifespan_wires_alert_service(monkeypatch, preserve_db_globals):
    db_file = f"test_lifespan_alerts_{_uid()}.db"
    reset_event_bus()
    test_settings = Settings(
        database_url=f"sqlite+aiosqlite:///./{db_file}",
        degradation_updater_enabled=False,
        alerts_enabled=True,
        alerts_seed_default_rules=True,
    )
    monkeypatch.setattr(app_module, "get_settings", lambda: test_settings)
    fastapi_app = app_module.create_app(rate_limit_enabled=False)
    ws = FakeWebSocket()
    try:
        async with fastapi_app.router.lifespan_context(fastapi_app):
            svc = get_alert_service()
            assert svc is not None
            async with get_session_factory()() as s:
                names = {r.name for r in await AlertRepository.list_rules(s)}
            assert names == {r["name"] for r in DEFAULT_RULES}

            await ws_manager.connect(ws)
            await ws_manager.subscribe(ws, "alerts")
            try:
                await get_event_bus().publish(
                    Event(
                        event_type=EventType.RESOURCE_UPDATED,
                        data={"resource_id": "bat-life", "soc": 4.0, "temperature": 25.0},
                        source="mqtt.telemetry_ingestion",
                    )
                )
                await asyncio.wait_for(svc.drain(), timeout=5)
            finally:
                await ws_manager.disconnect(ws)

            alert_msgs = [json.loads(m) for m in ws.sent]
            alert_msgs = [m for m in alert_msgs if m["channel"] == "alerts"]
            assert len(alert_msgs) == 1
            data = alert_msgs[0]["data"]
            assert data["title"] == "Battery SOC low"
            assert data["source"] == "bat-life"
            assert data["status"] == "active"
            assert (
                REGISTRY.get_sample_value("vpp_battery_soc", {"resource_id": "bat-life"}) == 0.04
            )
        assert get_alert_service() is None
    finally:
        reset_event_bus()
        if os.path.exists(db_file):
            os.remove(db_file)


@pytest.mark.asyncio
async def test_lifespan_alerts_disabled(monkeypatch, preserve_db_globals):
    db_file = f"test_lifespan_alerts_{_uid()}.db"
    test_settings = Settings(
        database_url=f"sqlite+aiosqlite:///./{db_file}",
        degradation_updater_enabled=False,
        alerts_enabled=False,
    )
    monkeypatch.setattr(app_module, "get_settings", lambda: test_settings)
    fastapi_app = app_module.create_app(rate_limit_enabled=False)
    try:
        async with fastapi_app.router.lifespan_context(fastapi_app):
            assert get_alert_service() is None
    finally:
        if os.path.exists(db_file):
            os.remove(db_file)
