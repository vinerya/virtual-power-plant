"""Telemetry freshness: thresholds, the resource API fields, the stale alert,
Modbus device health (degraded after failed polls) and releasing control
when a resource is set offline."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from _v2g_helpers import tmp_session_factory
from sqlalchemy import select
from test_device_control import SUNSPEC, Clock, FakeWriter, _actuator, _pv, _solar_row

from vpp.alert_service import (
    DEFAULT_RULES,
    STALE_METRIC,
    STALE_RULE_NAME,
    AlertRepository,
    AlertService,
)
from vpp.api.routes.protocols import _info
from vpp.control.actuator import SetpointActuator, set_setpoint_actuator
from vpp.db.models import (
    AlertModel,
    AlertRuleModel,
    AuditLogModel,
    BatteryStateModel,
    EventLogModel,
    ResourceModel,
)
from vpp.portal.telemetry import record_samples
from vpp.protocols.base import ProtocolStatus
from vpp.protocols.modbus import ModbusAdapter, RegisterDefinition, RegisterMap
from vpp.resources_freshness import (
    FreshnessSettings,
    is_stale,
    latest_telemetry_at,
    stale_after_s,
    stale_resource_ids,
    telemetry_freshness,
)
from vpp.settings import Settings

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)
CFG = FreshnessSettings(stale_after_s=300.0, modbus_poll_multiplier=10.0)


def _uid() -> str:
    return uuid.uuid4().hex[:8]


# ---------------------------------------------------------------------------
# Thresholds and status (pure)
# ---------------------------------------------------------------------------


def test_threshold_precedence():
    assert stale_after_s({}, CFG) == 300.0
    # Modbus: poll interval x multiplier (default poll interval 5 s)
    assert stale_after_s({"modbus": {"host": "x", "poll_interval_s": 2}}, CFG) == 20.0
    assert stale_after_s({"modbus": {"host": "x"}}, CFG) == 50.0
    # Polling off: the global threshold
    assert stale_after_s({"modbus": {"host": "x", "poll_interval_s": 0}}, CFG) == 300.0
    # The per-resource override wins; 0 turns the check off
    meta = {"telemetry_stale_after_s": 900, "modbus": {"host": "x"}}
    assert stale_after_s(meta, CFG) == 900.0
    assert stale_after_s({"telemetry_stale_after_s": 0}, CFG) is None
    # Junk overrides are ignored
    for junk in ("soon", -5, True, None):
        assert stale_after_s({"telemetry_stale_after_s": junk}, CFG) == 300.0
    # Multiplier 0: Modbus resources use the global threshold; global 0: off
    no_mult = FreshnessSettings(stale_after_s=120.0, modbus_poll_multiplier=0.0)
    assert stale_after_s({"modbus": {"host": "x"}}, no_mult) == 120.0
    assert stale_after_s({}, FreshnessSettings(stale_after_s=0.0)) is None
    # Read from the application settings
    settings = Settings(telemetry_stale_after_seconds=60, telemetry_stale_modbus_poll_multiplier=3)
    assert stale_after_s({}, settings) == 60.0
    assert stale_after_s({"modbus": {"poll_interval_s": 10}}, settings) == 30.0


def test_freshness_status_values():
    f = telemetry_freshness(None, 300.0, NOW)
    assert (f.data_status, f.stale, f.age_s) == ("no_data", False, None)
    f = telemetry_freshness(NOW - timedelta(seconds=299), 300.0, NOW)
    assert (f.data_status, f.stale, f.age_s) == ("fresh", False, 299.0)
    f = telemetry_freshness(NOW - timedelta(seconds=301), 300.0, NOW)
    assert (f.data_status, f.stale) == ("stale", True)
    f = telemetry_freshness(NOW - timedelta(days=3), None, NOW)
    assert (f.data_status, f.stale, f.stale_after_s) == ("unchecked", False, None)
    # Timestamps slightly ahead of the clock are age 0; naive means UTC
    assert telemetry_freshness(NOW + timedelta(seconds=30), 300.0, NOW).age_s == 0.0
    naive = (NOW - timedelta(seconds=10)).replace(tzinfo=None)
    assert telemetry_freshness(naive, 300.0, NOW).age_s == 10.0


def test_is_stale_accepts_rows_assets_and_dicts():
    from vpp.optimization.planning import FleetAsset

    old = NOW - timedelta(seconds=120)
    row = ResourceModel(id="r", metadata_json=json.dumps({"modbus": {"poll_interval_s": 5}}))
    assert is_stale(row, NOW, CFG, last_telemetry_at=old)  # 120 s > 50 s
    asset = FleetAsset(id="a", name="a", resource_type="solar", rated_power_kw=1, metadata={})
    assert not is_stale(asset, NOW, CFG, last_telemetry_at=old)  # 120 s < 300 s
    assert not is_stale({"id": "d", "metadata": {}}, NOW, CFG, last_telemetry_at=None)
    assert is_stale({"metadata": {"telemetry_stale_after_s": 60}}, NOW, CFG, last_telemetry_at=old)


# ---------------------------------------------------------------------------
# Last telemetry time (both tables)
# ---------------------------------------------------------------------------


async def test_latest_telemetry_at_merges_both_tables(tmp_path):
    db = tmp_session_factory(tmp_path / "fresh.db")
    async with db() as s:
        for rid in ("a", "b", "c"):
            s.add(ResourceModel(id=rid, name=rid, resource_type="battery", rated_power=5))
        await s.flush()
        await record_samples(
            s, "a", [(NOW - timedelta(seconds=600), 1.0, None), (NOW, 1.0, None)], source="t"
        )
        s.add(BatteryStateModel(resource_id="a", soc=50, timestamp=NOW - timedelta(seconds=5)))
        s.add(BatteryStateModel(resource_id="b", soc=50, timestamp=NOW - timedelta(seconds=900)))
        await s.commit()
        last = await latest_telemetry_at(s, ["a", "b", "c", "missing"])
        assert last == {"a": NOW, "b": NOW - timedelta(seconds=900)}
        assert await latest_telemetry_at(s, []) == {}
        rows = list((await s.execute(select(ResourceModel))).scalars().all())
        assert await stale_resource_ids(s, rows, NOW, CFG) == {"b"}


# ---------------------------------------------------------------------------
# Resource API fields
# ---------------------------------------------------------------------------


async def _create(client, headers, **metadata) -> str:
    resp = await client.post(
        "/api/v1/resources",
        json={
            "name": f"fresh-{_uid()}",
            "resource_type": "solar",
            "rated_power": 5,
            "metadata": metadata,
        },
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["data_status"] == "no_data" and body["telemetry_stale"] is False
    assert body["telemetry_last_at"] is None and body["link_status"] is None
    return body["id"]


async def _post_sample(client, headers, rid: str, age_s: float) -> None:
    ts = (datetime.now(timezone.utc) - timedelta(seconds=age_s)).isoformat()
    resp = await client.post(
        f"/api/v1/resources/{rid}/telemetry",
        json={"samples": [{"timestamp": ts, "power_kw": 1.0}]},
        headers=headers,
    )
    assert resp.status_code == 202, resp.text


async def test_resource_api_reports_freshness(client, auth_headers):
    rid = await _create(client, auth_headers)
    await _post_sample(client, auth_headers, rid, age_s=1200)
    body = (await client.get(f"/api/v1/resources/{rid}", headers=auth_headers)).json()
    assert body["data_status"] == "stale" and body["telemetry_stale"] is True
    assert body["telemetry_stale_after_s"] == 300.0
    assert body["telemetry_age_s"] >= 1200
    assert body["current_power"] == 0.0  # an old sample is history only

    await _post_sample(client, auth_headers, rid, age_s=0)
    listed = (await client.get("/api/v1/resources?limit=200", headers=auth_headers)).json()
    [body] = [r for r in listed if r["id"] == rid]
    assert body["data_status"] == "fresh" and body["telemetry_age_s"] < 60

    # The per-resource override (0 = off) shows in the response
    resp = await client.put(
        f"/api/v1/resources/{rid}",
        json={"metadata": {"telemetry_stale_after_s": 0}},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["data_status"] == "unchecked"
    assert resp.json()["telemetry_stale_after_s"] is None


# ---------------------------------------------------------------------------
# Built-in stale alert
# ---------------------------------------------------------------------------


async def _seeded_service(db, **kw) -> AlertService:
    async with db() as s:
        assert await AlertRepository.seed_default_rules(s) == len(DEFAULT_RULES)
        await s.commit()
    svc = AlertService(db, freshness=CFG, **kw)
    await svc.reload_rules()
    return svc


async def _stale_alerts(db, rid: str) -> list[AlertModel]:
    async with db() as s:
        stmt = (
            select(AlertModel)
            .where(AlertModel.source == rid, AlertModel.metric == STALE_METRIC)
            .order_by(AlertModel.fired_at)
        )
        return list((await s.execute(stmt)).scalars().all())


async def _sample(db, rid: str, at: datetime) -> None:
    async with db() as s:
        await record_samples(s, rid, [(at, 1.0, None)], source="t")
        await s.commit()


async def test_stale_alert_fires_once_per_episode_and_auto_resolves(tmp_path):
    db = tmp_session_factory(tmp_path / "alerts.db")
    async with db() as s:
        s.add(ResourceModel(id="pv", name="roof-pv", resource_type="solar", rated_power=5))
        s.add(ResourceModel(id="quiet", name="quiet", resource_type="solar", rated_power=5))
        await s.commit()
    await _sample(db, "pv", NOW - timedelta(seconds=100))
    svc = await _seeded_service(db)

    assert await svc.check_freshness(NOW) == []  # 100 s old: fresh
    fired = await svc.check_freshness(NOW + timedelta(seconds=250))  # 350 s: stale
    assert [a["source"] for a in fired] == ["pv"]  # "quiet" never reported: no alert
    [alert] = fired
    assert alert["title"] == STALE_RULE_NAME and alert["severity"] == "warning"
    assert "No telemetry for 350 s (threshold 300 s)" in alert["message"]
    assert "roof-pv" in alert["message"]
    assert alert["value"] == 350.0 and alert["threshold"] == 300.0

    # Still stale: nothing new, the open alert is not touched
    assert await svc.check_freshness(NOW + timedelta(seconds=500)) == []
    [row] = await _stale_alerts(db, "pv")
    assert row.status == "active" and row.occurrences == 1

    # Data returns: resolved automatically
    await _sample(db, "pv", NOW + timedelta(seconds=590))
    assert await svc.check_freshness(NOW + timedelta(seconds=600)) == []
    [row] = await _stale_alerts(db, "pv")
    assert row.status == "resolved" and row.resolved_by == "auto"

    # A second outage is a second alert
    assert len(await svc.check_freshness(NOW + timedelta(seconds=1000))) == 1
    assert [r.status for r in await _stale_alerts(db, "pv")] == ["resolved", "active"]

    # Set offline: the operator knows, the alert resolves
    async with db() as s:
        (await s.get(ResourceModel, "pv")).online = False
        await s.commit()
    assert await svc.check_freshness(NOW + timedelta(seconds=1100)) == []
    assert [r.status for r in await _stale_alerts(db, "pv")] == ["resolved", "resolved"]


async def test_stale_alert_survives_restart_and_respects_disabled_rule(tmp_path):
    db = tmp_session_factory(tmp_path / "alerts.db")
    async with db() as s:
        s.add(ResourceModel(id="pv", name="pv", resource_type="solar", rated_power=5))
        await s.commit()
    await _sample(db, "pv", NOW - timedelta(seconds=1000))
    svc = await _seeded_service(db)
    assert len(await svc.check_freshness(NOW)) == 1

    # A restarted evaluator re-feeds the state: de-duplicated, no new alert
    svc2 = AlertService(db, freshness=CFG)
    await svc2.reload_rules()
    assert await svc2.check_freshness(NOW + timedelta(seconds=60)) == []
    [row] = await _stale_alerts(db, "pv")
    assert row.status == "active"

    # Disabled rule: nothing fires, but data coming back still resolves
    async with db() as s:
        rule = await AlertRepository.get_rule_by_name(s, STALE_RULE_NAME)
        rule.enabled = False
        await s.commit()
    await svc2.reload_rules()
    await _sample(db, "pv", NOW + timedelta(seconds=100))
    await svc2.check_freshness(NOW + timedelta(seconds=110))
    [row] = await _stale_alerts(db, "pv")
    assert row.status == "resolved"
    assert await svc2.check_freshness(NOW + timedelta(seconds=2000)) == []
    assert len(await _stale_alerts(db, "pv")) == 1


async def test_started_monitor_gives_old_data_a_full_threshold(tmp_path):
    """After a restart (API down for a while) devices get time to report."""
    db = tmp_session_factory(tmp_path / "alerts.db")
    async with db() as s:
        s.add(ResourceModel(id="pv", name="pv", resource_type="solar", rated_power=5))
        await s.commit()
    await _sample(db, "pv", datetime.now(timezone.utc) - timedelta(hours=2))
    svc = await _seeded_service(db, freshness_check_interval_s=3600)
    from vpp.events import EventBus

    await svc.start(EventBus())
    try:
        now = datetime.now(timezone.utc)
        assert await svc.check_freshness(now + timedelta(seconds=60)) == []
        assert len(await svc.check_freshness(now + timedelta(seconds=400))) == 1
    finally:
        await svc.stop()


async def test_stale_rule_is_added_to_an_existing_rule_table(tmp_path):
    db = tmp_session_factory(tmp_path / "rules.db")
    async with db() as s:
        s.add(AlertRuleModel(name="Custom", metric="soc", comparison="<", threshold=0.2))
        await s.commit()
        assert await AlertRepository.seed_default_rules(s) == 1
        await s.commit()
        assert await AlertRepository.seed_default_rules(s) == 0
        names = {r.name for r in await AlertRepository.list_rules(s)}
    assert names == {"Custom", STALE_RULE_NAME}
    assert STALE_RULE_NAME in {r["name"] for r in DEFAULT_RULES}


async def test_no_freshness_query_without_a_stale_rule(tmp_path):
    db = tmp_session_factory(tmp_path / "none.db")
    svc = AlertService(db, freshness=CFG)
    await svc.reload_rules()
    assert await svc.check_freshness(NOW) == []


# ---------------------------------------------------------------------------
# Modbus device health
# ---------------------------------------------------------------------------


class _Result:
    def __init__(self, words: list[int]) -> None:
        self.registers = words

    def isError(self) -> bool:
        return False


class _FlakyClient:
    def __init__(self) -> None:
        self.down = False

    async def read_holding_registers(self, address, count=1, device_id=1):
        if self.down:
            raise ConnectionError("device gone")
        return _Result([1500] * count)

    def close(self) -> None:
        pass


def _polling_adapter(**config) -> tuple[ModbusAdapter, _FlakyClient]:
    adapter = ModbusAdapter()
    adapter.name = "modbus:pv"
    adapter.configure(**config)
    client = _FlakyClient()
    adapter._client = client
    adapter._register_map = RegisterMap("custom", {"ac_power": RegisterDefinition(address=0)})
    adapter._status = ProtocolStatus.CONNECTED
    return adapter, client


async def test_modbus_device_turns_degraded_after_failed_polls_and_recovers():
    adapter, client = _polling_adapter(max_failed_polls=3)
    assert await adapter.poll_once() == {"ac_power": 1500.0}
    assert adapter.last_successful_poll_at is not None

    client.down = True
    for n in (1, 2):
        assert await adapter.poll_once() == {}
        assert adapter.status == ProtocolStatus.CONNECTED
        assert adapter.consecutive_failed_polls == n
    await adapter.poll_once()
    assert adapter.status == ProtocolStatus.DEGRADED
    assert adapter.link_open and not adapter.is_connected  # writes refused
    info = _info(adapter)
    assert info.status == "degraded" and info.consecutive_failed_polls == 3
    with pytest.raises(ConnectionError):
        await adapter.write_registers(0, [1])

    # Still polled while degraded; the first good poll recovers
    await adapter.poll_once()
    assert adapter.consecutive_failed_polls == 4
    client.down = False
    assert await adapter.poll_once() == {"ac_power": 1500.0}
    assert adapter.status == ProtocolStatus.CONNECTED
    assert adapter.consecutive_failed_polls == 0
    assert _info(adapter).status == "connected"


async def test_modbus_failed_poll_threshold_default_and_poll_loop():
    adapter, client = _polling_adapter()
    assert adapter.max_failed_polls == 3
    client.down = True
    loop_adapter, loop_client = _polling_adapter(max_failed_polls=1)
    loop_client.down = True
    import asyncio

    task = asyncio.create_task(loop_adapter._poll_loop(0.01))
    try:
        for _ in range(100):
            if loop_adapter.status == ProtocolStatus.DEGRADED:
                break
            await asyncio.sleep(0.01)
        assert loop_adapter.status == ProtocolStatus.DEGRADED
        loop_client.down = False
        for _ in range(100):
            if loop_adapter.status == ProtocolStatus.CONNECTED:
                break
            await asyncio.sleep(0.01)
        assert loop_adapter.status == ProtocolStatus.CONNECTED  # loop kept polling
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_resource_api_shows_the_modbus_link_status(client, auth_headers):
    from vpp.api.routes.protocols import get_registry

    rid = await _create(client, auth_headers)
    adapter, _ = _polling_adapter()
    adapter.name = f"modbus:{rid}"
    adapter._status = ProtocolStatus.DEGRADED
    get_registry().register(adapter)
    try:
        body = (await client.get(f"/api/v1/resources/{rid}", headers=auth_headers)).json()
        assert body["link_status"] == "degraded"
    finally:
        get_registry().unregister(adapter.name)


# ---------------------------------------------------------------------------
# online: false releases control
# ---------------------------------------------------------------------------

CONTROL = {"host": "10.0.0.9", "control": {"enabled": True, "address": 40, "unit": "W"}}


@pytest.fixture
def fake_actuator():
    writers: dict[str, FakeWriter] = {}

    def factory(rid, modbus, control, rated):
        return writers.setdefault(rid, FakeWriter())

    actuator = SetpointActuator(enabled=True, writer_factory=factory)
    set_setpoint_actuator(actuator)
    yield actuator, writers
    set_setpoint_actuator(None)


async def _audit_rows(db_session, rid: str, action: str) -> list[AuditLogModel]:
    stmt = select(AuditLogModel).where(
        AuditLogModel.target_id == rid, AuditLogModel.action == action
    )
    return list((await db_session.execute(stmt)).scalars().all())


async def test_setting_offline_releases_the_setpoint_and_is_audited(
    client, auth_headers, fake_actuator, db_session
):
    actuator, writers = fake_actuator
    resp = await client.post(
        "/api/v1/resources",
        json={
            "name": f"ctl-{_uid()}",
            "resource_type": "battery",
            "rated_power": 50,
            "capacity_kwh": 200,
            "metadata": {"modbus": CONTROL},
        },
        headers=auth_headers,
    )
    rid = resp.json()["id"]
    resp = await client.post(
        "/api/v1/optimization/dispatch",
        json={"target_power_kw": 20.0, "resource_ids": [rid], "apply": True},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    assert rid in {a["resource_id"] for a in actuator.status()["active"]}

    resp = await client.put(
        f"/api/v1/resources/{rid}", json={"online": False}, headers=auth_headers
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["online"] is False
    assert writers[rid].calls[-1] == ("release", None)
    assert actuator.status()["active"] == []

    [entry] = await _audit_rows(db_session, rid, "control.resource_offline")
    assert entry.outcome == "success" and entry.actor_username == "testadmin"
    details = json.loads(entry.details_json)
    assert details["released"] == {"accepted": 1} and details["modes"] == ["release"]
    # The release itself is in the device command log like other control actions
    logged = (
        (
            await db_session.execute(
                select(EventLogModel).where(
                    EventLogModel.resource_id == rid, EventLogModel.event_type == "device_setpoint"
                )
            )
        )
        .scalars()
        .all()
    )
    sources = [json.loads(r.details_json)["source"] for r in logged]
    assert "api.resources.offline" in sources

    # Already offline, nothing held: no write, no audit entry
    await client.put(f"/api/v1/resources/{rid}", json={"online": False}, headers=auth_headers)
    assert len(await _audit_rows(db_session, rid, "control.resource_offline")) == 1
    # Back online is audited too
    await client.put(f"/api/v1/resources/{rid}", json={"online": True}, headers=auth_headers)
    assert len(await _audit_rows(db_session, rid, "control.resource_online")) == 1


async def test_offline_without_control_is_audited_and_writes_nothing(
    client, auth_headers, fake_actuator, db_session
):
    _, writers = fake_actuator
    rid = await _create(client, auth_headers)
    resp = await client.put(
        f"/api/v1/resources/{rid}", json={"online": False}, headers=auth_headers
    )
    assert resp.status_code == 200
    assert writers == {}
    [entry] = await _audit_rows(db_session, rid, "control.resource_offline")
    assert json.loads(entry.details_json)["released"] == {}


async def test_release_resource_handles_restored_caps(tmp_path):
    db = tmp_session_factory(tmp_path / "caps.db")
    for rid in ("pv1", "pv2", "pv3"):
        await _solar_row(db, rid)
    act, _ = _actuator(Clock(1_000.0), db)
    await act.apply(
        [_pv("pv1", **SUNSPEC), _pv("pv2", revert_timeout_s=60, **SUNSPEC), _pv("pv3", **SUNSPEC)],
        {"pv1": 4.0, "pv2": 4.0, "pv3": 4.0},
        source="t",
    )
    # Restart: the new process only knows the caps from the database
    act2, writers2 = _actuator(Clock(1_100.0), db)
    assert await act2.load_output_caps() == 2  # pv2's revert timer ran out
    meta = {"modbus": {"host": "x", "control": {"enabled": True, "address": 1, **SUNSPEC}}}

    [d] = await act2.release_resource("pv1", name="pv1", metadata=meta, reason="offline")
    assert d["status"] == "accepted" and d["mode"] == "release"
    assert writers2["pv1"].calls == [("release", None)]
    assert "pv1" not in act2.output_limits()

    # No usable control block: the cap is kept (the device holds it)
    [d] = await act2.release_resource("pv3", metadata={}, reason="offline")
    assert d["status"] == "failed" and "revert timer" in d["reason"]
    assert "pv3" in act2.output_limits()

    # Nothing held: nothing to do
    assert await act2.release_resource("pv2", metadata=meta, reason="offline") == []
    assert await act2.release_resource("unknown", reason="offline") == []


async def test_release_resource_of_expired_restored_cap_writes_nothing(tmp_path):
    db = tmp_session_factory(tmp_path / "caps.db")
    await _solar_row(db, "pv")
    act, _ = _actuator(Clock(1_000.0), db)
    await act.apply([_pv("pv", revert_timeout_s=600, **SUNSPEC)], {"pv": 4.0}, source="t")
    clock = Clock(1_100.0)
    act2, writers2 = _actuator(clock, db)
    assert await act2.load_output_caps() == 1
    clock.t = 2_000.0  # the device has reverted by itself since
    [d] = await act2.release_resource("pv", metadata={}, reason="offline")
    assert d["status"] == "unchanged" and d["mode"] == "reverted"
    assert writers2 == {}
    assert act2.output_limits() == {}
