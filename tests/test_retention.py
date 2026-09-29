"""Data retention: pruning old rows (src/vpp/retention.py, `vpp prune`)."""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

import pytest
from _v2g_helpers import tmp_session_factory
from click.testing import CliRunner
from sqlalchemy import func, select

from vpp import retention
from vpp.cli.main import cli
from vpp.cluster.lease import LeaderElector
from vpp.cluster.node import utcnow
from vpp.cluster.topology import LEASE_RETENTION
from vpp.db.models import (
    AlertModel,
    AuditLogModel,
    BatteryStateModel,
    EventLogModel,
    ResourceModel,
)
from vpp.settings import Settings

NOW = utcnow()


def _settings(**kw) -> Settings:
    return Settings(**kw)


@pytest.fixture
def factory(tmp_path):
    return tmp_session_factory(tmp_path / "retention.db")


async def _add(factory, *rows) -> None:
    async with factory() as s:
        s.add_all(rows)
        await s.commit()


def _audit(days_ago: float, action: str = "auth.login") -> AuditLogModel:
    return AuditLogModel(ts=NOW - timedelta(days=days_ago), action=action, outcome="success")


async def _count(factory, model) -> int:
    async with factory() as s:
        return int((await s.execute(select(func.count()).select_from(model))).scalar_one())


async def _audit_actions(factory) -> list[str]:
    async with factory() as s:
        return sorted((await s.execute(select(AuditLogModel.action))).scalars())


async def test_prune_respects_cutoff(factory):
    await _add(
        factory, _audit(400, "old"), _audit(366, "old"), _audit(364, "new"), _audit(1, "new")
    )
    await _add(
        factory,
        EventLogModel(event_type="x", created_at=NOW - timedelta(days=40)),
        EventLogModel(event_type="x", created_at=NOW - timedelta(days=5)),
    )
    results = await retention.prune(
        factory, _settings(audit_retention_days=365, event_log_retention_days=30), now=NOW
    )
    by_table = {r.table: r for r in results}
    assert by_table["audit_log"].rows == 2
    assert by_table["event_log"].rows == 1
    assert await _audit_actions(factory) == ["new", "new"]
    assert await _count(factory, EventLogModel) == 1


async def test_zero_keeps_forever(factory):
    await _add(factory, _audit(5000), _audit(10))
    results = await retention.prune(factory, _settings(audit_retention_days=0), now=NOW)
    audit = next(r for r in results if r.table == "audit_log")
    assert not audit.enabled and audit.cutoff is None and audit.rows == 0
    assert await _count(factory, AuditLogModel) == 2


def test_retention_enabled():
    zero = dict.fromkeys({r.setting for r in retention.RULES}, 0)
    assert retention.retention_enabled(Settings())  # defaults are finite
    assert not retention.retention_enabled(Settings(**zero))
    assert retention.retention_enabled(Settings(**{**zero, "audit_retention_days": 7}))


async def test_only_resolved_alerts_are_pruned(factory):
    old = NOW - timedelta(days=100)

    def alert(status: str, resolved_at=None) -> AlertModel:
        return AlertModel(status=status, fired_at=old, last_fired_at=old, resolved_at=resolved_at)

    await _add(
        factory,
        alert("resolved", old),
        alert("resolved", NOW - timedelta(days=1)),
        alert("resolved", None),  # falls back to last_fired_at
        alert("active"),
        alert("acknowledged"),
    )
    results = await retention.prune(factory, _settings(alert_retention_days=30), now=NOW)
    assert next(r for r in results if r.table == "alerts").rows == 2
    async with factory() as s:
        left = sorted((await s.execute(select(AlertModel.status))).scalars())
    assert left == ["acknowledged", "active", "resolved"]


async def test_telemetry_rule(factory):
    await _add(
        factory, ResourceModel(id="b1", name="b1", resource_type="battery", rated_power=10.0)
    )
    await _add(
        factory,
        BatteryStateModel(resource_id="b1", soc=50, timestamp=NOW - timedelta(days=500)),
        BatteryStateModel(resource_id="b1", soc=50, timestamp=NOW - timedelta(hours=1)),
    )
    await retention.prune(factory, _settings(telemetry_retention_days=366), now=NOW)
    assert await _count(factory, BatteryStateModel) == 1


async def test_deletes_in_batches(factory, monkeypatch, caplog):
    await _add(factory, *[_audit(400) for _ in range(5)], _audit(1))
    calls: list[int] = []
    real = retention._delete_batch

    async def spy(f, rule, cutoff, size):
        n = await real(f, rule, cutoff, size)
        if rule.table == "audit_log":
            calls.append(n)
        return n

    monkeypatch.setattr(retention, "_delete_batch", spy)
    with caplog.at_level(logging.INFO, logger="vpp.retention"):
        results = await retention.prune(
            factory, _settings(audit_retention_days=365), now=NOW, batch_size=2
        )
    assert calls == [2, 2, 1]
    assert next(r for r in results if r.table == "audit_log").rows == 5
    assert await _count(factory, AuditLogModel) == 1
    assert "deleted 5 row(s) from audit_log" in caplog.text
    # Pruning is logged, never audited.
    assert await _audit_actions(factory) == ["auth.login"]


async def test_dry_run_counts_without_deleting(factory):
    await _add(factory, _audit(400), _audit(400), _audit(1))
    results = await retention.prune(
        factory, _settings(audit_retention_days=365), now=NOW, dry_run=True
    )
    assert next(r for r in results if r.table == "audit_log").rows == 2
    assert await _count(factory, AuditLogModel) == 3


def test_cli_prune_dry_run_then_prune(tmp_path, monkeypatch):
    from vpp.settings import get_settings

    db = tmp_path / "cli.db"
    factory = tmp_session_factory(db)
    asyncio.run(_add(factory, _audit(400), _audit(400), _audit(1)))

    monkeypatch.setenv("VPP_DATABASE_URL", f"sqlite+aiosqlite:///{db}")
    monkeypatch.setenv("VPP_AUDIT_RETENTION_DAYS", "365")
    monkeypatch.setenv("VPP_EVENT_LOG_RETENTION_DAYS", "0")
    get_settings.cache_clear()
    try:
        dry = CliRunner().invoke(cli, ["prune", "--dry-run"])
        assert dry.exit_code == 0, dry.output
        assert "audit_log" in dry.output and "would delete        2 row(s)" in dry.output
        assert "event_log            kept forever (VPP_EVENT_LOG_RETENTION_DAYS=0)" in dry.output
        assert "Dry run: 2 row(s) would delete in total" in dry.output
        assert asyncio.run(_count(factory, AuditLogModel)) == 3

        real = CliRunner().invoke(cli, ["prune"])
        assert real.exit_code == 0, real.output
        assert "deleted        2 row(s)" in real.output
        assert asyncio.run(_count(factory, AuditLogModel)) == 1
    finally:
        get_settings.cache_clear()


async def test_retention_runs_only_on_lease_holder(factory, monkeypatch):
    """Two workers share the DB: only the data-retention lease holder prunes."""
    runs: list[str] = []
    real = retention.prune

    async def recording_prune(f, settings, **kw):
        runs.append("run")
        return await real(f, settings, **kw)

    monkeypatch.setattr(retention, "prune", recording_prune)
    await _add(factory, _audit(400), _audit(1))
    settings = _settings(audit_retention_days=365, retention_interval_minutes=60)

    def worker(holder: str) -> LeaderElector:
        return LeaderElector(
            LEASE_RETENTION,
            factory,
            leader=retention.retention_role(factory, settings, initial_delay_s=0),
            holder=holder,
            ttl_s=30,
        )

    a, b = worker("node-a"), worker("node-b")
    await a.step()
    await b.step()
    try:
        assert a.is_leader and not b.is_leader
        for _ in range(50):
            if runs:
                break
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.05)
        assert runs == ["run"]  # once, on node-a; node-b stays idle
        assert await _count(factory, AuditLogModel) == 1

        await a.stop()  # releases the lease; node-b takes over
        await b.step()
        assert b.is_leader
        for _ in range(50):
            if len(runs) == 2:
                break
            await asyncio.sleep(0.02)
        assert runs == ["run", "run"]
    finally:
        await a.stop()
        await b.stop()
