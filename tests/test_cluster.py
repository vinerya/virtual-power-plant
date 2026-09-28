"""Multi-process coordination: DB leases, leader election, forwarded calls, topology.

The lease / call / relay tests run on a throwaway SQLite file by default. Set
``VPP_CLUSTER_TEST_DB_URL`` to an *async* URL (e.g.
``postgresql+asyncpg://postgres@/vpp_cluster?host=/run/postgresql``) to run
them against PostgreSQL instead; the ``cluster_*`` tables there are
recreated.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import subprocess
import sys
import textwrap
import time
from datetime import timedelta
from pathlib import Path

import pytest
from _v2g_helpers import tmp_session_factory
from click.testing import CliRunner
from fastapi import HTTPException
from sqlalchemy import select

from vpp.cluster import lease as lease_module
from vpp.cluster import rpc
from vpp.cluster.lease import (
    LeaderElector,
    TaskRole,
    is_local,
    leadership,
    release,
    try_acquire,
)
from vpp.cluster.node import _hostname, holder_is_dead_local_process, node_id, utcnow
from vpp.cluster.topology import TopologyError, validate_topology
from vpp.db.models import ClusterCallModel, ClusterEventModel, ClusterLeaseModel
from vpp.settings import Settings

DB_URL_ENV = "VPP_CLUSTER_TEST_DB_URL"
_CLUSTER_TABLES = [
    ClusterLeaseModel.__table__,
    ClusterCallModel.__table__,
    ClusterEventModel.__table__,
]


@pytest.fixture
def db_url(tmp_path):
    url = os.environ.get(DB_URL_ENV)
    if not url:
        return f"sqlite+aiosqlite:///{tmp_path / 'cluster.db'}"
    from sqlalchemy import create_engine

    sync_url = url.replace("+asyncpg", "+psycopg2").replace("+aiosqlite", "")
    engine = create_engine(sync_url)
    from vpp.db.base import Base

    Base.metadata.drop_all(engine, tables=_CLUSTER_TABLES)
    Base.metadata.create_all(engine, tables=_CLUSTER_TABLES)
    engine.dispose()
    return url


@pytest.fixture
def factory(tmp_path, db_url):
    if db_url.startswith("sqlite"):
        return tmp_session_factory(tmp_path / "cluster.db")
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    engine = create_async_engine(db_url, poolclass=NullPool)
    return async_sessionmaker(engine, expire_on_commit=False)


class RecordingRole:
    def __init__(self) -> None:
        self.active = False
        self.starts = 0
        self.stops = 0

    async def start(self) -> None:
        self.active = True
        self.starts += 1

    async def stop(self) -> None:
        self.active = False
        self.stops += 1


# ---------------------------------------------------------------------------
# Lease primitives
# ---------------------------------------------------------------------------


async def test_lease_is_exclusive_until_it_expires(factory):
    now = utcnow()
    assert await try_acquire(factory, "job", "node-a", 10, now=now)
    assert not await try_acquire(factory, "job", "node-b", 10, now=now)
    # Renewal by the holder succeeds and keeps the original acquisition time.
    assert await try_acquire(factory, "job", "node-a", 10, now=now + timedelta(seconds=5))
    assert not await try_acquire(factory, "job", "node-b", 10, now=now + timedelta(seconds=14))
    # A holder that stopped renewing loses the lease once it has expired.
    assert await try_acquire(factory, "job", "node-b", 10, now=now + timedelta(seconds=16))
    assert not await try_acquire(factory, "job", "node-a", 10, now=now + timedelta(seconds=17))

    async with factory() as s:
        row = await s.get(ClusterLeaseModel, "job")
    assert row.holder == "node-b"


async def test_release_hands_over_immediately(factory):
    assert await try_acquire(factory, "job", "node-a", 60)
    assert not await release(factory, "job", "node-b")  # only the holder can release
    assert not await try_acquire(factory, "job", "node-b", 60)
    assert await release(factory, "job", "node-a")
    assert await try_acquire(factory, "job", "node-b", 60)


async def test_leases_are_independent(factory):
    assert await try_acquire(factory, "a", "node-a", 60)
    assert await try_acquire(factory, "b", "node-b", 60)
    assert not await try_acquire(factory, "a", "node-b", 60)


async def test_concurrent_first_acquisition_has_one_winner(factory):
    results = await asyncio.gather(
        *(try_acquire(factory, "race", f"node-{i}", 60) for i in range(5))
    )
    assert sum(results) == 1


async def test_dead_local_holder_is_taken_over_at_once(factory):
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    dead = f"{_hostname()}:{proc.pid}:deadbeef"
    assert holder_is_dead_local_process(dead)
    assert not holder_is_dead_local_process(node_id())
    assert not holder_is_dead_local_process(f"other-host:{proc.pid}:x")
    same_name_other_container = _hostname().split("~")[0] + "~1:" + f"{proc.pid}:x"
    assert not holder_is_dead_local_process(same_name_other_container)
    assert not holder_is_dead_local_process("no-structure")

    assert await try_acquire(factory, "job", dead, 600)
    assert await try_acquire(factory, "job", node_id(), 600)


# ---------------------------------------------------------------------------
# Elector
# ---------------------------------------------------------------------------


async def test_two_electors_one_leader_and_failover(factory):
    lead_a, follow_a = RecordingRole(), RecordingRole()
    lead_b, follow_b = RecordingRole(), RecordingRole()
    a = LeaderElector("job", factory, leader=lead_a, follower=follow_a, holder="node-a", ttl_s=30)
    b = LeaderElector("job", factory, leader=lead_b, follower=follow_b, holder="node-b", ttl_s=30)
    try:
        await a.step()
        await b.step()
        assert a.is_leader and not b.is_leader
        assert lead_a.active and not follow_a.active
        assert follow_b.active and not lead_b.active

        await b.step()  # still held by a: nothing changes, no restarts
        assert follow_b.starts == 1 and lead_b.starts == 0

        await a.stop()  # graceful shutdown releases the lease
        assert not lead_a.active
        await b.step()
        assert b.is_leader
        assert lead_b.active and not follow_b.active
    finally:
        await a.stop()
        await b.stop()


async def test_leader_that_loses_its_lease_stops_its_work(factory):
    role = RecordingRole()
    a = LeaderElector("job", factory, leader=role, holder="node-a", ttl_s=30)
    await a.step()
    assert a.is_leader and role.active
    async with factory() as s:  # someone else took it (e.g. we stalled past the TTL)
        row = await s.get(ClusterLeaseModel, "job")
        row.holder = "node-b"
        row.expires_at = utcnow() + timedelta(seconds=60)
        await s.commit()
    await a.step()
    assert not a.is_leader and not role.active
    await a.stop()


async def test_db_outage_keeps_leader_only_while_its_lease_is_valid(factory, monkeypatch):
    role = RecordingRole()
    a = LeaderElector("job", factory, leader=role, holder="node-a", ttl_s=0.3)
    await a.step()
    assert a.is_leader

    async def boom(*args, **kwargs):
        raise RuntimeError("database unreachable")

    monkeypatch.setattr(lease_module, "try_acquire", boom)
    await a.step()
    assert a.is_leader and role.active  # within the TTL: keep going
    await asyncio.sleep(0.35)
    assert not a.is_leader  # routing stops trusting it at once
    await a.step()
    assert not role.active  # and the work is stopped
    monkeypatch.undo()
    await a.stop()


async def test_start_acquires_inline_and_registers(factory):
    """A lone process is leader as soon as start() returns (single-process parity)."""
    started = asyncio.Event()

    async def work() -> None:
        started.set()
        await asyncio.sleep(3600)

    role = TaskRole(work, name="test-work")
    elector = LeaderElector("solo", factory, leader=role, ttl_s=30)
    assert is_local("solo")  # no elector running: local
    await elector.start()
    try:
        assert elector.is_leader
        assert leadership()["solo"] is True
        assert role.task is not None
        await asyncio.wait_for(started.wait(), 1)
    finally:
        task = role.task
        await elector.stop()
    assert task.cancelled() or task.done()
    assert "solo" not in leadership()
    async with factory() as s:
        assert await s.get(ClusterLeaseModel, "solo") is None  # released


_ELECTOR_PROCESS = textwrap.dedent(
    """
    import asyncio, sys, time
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool
    from vpp.cluster.lease import CallbackRole, LeaderElector

    async def main():
        engine = create_async_engine(sys.argv[1], poolclass=NullPool)
        factory = async_sessionmaker(engine, expire_on_commit=False)

        async def lead():
            print("LEAD", time.time(), flush=True)

        async def step_down():
            print("STOP", time.time(), flush=True)

        elector = LeaderElector(
            "failover", factory, leader=CallbackRole(lead, step_down), ttl_s=1.0
        )
        await elector.start()
        print("READY", flush=True)
        await asyncio.sleep(60)

    asyncio.run(main())
    """
)


def _spawn_elector(db_url: str) -> subprocess.Popen:
    import queue
    import threading

    import vpp

    env = dict(os.environ, PYTHONPATH=str(Path(vpp.__file__).resolve().parents[1]))
    proc = subprocess.Popen(
        [sys.executable, "-c", _ELECTOR_PROCESS, db_url],
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    lines: queue.Queue[str] = queue.Queue()

    def pump() -> None:
        for line in proc.stdout:
            lines.put(line.strip())

    threading.Thread(target=pump, daemon=True).start()
    proc.lines = lines  # type: ignore[attr-defined]
    return proc


def _read_until(proc: subprocess.Popen, word: str, timeout: float) -> list[str]:
    import queue

    lines: list[str] = []
    deadline = time.monotonic() + timeout
    while (left := deadline - time.monotonic()) > 0:
        try:
            line = proc.lines.get(timeout=left)  # type: ignore[attr-defined]
        except queue.Empty:
            break
        lines.append(line)
        if line.startswith(word):
            return lines
    raise AssertionError(f"no {word!r} within {timeout}s; got {lines}")


def test_two_processes_fail_over(db_url, factory, tmp_path):
    """Real OS processes: one leads, the other waits and takes over when it dies."""
    first = _spawn_elector(db_url)
    second = None
    try:
        assert _read_until(first, "READY", 20)[0].startswith("LEAD")
        second = _spawn_elector(db_url)
        assert _read_until(second, "READY", 20) == ["READY"]  # follower
        with pytest.raises(AssertionError):  # the leader keeps renewing past the TTL
            _read_until(second, "LEAD", 2.0)

        first.send_signal(signal.SIGKILL)  # crash, no release
        first.wait(5)
        lead = _read_until(second, "LEAD", 5.0)[-1]
        assert lead.startswith("LEAD")
    finally:
        for proc in (first, second):
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait(5)  # the pump thread then sees EOF and ends


async def test_follower_is_not_local(factory):
    assert await try_acquire(factory, "held", "node-elsewhere", 60)
    elector = LeaderElector("held", factory, holder="node-here", ttl_s=60)
    await elector.start()
    try:
        assert not elector.is_leader
        assert not is_local("held")
    finally:
        await elector.stop()
    assert is_local("held")


# ---------------------------------------------------------------------------
# Forwarded calls
# ---------------------------------------------------------------------------


@pytest.fixture
def echo_target():
    calls = []

    async def echo(payload, session):
        calls.append(payload)
        return {"echo": payload, "when": utcnow()}

    async def conflict(payload, session):
        raise HTTPException(409, detail={"code": "nope"})

    async def crash(payload, session):
        raise RuntimeError("bug")

    rpc.register_handler("t-echo", "echo", echo)
    rpc.register_handler("t-echo", "conflict", conflict)
    rpc.register_handler("t-echo", "crash", crash)
    yield calls
    for method in ("echo", "conflict", "crash"):
        rpc._handlers.pop(("t-echo", method), None)


@contextlib.asynccontextmanager
async def running_executor(factory, targets):
    task = asyncio.create_task(rpc.run_executor(factory, lambda: targets, poll_s=0.01))
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def test_call_is_executed_by_the_leader(factory, echo_target):
    async with running_executor(factory, ["t-echo"]):
        result = await rpc.call("t-echo", "echo", {"x": 1}, timeout_s=5, session_factory=factory)
    assert result["echo"] == {"x": 1}
    assert isinstance(result["when"], str)
    assert echo_target == [{"x": 1}]
    async with factory() as s:
        row = (await s.execute(select(ClusterCallModel))).scalar_one()
    assert row.status == "done" and row.executor == node_id()


async def test_http_errors_cross_the_process_boundary(factory, echo_target):
    async with running_executor(factory, ["t-echo"]):
        with pytest.raises(rpc.ClusterCallError) as conflict:
            await rpc.call("t-echo", "conflict", {}, timeout_s=5, session_factory=factory)
        with pytest.raises(rpc.ClusterCallError) as crash:
            await rpc.call("t-echo", "crash", {}, timeout_s=5, session_factory=factory)
    assert conflict.value.status_code == 409
    assert conflict.value.detail == {"code": "nope"}
    assert crash.value.status_code == 500


async def test_unclaimed_call_is_cancelled_and_never_runs(factory, echo_target):
    with pytest.raises(rpc.LeaderUnavailableError) as exc:
        await rpc.call("t-echo", "echo", {"x": 2}, timeout_s=0.2, session_factory=factory)
    assert exc.value.status_code == 503
    assert exc.value.detail["code"] == "leader_unavailable"
    # A leader that shows up later must not execute the abandoned call.
    assert await rpc.execute_pending(factory, ["t-echo"]) == 0
    assert echo_target == []
    async with factory() as s:
        row = (await s.execute(select(ClusterCallModel))).scalar_one()
    assert row.status == "cancelled"


async def test_expired_fire_and_forget_calls_are_dropped(factory, echo_target):
    await rpc.submit("t-echo", "echo", {"late": True}, timeout_s=0.01, session_factory=factory)
    await asyncio.sleep(0.05)
    assert await rpc.execute_pending(factory, ["t-echo"]) == 0
    assert echo_target == []


async def test_executor_only_serves_its_targets(factory, echo_target):
    await rpc.submit("t-echo", "echo", {}, timeout_s=30, session_factory=factory)
    assert await rpc.execute_pending(factory, ["some-other-lease"]) == 0
    assert await rpc.execute_pending(factory, ["t-echo"]) == 1
    assert await rpc.execute_pending(factory, ["t-echo"]) == 0  # exactly once


async def test_purge_removes_old_finished_calls(factory, echo_target):
    await rpc.submit("t-echo", "echo", {}, timeout_s=30, session_factory=factory)
    await rpc.execute_pending(factory, ["t-echo"])
    assert await rpc.purge_finished(factory, older_than=timedelta(hours=1)) == 0
    assert await rpc.purge_finished(factory, older_than=timedelta(seconds=-1)) == 1


# ---------------------------------------------------------------------------
# Trading venue across workers (this process plays both roles)
# ---------------------------------------------------------------------------


@pytest.fixture
async def follower_of_trading_venue(app):
    """Make this process a follower: another node holds the trading-venue lease."""
    from vpp.db.engine import get_session_factory

    factory = get_session_factory()
    async with factory() as s:
        await s.execute(
            ClusterLeaseModel.__table__.delete().where(ClusterLeaseModel.name == "trading-venue")
        )
        await s.commit()
    assert await try_acquire(factory, "trading-venue", "venue-leader-node", 120)
    elector = LeaderElector("trading-venue", factory, holder="this-worker", ttl_s=120)
    await elector.start()
    assert not is_local("trading-venue")
    yield factory
    await elector.stop()
    await release(factory, "trading-venue", "venue-leader-node")


async def test_follower_forwards_orders_to_the_venue_leader(
    client, auth_headers, follower_of_trading_venue
):
    from vpp.trading.service import TradingServiceConfig, reset_trading_service

    reset_trading_service(TradingServiceConfig(seed=11, base_volume=20.0))
    factory = follower_of_trading_venue
    async with running_executor(factory, ["trading-venue"]):  # the "leader process"
        markets = await client.get("/api/v1/trading/markets", headers=auth_headers)
        assert markets.status_code == 200, markets.text
        assert {m["market"] for m in markets.json()} >= {"day_ahead", "real_time"}

        resp = await client.post(
            "/api/v1/trading/orders",
            json={"order_type": "market", "market": "real_time", "side": "buy", "quantity": 1},
            headers=auth_headers,
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["status"] == "filled"
        assert body["metadata"]["submitted_by"] == "testadmin"
        assert body["fills"]

        bad = await client.post(
            "/api/v1/trading/orders",
            json={"order_type": "limit", "market": "nowhere", "side": "buy", "quantity": 1},
            headers=auth_headers,
        )
        assert bad.status_code == 422  # the leader's validation error, verbatim

        portfolio = await client.get("/api/v1/trading/portfolio", headers=auth_headers)
        assert portfolio.status_code == 200, portfolio.text
        assert portfolio.json()["venue"] == "simulated"

    async with factory() as s:
        rows = (
            (
                await s.execute(
                    select(ClusterCallModel).where(ClusterCallModel.target == "trading-venue")
                )
            )
            .scalars()
            .all()
        )
    assert {r.method for r in rows} >= {"markets", "submit_order", "portfolio"}
    assert all(r.status in ("done", "failed") for r in rows)
    reset_trading_service()


async def test_follower_without_a_live_leader_refuses_honestly(
    client, auth_headers, follower_of_trading_venue, monkeypatch
):
    from vpp.settings import get_settings

    monkeypatch.setattr(get_settings(), "cluster_call_timeout_seconds", 0.3)
    resp = await client.post(
        "/api/v1/trading/orders",
        json={"order_type": "market", "market": "real_time", "side": "buy", "quantity": 1},
        headers=auth_headers,
    )
    assert resp.status_code == 503
    assert resp.json()["detail"]["code"] == "leader_unavailable"
    assert resp.headers.get("retry-after") == "5"


# ---------------------------------------------------------------------------
# Alert evaluation across workers
# ---------------------------------------------------------------------------


async def test_follower_forwards_telemetry_to_the_alert_leader(factory, monkeypatch):
    from vpp import alert_service
    from vpp.events import Event, EventBus, EventType

    monkeypatch.setattr(rpc, "_factory", lambda sf: sf or factory)
    bus = EventBus()
    forwarder = alert_service.AlertForwarder(bus)
    await forwarder.start()
    try:
        await bus.publish(
            Event(EventType.RESOURCE_UPDATED, data={"resource_id": "bat-1", "soc": 0.03})
        )
        await bus.publish(Event(EventType.RESOURCE_UPDATED, data={"resource_id": "x"}))
        for _ in range(100):
            if forwarder.forwarded:
                break
            await asyncio.sleep(0.01)
    finally:
        await forwarder.stop()
    assert forwarder.forwarded == 1

    received = []

    class StubService:
        async def _on_event(self, event):
            received.append(event)

    monkeypatch.setattr(alert_service, "_service", StubService())
    assert await rpc.execute_pending(factory, [alert_service.ALERTS_LEASE]) == 1
    assert received[0].data == {"resource_id": "bat-1", "soc": 0.03}
    assert received[0].event_type == EventType.RESOURCE_UPDATED


# ---------------------------------------------------------------------------
# Topology + serve
# ---------------------------------------------------------------------------


def test_ocpp_requires_a_single_worker():
    validate_topology(Settings(api_workers=1, ocpp_enabled=True))
    validate_topology(Settings(api_workers=4, ocpp_enabled=False))
    with pytest.raises(TopologyError, match="VPP_API_WORKERS=1"):
        validate_topology(Settings(api_workers=2, ocpp_enabled=True))


def test_serve_uses_vpp_api_workers(monkeypatch):
    import uvicorn

    from vpp.cli.main import cli
    from vpp.settings import get_settings

    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: seen.update(kw))
    monkeypatch.setenv("VPP_API_WORKERS", "3")
    get_settings.cache_clear()
    try:
        result = CliRunner().invoke(cli, ["serve"])
        assert result.exit_code == 0, result.output
        assert seen["workers"] == 3

        result = CliRunner().invoke(cli, ["serve", "--workers", "2", "--port", "9000"])
        assert result.exit_code == 0, result.output
        assert seen["workers"] == 2 and seen["port"] == 9000

        monkeypatch.setenv("VPP_OCPP_ENABLED", "true")
        get_settings.cache_clear()
        seen.clear()
        result = CliRunner().invoke(cli, ["serve", "--workers", "2"])
        assert result.exit_code == 2
        assert "VPP_API_WORKERS=1" in result.output
        assert seen == {}
    finally:
        monkeypatch.delenv("VPP_OCPP_ENABLED", raising=False)
        monkeypatch.delenv("VPP_API_WORKERS", raising=False)
        get_settings.cache_clear()


@pytest.fixture
def preserve_db_globals():
    from vpp.db import engine as db_engine

    saved = (db_engine._engine, db_engine._session_factory)
    yield
    db_engine._engine, db_engine._session_factory = saved


async def test_single_process_lifespan_holds_every_lease(
    monkeypatch, preserve_db_globals, tmp_path, caplog
):
    from vpp.api import app as app_module

    async def idle(*args, **kwargs):
        await asyncio.sleep(3600)

    monkeypatch.setattr(app_module, "_degradation_periodic_loop", idle)
    monkeypatch.setattr(app_module, "_mqtt_ingestion_loop", idle)
    monkeypatch.setattr(app_module, "_modbus_ingestion_loop", idle)
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'solo.db'}",
        degradation_updater_enabled=True,
        mqtt_ingestion_enabled=True,
        modbus_ingestion_enabled=True,
        trading_market_data_interval_seconds=3600,
    )
    monkeypatch.setattr(app_module, "get_settings", lambda: settings)
    fastapi_app = app_module.create_app()
    with caplog.at_level(logging.INFO, logger="vpp.cluster.topology"):
        async with fastapi_app.router.lifespan_context(fastapi_app):
            held = leadership()
            assert held == {
                "alert-evaluator": True,
                "degradation-updater": True,
                "modbus-ingestion": True,
                "mqtt-ingestion": True,
                "trading-venue": True,
            }
            assert fastapi_app.state.degradation_task is not None
            assert fastapi_app.state.trading_market_data_task is not None
    assert "API topology: VPP_API_WORKERS=1" in caplog.text
    assert leadership() == {}


async def test_lifespan_refuses_ocpp_with_several_workers(monkeypatch, preserve_db_globals):
    from vpp.api import app as app_module

    settings = Settings(api_workers=2, ocpp_enabled=True)
    monkeypatch.setattr(app_module, "get_settings", lambda: settings)
    fastapi_app = app_module.create_app()
    with pytest.raises(TopologyError):
        async with fastapi_app.router.lifespan_context(fastapi_app):
            pass


async def test_second_worker_lifespan_stays_idle(monkeypatch, preserve_db_globals, tmp_path):
    """Another process holds every lease: this worker runs no singleton work."""
    from vpp.api import app as app_module

    db = tmp_path / "shared.db"
    other = tmp_session_factory(db)
    for name in ("degradation-updater", "trading-venue", "alert-evaluator"):
        assert await try_acquire(other, name, "worker-1", 120)

    ran = []

    async def loop(*args, **kwargs):
        ran.append(1)

    monkeypatch.setattr(app_module, "_degradation_periodic_loop", loop)
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{db}", api_workers=2, degradation_updater_enabled=True
    )
    monkeypatch.setattr(app_module, "get_settings", lambda: settings)
    from vpp.alert_service import get_alert_service

    fastapi_app = app_module.create_app()
    async with fastapi_app.router.lifespan_context(fastapi_app):
        assert leadership() == {
            "alert-evaluator": False,
            "degradation-updater": False,
            "trading-venue": False,
        }
        assert fastapi_app.state.degradation_task is None
        assert fastapi_app.state.trading_market_data_task is None
        assert get_alert_service() is None
    assert ran == []


# ---------------------------------------------------------------------------
# WebSocket relay
# ---------------------------------------------------------------------------


class _FakeSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def accept(self, subprotocol=None) -> None:
        pass

    async def send_text(self, text: str) -> None:
        import json

        self.sent.append(json.loads(text))


async def test_relay_delivers_broadcasts_to_other_workers_once(factory):
    from vpp.api.websocket import ConnectionManager
    from vpp.cluster.relay import WebSocketRelay

    managers, sockets, relays = [], [], []
    for origin in ("worker-a", "worker-b"):
        mgr, ws = ConnectionManager(), _FakeSocket()
        await mgr.connect(ws)
        await mgr.subscribe(ws, "market_data")
        relay = WebSocketRelay(mgr, factory, poll_s=3600, origin=origin)
        await relay.start()
        managers.append(mgr)
        sockets.append(ws)
        relays.append(relay)
    try:
        await managers[0].broadcast("market_data", {"price": 42.0})
        assert [m["data"]["price"] for m in sockets[0].sent] == [42.0]  # local, immediate
        for _ in range(100):  # the writer task batches inserts
            async with factory() as s:
                if (await s.execute(select(ClusterEventModel))).first():
                    break
            await asyncio.sleep(0.01)

        # (the reader task may already have picked it up on its first pass)
        await relays[1].poll_once()
        assert relays[1].relayed_in == 1
        assert await relays[0].poll_once() == 0  # own broadcasts are not echoed back
        assert await relays[1].poll_once() == 0  # delivered once
        assert [m["data"]["price"] for m in sockets[1].sent] == [42.0]
        assert len(sockets[0].sent) == 1
        assert sockets[1].sent[0]["channel"] == "market_data"
    finally:
        for relay in relays:
            await relay.stop()
    assert all(m.relay is None for m in managers)


async def test_relay_is_only_started_with_several_workers(
    monkeypatch, preserve_db_globals, tmp_path
):
    from vpp.api import app as app_module
    from vpp.api.websocket import manager

    for workers in (1, 2):
        settings = Settings(
            database_url=f"sqlite+aiosqlite:///{tmp_path / f'relay{workers}.db'}",
            api_workers=workers,
            degradation_updater_enabled=False,
            trading_market_data_enabled=False,
        )
        monkeypatch.setattr(app_module, "get_settings", lambda s=settings: s)
        fastapi_app = app_module.create_app()
        async with fastapi_app.router.lifespan_context(fastapi_app):
            assert (manager.relay is not None) == (workers > 1)
        assert manager.relay is None
