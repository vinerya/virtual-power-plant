"""OpenADR / IEEE 2030.5 / DR views with several API workers.

This process plays both roles, like the trading tests in ``test_cluster``:
the HTTP routes see an *empty* protocol registry (a worker that does not hold
the ``protocol-adapters`` lease), while the forwarded-call executor -- the
"lease holder" -- reads the process registry where the adapters live. Data
can only reach the response through ``cluster_calls``.
"""

from __future__ import annotations

import asyncio
import contextlib
import time

import pytest
from sqlalchemy import select

from vpp.api.routes import protocols as protocols_module
from vpp.api.routes.protocols import get_registry
from vpp.cluster import rpc
from vpp.cluster.lease import LeaderElector, is_local, release, try_acquire
from vpp.cluster.topology import LEASE_PROTOCOLS
from vpp.db.models import ClusterCallModel, ClusterLeaseModel
from vpp.protocols.base import ProtocolRegistry
from vpp.protocols.ieee2030_5 import DERControl, DERProgram, IEEE2030_5Adapter
from vpp.protocols.openadr import DREvent, DRSignalType, OpenADRAdapter


@pytest.fixture
async def holder_registry(monkeypatch):
    """The lease holder's registry: a simulated VEN with one event + a 2030.5 client."""
    reg = ProtocolRegistry()
    monkeypatch.setattr(protocols_module, "_registry", reg)
    oadr = OpenADRAdapter()
    oadr.configure(role="ven", poll_interval_s=0)
    await oadr.connect()
    reg.register(oadr)
    await oadr.handle_incoming_event(
        DREvent(
            event_id="EVT-CLUSTER",
            signal_type=DRSignalType.SIMPLE,
            signal_level=1,
            start_time=time.time() - 10,
            duration_seconds=600,
        )
    )
    ieee = IEEE2030_5Adapter()
    ieee.configure(poll_interval_s=0)
    await ieee.connect()
    reg.register(ieee)
    ieee.register_program(DERProgram(program_id="P", primacy=2, description="curtail"))
    await ieee.apply_control(
        "P", DERControl(control_id="C1", set_watts=5000, start_time=time.time() - 5)
    )
    return reg


async def _clear_calls(factory) -> None:
    async with factory() as s:
        await s.execute(ClusterCallModel.__table__.delete())
        await s.commit()


async def _protocol_calls(factory) -> list[ClusterCallModel]:
    async with factory() as s:
        return list(
            (
                await s.execute(
                    select(ClusterCallModel).where(ClusterCallModel.target == LEASE_PROTOCOLS)
                )
            )
            .scalars()
            .all()
        )


@pytest.fixture
async def follower_of_protocol_adapters(app, holder_registry):
    """This worker follows: another node holds the protocol-adapters lease."""
    from vpp.db.engine import get_session_factory

    factory = get_session_factory()
    await _clear_calls(factory)
    async with factory() as s:
        await s.execute(
            ClusterLeaseModel.__table__.delete().where(ClusterLeaseModel.name == LEASE_PROTOCOLS)
        )
        await s.commit()
    assert await try_acquire(factory, LEASE_PROTOCOLS, "protocol-leader-node", 120)
    elector = LeaderElector(LEASE_PROTOCOLS, factory, holder="this-worker", ttl_s=120)
    await elector.start()
    assert not is_local(LEASE_PROTOCOLS)
    # The follower's own registry has none of the lease-bound adapters.
    app.dependency_overrides[get_registry] = ProtocolRegistry
    yield factory
    app.dependency_overrides.pop(get_registry, None)
    await elector.stop()
    await release(factory, LEASE_PROTOCOLS, "protocol-leader-node")


@contextlib.asynccontextmanager
async def running_executor(factory, targets):
    task = asyncio.create_task(rpc.run_executor(factory, lambda: targets, poll_s=0.01))
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def test_follower_forwards_openadr_and_2030_5_views(
    client, auth_headers, viewer_headers, follower_of_protocol_adapters
):
    factory = follower_of_protocol_adapters
    async with running_executor(factory, [LEASE_PROTOCOLS]):  # the lease holder
        events = await client.get("/api/v1/protocols/openadr/events", headers=viewer_headers)
        assert events.status_code == 200, events.text
        body = events.json()
        assert body["mode"] == "simulated"
        [evt] = [e for e in body["events"] if e["event_id"] == "EVT-CLUSTER"]
        assert evt["active"] is True and evt["opt_type"] == "optIn"

        single = await client.get(
            "/api/v1/protocols/openadr/events/EVT-CLUSTER", headers=viewer_headers
        )
        assert single.status_code == 200, single.text
        assert single.json()["event_id"] == "EVT-CLUSTER"

        missing = await client.get("/api/v1/protocols/openadr/events/nope", headers=auth_headers)
        assert missing.status_code == 404  # the holder's answer, verbatim
        assert missing.json()["detail"] == "DR event nope not found"

        opt = await client.post(
            "/api/v1/protocols/openadr/events/EVT-CLUSTER/opt",
            json={"opt_type": "optOut", "reason": "maintenance"},
            headers=auth_headers,
        )
        assert opt.status_code == 200, opt.text
        again = await client.get(
            "/api/v1/protocols/openadr/events/EVT-CLUSTER", headers=auth_headers
        )
        assert again.json()["opt_type"] == "optOut"

        controls = await client.get("/api/v1/protocols/ieee2030_5/controls", headers=auth_headers)
        assert controls.status_code == 200, controls.text
        assert [c["control_id"] for c in controls.json()["active_controls"]] == ["C1"]
        assert controls.json()["programs"][0]["control_count"] == 1

        status = await client.get("/api/v1/dr/status", headers=auth_headers)
        assert status.status_code == 200, status.text
        assert status.json()["running"] is False

        listing = await client.get("/api/v1/protocols/", headers=auth_headers)
        assert listing.status_code == 200, listing.text
        assert {p["name"] for p in listing.json()} >= {"openadr", "ieee2030_5"}

        metrics = await client.get("/api/v1/protocols/openadr/metrics", headers=auth_headers)
        assert metrics.status_code == 200, metrics.text
        assert metrics.json()["mode"] == "simulated"

    rows = await _protocol_calls(factory)
    assert {r.method for r in rows} >= {
        "openadr_events",
        "openadr_event",
        "openadr_opt",
        "ieee2030_5_controls",
        "dr_status",
        "list_adapters",
        "metrics",
    }
    assert all(r.status in ("done", "failed") for r in rows)
    # The opt override ran on the holder, with the caller's identity.
    [opt_row] = [r for r in rows if r.method == "openadr_opt"]
    assert '"username": "testadmin"' in opt_row.payload_json


async def test_follower_without_a_live_holder_answers_503(
    client, auth_headers, follower_of_protocol_adapters, monkeypatch
):
    from vpp.settings import get_settings

    monkeypatch.setattr(get_settings(), "cluster_call_timeout_seconds", 0.3)
    for path in (
        "/api/v1/protocols/openadr/events",
        "/api/v1/protocols/openadr/events/EVT-CLUSTER",
        "/api/v1/protocols/ieee2030_5/controls",
        "/api/v1/dr/status",
        "/api/v1/protocols/",
    ):
        resp = await client.get(path, headers=auth_headers)
        assert resp.status_code == 503, (path, resp.text)
        assert resp.json()["detail"]["code"] == "leader_unavailable"
        assert resp.headers.get("retry-after") == "5"
    rows = await _protocol_calls(follower_of_protocol_adapters)
    assert len(rows) == 5 and all(r.status == "cancelled" for r in rows)


async def test_forwarded_reads_use_a_short_timeout(
    client, auth_headers, follower_of_protocol_adapters, monkeypatch
):
    """Reads give up after PROTOCOL_READ_TIMEOUT_S even with a long cluster timeout."""
    from vpp.api.routes import protocol_ops
    from vpp.settings import get_settings

    assert get_settings().cluster_call_timeout_seconds > protocol_ops.PROTOCOL_READ_TIMEOUT_S
    monkeypatch.setattr(protocol_ops, "PROTOCOL_READ_TIMEOUT_S", 0.2)
    started = time.monotonic()
    resp = await client.get("/api/v1/protocols/openadr/events", headers=auth_headers)
    assert resp.status_code == 503
    assert time.monotonic() - started < 2.0


async def test_single_worker_reads_locally_without_forwarding(
    app, client, auth_headers, holder_registry
):
    """No elector (one process) or holding the lease: direct call, no cluster_calls row."""
    from vpp.db.engine import get_session_factory

    factory = get_session_factory()
    await _clear_calls(factory)
    assert is_local(LEASE_PROTOCOLS)
    for path in (
        "/api/v1/protocols/openadr/events",
        "/api/v1/protocols/ieee2030_5/controls",
        "/api/v1/dr/status",
        "/api/v1/protocols/",
        "/api/v1/protocols/openadr/metrics",
    ):
        resp = await client.get(path, headers=auth_headers)
        assert resp.status_code == 200, (path, resp.text)

    # Holding the lease behaves the same.
    async with factory() as s:
        await s.execute(
            ClusterLeaseModel.__table__.delete().where(ClusterLeaseModel.name == LEASE_PROTOCOLS)
        )
        await s.commit()
    elector = LeaderElector(LEASE_PROTOCOLS, factory, holder="this-worker", ttl_s=120)
    await elector.start()
    try:
        assert elector.is_leader
        resp = await client.get("/api/v1/protocols/openadr/events", headers=auth_headers)
        assert resp.status_code == 200
        assert [e["event_id"] for e in resp.json()["events"]] == ["EVT-CLUSTER"]
    finally:
        await elector.stop()
    assert await _protocol_calls(factory) == []


async def test_non_lease_adapters_are_not_forwarded(
    client, auth_headers, follower_of_protocol_adapters
):
    """Only OCPP / OpenADR / IEEE 2030.5 belong to the holder; other names stay local."""
    resp = await client.get("/api/v1/protocols/mqtt/metrics", headers=auth_headers)
    assert resp.status_code == 404
    assert await _protocol_calls(follower_of_protocol_adapters) == []
