"""Dispatch -> device setpoints: the optimization API (apply=true), the DR
orchestrator (OpenADR / IEEE 2030.5 incl. DefaultDERControl) and release."""

from __future__ import annotations

import json
import time
import uuid

import pytest
from _v2g_helpers import tmp_session_factory
from sqlalchemy import select
from test_device_control import Clock, FakeWriter, ModbusServer
from test_dr_orchestrator import NOW, _event

from vpp.control.actuator import SetpointActuator, set_setpoint_actuator
from vpp.db.models import DREventResponseModel, ResourceModel
from vpp.dr.orchestrator import DROrchestrator
from vpp.dr.translate import DRPolicy
from vpp.events import EventType, get_event_bus
from vpp.protocols.base import ProtocolRegistry
from vpp.protocols.ieee2030_5 import DERControl, DERProgram, EventStatusCode, IEEE2030_5Adapter
from vpp.protocols.openadr import OpenADRAdapter

CONTROL = {"host": "10.0.0.9", "control": {"enabled": True, "address": 40, "unit": "W"}}

# ---------------------------------------------------------------------------
# POST /api/v1/optimization/dispatch?apply
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_actuator():
    writers: dict[str, FakeWriter] = {}

    def factory(rid, modbus, control, rated):
        return writers.setdefault(rid, FakeWriter())

    actuator = SetpointActuator(enabled=True, writer_factory=factory)
    set_setpoint_actuator(actuator)
    yield actuator, writers
    set_setpoint_actuator(None)


async def _resource(client, headers, rtype="battery", rated=50.0, **meta) -> str:
    resp = await client.post(
        "/api/v1/resources/",
        json={
            "name": f"act-{rtype}-{uuid.uuid4().hex[:8]}",
            "resource_type": rtype,
            "rated_power": rated,
            "metadata": {"capacity_kwh": 400, "soc": 0.5, **meta},
        },
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


async def test_dispatch_plans_only_by_default(client, auth_headers, fake_actuator):
    _, writers = fake_actuator
    b = await _resource(client, auth_headers, modbus=CONTROL)
    body = (
        await client.post(
            "/api/v1/optimization/dispatch",
            json={"target_power_kw": 20.0, "resource_ids": [b]},
            headers=auth_headers,
        )
    ).json()
    assert body["success"] is True
    assert body["applied"] is False and body["device_deliveries"] == []
    assert writers == {}


async def test_dispatch_apply_writes_setpoints_and_records(client, auth_headers, fake_actuator):
    actuator, writers = fake_actuator
    controlled = await _resource(client, auth_headers, rated=50.0, modbus=CONTROL)
    plain = await _resource(client, auth_headers, rated=50.0)
    resp = await client.post(
        "/api/v1/optimization/dispatch",
        json={
            "target_power_kw": 60.0,
            "resource_ids": [controlled, plain],
            "apply": True,
            "interval_minutes": 5,
        },
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["applied"] is True
    alloc = {a["resource_id"]: a["allocated_power_kw"] for a in body["allocations"]}
    by_id = {d["resource_id"]: d for d in body["device_deliveries"]}
    assert by_id[plain]["status"] == "not_configured"
    assert by_id[controlled]["status"] == "accepted"
    assert by_id[controlled]["setpoint_kw"] == pytest.approx(alloc[controlled])
    assert writers[controlled].calls == [("write", pytest.approx(alloc[controlled]))]
    [active] = actuator.status()["active"]
    assert active["run_id"] == body["run_id"] and active["source"] == "api.optimization.dispatch"

    run = (
        await client.get(f"/api/v1/optimization/runs/{body['run_id']}", headers=auth_headers)
    ).json()
    assert run["inputs"]["apply"] is True
    assert {d["status"] for d in run["metadata"]["device_deliveries"]} == {
        "accepted",
        "not_configured",
    }

    status = (
        await client.get(
            "/api/v1/optimization/setpoints",
            params={"resource_id": controlled},
            headers=auth_headers,
        )
    ).json()
    assert status["enabled"] is True
    assert [r["status"] for r in status["recent"]] == ["accepted"]
    assert status["recent"][0]["run_id"] == body["run_id"]


async def test_viewer_may_plan_but_not_apply(client, auth_headers, viewer_headers, fake_actuator):
    _, writers = fake_actuator
    b = await _resource(client, auth_headers, modbus=CONTROL)
    payload = {"target_power_kw": 5.0, "resource_ids": [b]}
    ok = await client.post("/api/v1/optimization/dispatch", json=payload, headers=viewer_headers)
    assert ok.status_code == 200
    denied = await client.post(
        "/api/v1/optimization/dispatch", json={**payload, "apply": True}, headers=viewer_headers
    )
    assert denied.status_code == 403
    assert writers == {}


async def test_dispatch_apply_with_kill_switch_off_reports_disabled(client, auth_headers):
    set_setpoint_actuator(SetpointActuator(enabled=False))
    try:
        b = await _resource(client, auth_headers, modbus=CONTROL)
        body = (
            await client.post(
                "/api/v1/optimization/dispatch",
                json={"target_power_kw": 10.0, "resource_ids": [b], "apply": True},
                headers=auth_headers,
            )
        ).json()
        [d] = body["device_deliveries"]
        assert d["status"] == "disabled" and "VPP_CONTROL_ENABLED" in d["reason"]
    finally:
        set_setpoint_actuator(None)


# ---------------------------------------------------------------------------
# DR orchestrator -> real Modbus device
# ---------------------------------------------------------------------------


async def _seed(factory, *, rated=50.0, meta=None, name="bess") -> str:
    async with factory() as session:
        row = ResourceModel(
            name=name,
            resource_type="battery",
            rated_power=rated,
            online=True,
            nominal_energy_kwh=400.0,
            config_json=json.dumps({"state_of_charge": 0.5}),
            metadata_json=json.dumps(meta or {}),
        )
        session.add(row)
        await session.commit()
        return row.id


async def _orchestrator(tmp_path, clock, *, enabled=True, policy=None):
    factory = tmp_session_factory(tmp_path / "act.db")
    registry = ProtocolRegistry()
    openadr = OpenADRAdapter()
    openadr.configure(role="ven", poll_interval_s=0)
    await openadr.connect()
    registry.register(openadr)
    actuator = SetpointActuator(enabled=enabled, session_factory=factory, clock=clock)
    orch = DROrchestrator(
        policy or DRPolicy(auto_response=True, redispatch_interval_s=600),
        registry,
        session_factory=factory,
        clock=clock,
        actuator=actuator,
    )
    orch.attach_openadr(openadr)
    return factory, registry, openadr, orch, actuator


@pytest.fixture
async def device():
    server = await ModbusServer().start()
    yield server
    await server.stop()


def _modbus(port: int) -> dict:
    return {
        "mode": "tcp",
        "host": "127.0.0.1",
        "port": port,
        "device_profile": "none",
        # 0.1 kW per count, signed: -3276.8 .. 3276.7 kW
        "control": {
            "enabled": True,
            "address": 50,
            "data_type": "int16",
            "unit": "kW",
            "scale": 0.1,
            "min_interval_s": 0,
        },
    }


async def test_dr_event_drives_device_and_release_hands_control_back(tmp_path, device):
    clock = Clock(NOW)
    events: list = []

    async def on_event(e):
        events.append(e)

    sub = get_event_bus().subscribe(
        on_event, event_types={EventType.DISPATCH_EXECUTED, EventType.DEVICE_SETPOINT}
    )
    factory, _, openadr, orch, actuator = await _orchestrator(tmp_path, clock)
    try:
        battery = await _seed(factory, meta={"modbus": _modbus(device.port)})
        await openadr.handle_incoming_event(_event(level=30, start=NOW - 60, duration=1800))

        summary = await orch.tick()
        assert summary["action"] == "dispatched"
        assert summary["device_deliveries"] == {"accepted": 1}
        assert await device.read(50) == [300]  # 30.0 kW at 0.1 kW/count
        [row] = [
            r
            for r in (await _session_rows(factory))
            if r.action == "dispatched" and r.source_id == "E1"
        ]
        [d] = json.loads(row.details_json)["device_deliveries"]
        assert d["resource_id"] == battery and d["verified"] is True
        assert orch.status()["device_control_enabled"] is True

        clock.t = NOW + 1800  # event over
        released = await orch.tick()
        assert released["action"] == "released"
        assert released["device_releases"] == {"accepted": 1}
        assert await device.read(50) == [0]
        assert actuator.status()["active"] == []
        kinds = [(e.event_type, e.data.get("source")) for e in events]
        assert (EventType.DEVICE_SETPOINT, "dr") in kinds
    finally:
        get_event_bus().unsubscribe(sub)
        await actuator.shutdown()


async def test_dr_dispatch_with_kill_switch_off_writes_nothing(tmp_path, device):
    clock = Clock(NOW)
    factory, _, openadr, orch, _act = await _orchestrator(tmp_path, clock, enabled=False)
    await _seed(factory, meta={"modbus": _modbus(device.port)})
    await device._probe.write_registers(50, [123])
    await openadr.handle_incoming_event(_event(level=30, start=NOW - 60))
    summary = await orch.tick()
    assert summary["device_deliveries"] == {"disabled": 1}
    assert await device.read(50) == [123]  # untouched


async def test_dr_setpoint_expires_via_watchdog_when_orchestrator_stops(tmp_path, device):
    clock = Clock(NOW)
    factory, _, openadr, orch, actuator = await _orchestrator(tmp_path, clock)
    await _seed(factory, meta={"modbus": _modbus(device.port)})
    await openadr.handle_incoming_event(_event(level=20, start=NOW - 60, duration=3600))
    await orch.tick()
    assert await device.read(50) == [200]
    # The orchestrator stops ticking (crash / hang); the watchdog falls back.
    clock.t += 600 + actuator.expiry_grace_s
    [d] = await actuator.tick()
    assert d["action"] == "expire" and d["status"] == "accepted"
    assert await device.read(50) == [0]
    await actuator.shutdown()


async def _session_rows(factory):
    async with factory() as s:
        return list((await s.execute(select(DREventResponseModel))).scalars().all())


# ---------------------------------------------------------------------------
# IEEE 2030.5 DefaultDERControl
# ---------------------------------------------------------------------------


async def _with_ieee(registry, program: DERProgram) -> IEEE2030_5Adapter:
    ieee = IEEE2030_5Adapter()
    ieee.configure(poll_interval_s=0)
    await ieee.connect()  # simulated
    registry.register(ieee)
    ieee.register_program(program)
    return ieee


async def test_default_der_control_applies_until_an_event_control_is_active(tmp_path):
    clock = Clock(time.time())
    factory, registry, _, orch, _act = await _orchestrator(tmp_path, clock)
    await _seed(factory, rated=80.0)
    default = DERControl(control_id="DEF", set_watts=10_000, start_time=0.0, duration_seconds=0)
    ieee = await _with_ieee(registry, DERProgram(program_id="P1", default_control=default))

    summary = await orch.tick()
    assert summary["protocol"] == "ieee2030_5" and summary["target_kw"] == pytest.approx(10.0)
    assert summary["sources"][0]["default"] is True
    assert summary["reason"].startswith("DefaultDERControl")
    assert await orch.tick() is None  # standing default: no churn

    event = DERControl(
        control_id="EVT",
        set_watts=-5_000,
        start_time=clock.t - 10,
        duration_seconds=900,
        program_id="P1",
    )
    await ieee.apply_control("P1", event)
    summary = await orch.tick()
    assert summary["source_id"] == "EVT" and summary["target_kw"] == pytest.approx(-5.0)

    event.event_status = EventStatusCode.CANCELLED  # event gone -> default again
    summary = await orch.tick()
    assert summary["source_id"] == "DEF" and summary["target_kw"] == pytest.approx(10.0)


async def test_openadr_event_beats_default_target_but_default_limits_clamp(tmp_path):
    clock = Clock(NOW)
    factory, registry, openadr, orch, _ = await _orchestrator(tmp_path, clock)
    await _seed(factory, rated=80.0)
    default = DERControl(control_id="DEF", set_watts=10_000, gen_limit_w=20_000)
    await _with_ieee(registry, DERProgram(program_id="P1", default_control=default))
    await openadr.handle_incoming_event(_event(level=30, start=NOW - 60))
    summary = await orch.tick()
    assert summary["protocol"] == "openadr"
    assert summary["target_kw"] == pytest.approx(20.0)  # 30 kW clamped by opModGenLimW
    assert "clamped to IEEE 2030.5 export limit" in summary["reason"]
