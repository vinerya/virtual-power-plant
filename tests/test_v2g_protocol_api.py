"""V2G persistence + protocol data API on the full app (real auth, real DB)."""

from __future__ import annotations

import time

import pytest
from _portal_helpers import uid, user_headers
from sqlalchemy import select

from vpp.db.models import DREventResponseModel, V2GVehicleModel
from vpp.protocols.base import ProtocolRegistry
from vpp.protocols.ieee2030_5 import DERControl, DERProgram, IEEE2030_5Adapter
from vpp.protocols.ocpp import ChargePoint, OCPPAdapter
from vpp.protocols.openadr import DREvent, DRSignalType, OpenADRAdapter
from vpp.v2g.store import V2GRepository, load_fleet


@pytest.fixture
def registry(monkeypatch):
    from vpp.api.routes import protocols as protocols_module

    reg = ProtocolRegistry()
    monkeypatch.setattr(protocols_module, "_registry", reg)
    return reg


async def _create(client, headers, **fields):
    body = {"ev_id": uid("ev"), **fields}
    resp = await client.post("/api/v1/v2g/vehicles", json=body, headers=headers)
    assert resp.status_code == 201, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# V2G persistence
# ---------------------------------------------------------------------------


async def test_vehicle_crud_is_persisted(client, auth_headers, db_session):
    departure = time.time() + 8 * 3600
    ev = await _create(
        client, auth_headers, capacity_kwh=75, current_soc=0.4, departure_time=departure
    )
    ev_id = ev["ev_id"]
    assert ev["connection_state"] == "connected_idle"  # legacy default
    assert ev["departure_time"] == pytest.approx(departure, abs=1)
    assert ev["soc_source"] == "api"

    row = await db_session.get(V2GVehicleModel, ev_id)
    assert row is not None and row.capacity_kwh == 75
    fleet = await load_fleet(db_session, [ev_id])  # what a fresh process would see
    assert fleet.get_vehicle(ev_id).current_soc == pytest.approx(0.4)

    dup = await client.post("/api/v1/v2g/vehicles", json={"ev_id": ev_id}, headers=auth_headers)
    assert dup.status_code == 409

    patched = await client.patch(
        f"/api/v1/v2g/vehicles/{ev_id}",
        json={"target_soc": 0.95, "current_soc": 0.6, "clear_departure_time": True},
        headers=auth_headers,
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["target_soc"] == 0.95
    assert patched.json()["departure_time"] is None
    assert patched.json()["soc_updated_at"] is not None

    listed = await client.get("/api/v1/v2g/vehicles", headers=auth_headers)
    assert ev_id in {v["ev_id"] for v in listed.json()}
    assert (
        await client.delete(f"/api/v1/v2g/vehicles/{ev_id}", headers=auth_headers)
    ).status_code == 204
    assert (
        await client.get(f"/api/v1/v2g/vehicles/{ev_id}", headers=auth_headers)
    ).status_code == 404
    assert (
        await client.delete(f"/api/v1/v2g/vehicles/{ev_id}", headers=auth_headers)
    ).status_code == 204


async def test_v2g_rbac(client, auth_headers, viewer_headers, db_session):
    ev = await _create(client, auth_headers)
    assert (await client.get("/api/v1/v2g/vehicles", headers=viewer_headers)).status_code == 200
    assert (await client.get("/api/v1/v2g/fleet", headers=viewer_headers)).status_code == 200
    denied = await client.post("/api/v1/v2g/vehicles", json={}, headers=viewer_headers)
    assert denied.status_code == 403
    denied = await client.put(
        f"/api/v1/v2g/vehicles/{ev['ev_id']}/binding",
        json={"charge_point_id": "CP"},
        headers=viewer_headers,
    )
    assert denied.status_code == 403
    operator = await user_headers(db_session, "operator")
    assert (
        await client.post("/api/v1/v2g/vehicles", json={}, headers=operator)
    ).status_code == 201
    customer = await user_headers(db_session, "customer")
    assert (await client.get("/api/v1/v2g/vehicles", headers=customer)).status_code == 403
    assert (await client.get("/api/v1/v2g/vehicles")).status_code == 401


async def test_binding_rules(client, auth_headers, registry):
    cp_id = uid("CP")
    a = await _create(client, auth_headers)
    b = await _create(client, auth_headers, id_tag=uid("tag"))

    partial = await client.post(
        "/api/v1/v2g/vehicles", json={"charge_point_id": cp_id}, headers=auth_headers
    )
    assert partial.status_code == 422

    bound = await client.put(
        f"/api/v1/v2g/vehicles/{a['ev_id']}/binding",
        json={"charge_point_id": cp_id, "connector_id": 1},
        headers=auth_headers,
    )
    assert bound.status_code == 200
    assert bound.json()["binding_source"] == "manual"
    conflict = await client.put(
        f"/api/v1/v2g/vehicles/{b['ev_id']}/binding",
        json={"charge_point_id": cp_id, "connector_id": 1},
        headers=auth_headers,
    )
    assert conflict.status_code == 409
    unbound = await client.delete(
        f"/api/v1/v2g/vehicles/{a['ev_id']}/binding", headers=auth_headers
    )
    assert unbound.json()["charge_point_id"] is None
    ok = await client.put(
        f"/api/v1/v2g/vehicles/{b['ev_id']}/binding",
        json={"charge_point_id": cp_id, "connector_id": 1},
        headers=auth_headers,
    )
    assert ok.status_code == 200

    same_tag = await client.patch(
        f"/api/v1/v2g/vehicles/{a['ev_id']}", json={"id_tag": b["id_tag"]}, headers=auth_headers
    )
    assert same_tag.status_code == 409


async def test_schedule_reports_why_nothing_was_pushed(client, auth_headers, registry, db_session):
    unbound = await _create(client, auth_headers, target_soc=0.9)
    bound = await _create(client, auth_headers, target_soc=0.9)
    await client.put(
        f"/api/v1/v2g/vehicles/{bound['ev_id']}/binding",
        json={"charge_point_id": uid("CP"), "connector_id": 1},
        headers=auth_headers,
    )
    resp = await client.post(
        "/api/v1/v2g/schedule",
        json={"ev_ids": [unbound["ev_id"], bound["ev_id"], "ghost"]},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    by_ev = {d["ev_id"]: d["status"] for d in body["deliveries"]}
    assert by_ev == {unbound["ev_id"]: "not_bound", bound["ev_id"]: "no_ocpp"}
    assert body["missing_ev_ids"] == ["ghost"]
    assert body["schedule"]  # legacy fields still there
    [record] = [
        r for r in await V2GRepository.list_schedules(db_session) if r.id == body["schedule_id"]
    ]
    assert record.kind == "schedule"


# ---------------------------------------------------------------------------
# Protocol data API
# ---------------------------------------------------------------------------


async def test_protocol_reads_404_when_not_running(client, auth_headers, registry):
    for path in (
        "/api/v1/protocols/ocpp/charge-points",
        "/api/v1/protocols/openadr/events",
        "/api/v1/protocols/ieee2030_5/controls",
    ):
        assert (await client.get(path, headers=auth_headers)).status_code == 404


async def test_simulated_ocpp_reads_and_actions(
    client, auth_headers, viewer_headers, registry, db_session
):
    ocpp = OCPPAdapter()
    await ocpp.connect()
    ocpp.register_charge_point(ChargePoint(charge_point_id="SIM-CP"))
    registry.register(ocpp)

    listing = await client.get("/api/v1/protocols/ocpp/charge-points", headers=viewer_headers)
    assert listing.status_code == 200
    assert listing.json()["mode"] == "simulated" and listing.json()["simulated"] is True
    assert listing.json()["charge_points"][0]["connected"] is False

    denied = await client.post(
        "/api/v1/protocols/ocpp/charge-points/SIM-CP/remote-start", headers=viewer_headers
    )
    assert denied.status_code == 403
    customer = await user_headers(db_session, "customer")
    assert (
        await client.get("/api/v1/protocols/ocpp/charge-points", headers=customer)
    ).status_code == 403

    started = await client.post(
        "/api/v1/protocols/ocpp/charge-points/SIM-CP/remote-start", headers=auth_headers
    )
    assert started.json()["accepted"] is True
    assert started.json()["live"] is False  # simulated: nothing left the process
    stopped = await client.post(
        "/api/v1/protocols/ocpp/charge-points/SIM-CP/remote-stop", headers=auth_headers
    )
    assert stopped.json()["accepted"] is True


async def test_openadr_events_and_operator_opt_override(
    client, auth_headers, viewer_headers, registry, db_session
):
    oadr = OpenADRAdapter()
    oadr.configure(role="ven", poll_interval_s=0)
    await oadr.connect()
    registry.register(oadr)
    event_id = uid("EVT")
    now = time.time()
    await oadr.handle_incoming_event(
        DREvent(
            event_id=event_id,
            signal_type=DRSignalType.SIMPLE,
            signal_level=1,
            start_time=now - 10,
            duration_seconds=600,
        )
    )
    events = await client.get("/api/v1/protocols/openadr/events", headers=viewer_headers)
    assert events.status_code == 200
    [evt] = [e for e in events.json()["events"] if e["event_id"] == event_id]
    assert evt["active"] is True
    assert evt["opt_type"] == "optIn"
    assert evt["source"] == "local"
    assert events.json()["mode"] == "simulated"

    denied = await client.post(
        f"/api/v1/protocols/openadr/events/{event_id}/opt",
        json={"opt_type": "optOut"},
        headers=viewer_headers,
    )
    assert denied.status_code == 403
    bad = await client.post(
        f"/api/v1/protocols/openadr/events/{event_id}/opt",
        json={"opt_type": "maybe"},
        headers=auth_headers,
    )
    assert bad.status_code == 422
    missing = await client.post(
        "/api/v1/protocols/openadr/events/nope/opt",
        json={"opt_type": "optOut"},
        headers=auth_headers,
    )
    assert missing.status_code == 404

    resp = await client.post(
        f"/api/v1/protocols/openadr/events/{event_id}/opt",
        json={"opt_type": "optOut", "reason": "site maintenance"},
        headers=auth_headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["sent_to_vtn"] is False
    single = await client.get(f"/api/v1/protocols/openadr/events/{event_id}", headers=auth_headers)
    assert single.json()["opt_type"] == "optOut"

    audit = await client.get(
        "/api/v1/dr/responses", params={"source_id": event_id}, headers=viewer_headers
    )
    [row] = audit.json()
    assert row["action"] == "opt_override" and row["opt_type"] == "optOut"
    assert row["reason"] == "site maintenance"
    stored = (
        await db_session.execute(
            select(DREventResponseModel).where(DREventResponseModel.source_id == event_id)
        )
    ).scalar_one()
    assert stored.protocol == "openadr"


async def test_ieee2030_5_controls_and_dr_status(client, auth_headers, registry):
    ieee = IEEE2030_5Adapter()
    ieee.configure(poll_interval_s=0)
    await ieee.connect()
    registry.register(ieee)
    ieee.register_program(DERProgram(program_id="P", primacy=2, description="curtail"))
    await ieee.apply_control(
        "P", DERControl(control_id="C1", set_watts=5000, start_time=time.time() - 5)
    )
    resp = await client.get("/api/v1/protocols/ieee2030_5/controls", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "simulated"
    assert [c["control_id"] for c in body["active_controls"]] == ["C1"]
    assert body["programs"][0]["control_count"] == 1

    status = await client.get("/api/v1/dr/status", headers=auth_headers)
    assert status.status_code == 200
    assert status.json()["running"] is False
    assert status.json()["auto_response_enabled"] is False  # safe default
