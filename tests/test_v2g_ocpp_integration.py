# ruff: noqa: SIM117  (nested TestClient/websocket contexts read clearer)
"""V2G fleet <-> OCPP Central System integration.

A simulated charge point talks OCPP-J to the real ``/ocpp/{id}`` route while
the V2G API runs against an isolated SQLite database, so the whole loop is
exercised: idTag auto-binding, MeterValues -> SOC, StatusNotification ->
plug state, schedule -> SetChargingProfile with per-vehicle results.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest
from _v2g_helpers import (
    SUBPROTOCOL,
    ChargePointSim,
    build_app,
    live_ocpp_adapter,
    tmp_session_factory,
)
from starlette.testclient import TestClient

from vpp.events import EventType, get_event_bus
from vpp.protocols.base import ProtocolRegistry
from vpp.protocols.ocpp import ChargePoint, OCPPAdapter
from vpp.v2g.ocpp_bridge import OCPPVehicleBridge, compact_slots


@pytest.fixture
def setup(tmp_path):
    factory = tmp_session_factory(tmp_path / "v2g.db")
    adapter = live_ocpp_adapter()
    OCPPVehicleBridge(adapter, factory).attach()
    registry = ProtocolRegistry()
    registry.register(adapter)
    return build_app(factory, registry), adapter, factory


def _vehicle(client: TestClient, **fields) -> dict:
    body = {"ev_id": "ev-1", "capacity_kwh": 60, "current_soc": 0.5, **fields}
    resp = client.post("/api/v1/v2g/vehicles", json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()


def test_id_tag_auto_binding_meter_values_and_unplug(setup):
    app, _, _ = setup
    events: list = []

    async def on_event(event):
        events.append(event)

    sub = get_event_bus().subscribe(
        on_event,
        event_types={
            EventType.EV_CONNECTED,
            EventType.EV_DISCONNECTED,
            EventType.RESOURCE_UPDATED,
        },
    )
    try:
        with TestClient(app) as client:
            _vehicle(client, id_tag="TAG-EV1", connection_state="disconnected")
            with client.websocket_connect("/ocpp/CP1", subprotocols=SUBPROTOCOL) as ws:
                cp = ChargePointSim(ws)
                cp.boot()
                cp.status(2, "Preparing")  # nobody bound to connector 2 yet: ignored
                assert client.get("/api/v1/v2g/vehicles/ev-1").json()["charge_point_id"] is None

                tx = cp.start(2, "TAG-EV1", meter_start=1000)
                ev = client.get("/api/v1/v2g/vehicles/ev-1").json()
                assert ev["charge_point_id"] == "CP1"
                assert ev["connector_id"] == 2
                assert ev["binding_source"] == "id_tag"
                assert ev["active_transaction_id"] == tx
                assert ev["connection_state"] == "connected_idle"

                cp.status(2, "Charging")
                cp.meter(2, tx, power_w=7400, soc=63)
                ev = client.get("/api/v1/v2g/vehicles/ev-1").json()
                assert ev["current_soc"] == pytest.approx(0.63)
                assert ev["soc_source"] == "ocpp"
                assert ev["current_power_kw"] == pytest.approx(7.4)
                assert ev["connection_state"] == "charging"
                assert ev["charger_status"] == "Charging"

                # V2G discharge reported as export power
                cp.call(
                    "MeterValues",
                    {
                        "connectorId": 2,
                        "transactionId": tx,
                        "meterValue": [
                            {
                                "timestamp": "2026-09-28T10:10:00Z",
                                "sampledValue": [
                                    {
                                        "value": "5000",
                                        "measurand": "Power.Active.Export",
                                        "unit": "W",
                                    }
                                ],
                            }
                        ],
                    },
                )
                ev = client.get("/api/v1/v2g/vehicles/ev-1").json()
                assert ev["connection_state"] == "discharging"
                assert ev["current_power_kw"] == pytest.approx(-5.0)

                cp.stop(tx, meter_stop=9000)
                ev = client.get("/api/v1/v2g/vehicles/ev-1").json()
                assert ev["active_transaction_id"] is None
                assert ev["connection_state"] == "connected_idle"

                sessions = client.get("/api/v1/v2g/sessions", params={"ev_id": "ev-1"}).json()
                assert len(sessions) == 1
                assert sessions[0]["transaction_id"] == tx
                assert sessions[0]["status"] == "completed"
                assert sessions[0]["energy_kwh"] == pytest.approx(8.0)
                assert sessions[0]["id_tag"] == "TAG-EV1"

                cp.status(2, "Available")  # unplugged: auto binding released
                ev = client.get("/api/v1/v2g/vehicles/ev-1").json()
                assert ev["connection_state"] == "disconnected"
                assert ev["charge_point_id"] is None
                assert ev["binding_source"] is None
    finally:
        get_event_bus().unsubscribe(sub)

    types = [e.event_type for e in events if e.source == "v2g.ocpp"]
    assert EventType.EV_CONNECTED in types
    assert EventType.EV_DISCONNECTED in types
    soc_updates = [
        e
        for e in events
        if e.event_type == EventType.RESOURCE_UPDATED and e.data.get("resource_id") == "ev:ev-1"
    ]
    assert soc_updates and soc_updates[0].data["soc"] == pytest.approx(0.63)
    assert soc_updates[0].data["resource_type"] == "ev"


def test_manual_binding_picks_up_transaction_and_survives_unplug(setup):
    app, _, _ = setup
    with TestClient(app) as client:
        _vehicle(client, charge_point_id="CP1", connector_id=1)
        with client.websocket_connect("/ocpp/CP1", subprotocols=SUBPROTOCOL) as ws:
            cp = ChargePointSim(ws)
            cp.boot()
            tx = cp.start(1, "SOMEONE-ELSE")
            ev = client.get("/api/v1/v2g/vehicles/ev-1").json()
            assert ev["active_transaction_id"] == tx
            assert ev["binding_source"] == "manual"
            cp.meter(1, None, power_w=3000, soc=55)  # no transactionId: matched by connector
            assert client.get("/api/v1/v2g/vehicles/ev-1").json()["current_soc"] == pytest.approx(
                0.55
            )
            cp.stop(tx, 500)
            cp.status(1, "Available")
            ev = client.get("/api/v1/v2g/vehicles/ev-1").json()
            assert ev["connection_state"] == "disconnected"
            assert ev["charge_point_id"] == "CP1"  # manual bindings stay


def test_idtag_rebinds_and_releases_previous_occupant(setup):
    app, _, _ = setup
    with TestClient(app) as client:
        _vehicle(client, ev_id="ev-a", charge_point_id="CP1", connector_id=1)
        _vehicle(client, ev_id="ev-b", id_tag="TAG-B")
        with client.websocket_connect("/ocpp/CP1", subprotocols=SUBPROTOCOL) as ws:
            cp = ChargePointSim(ws)
            cp.boot()
            cp.start(1, "TAG-B")
        a = client.get("/api/v1/v2g/vehicles/ev-a").json()
        b = client.get("/api/v1/v2g/vehicles/ev-b").json()
        assert a["charge_point_id"] is None
        assert a["connection_state"] == "disconnected"
        assert (b["charge_point_id"], b["connector_id"]) == ("CP1", 1)


def test_rejected_id_tag_does_not_bind(tmp_path):
    factory = tmp_session_factory(tmp_path / "v2g.db")
    adapter = live_ocpp_adapter(authorized_id_tags=["GOOD"])
    OCPPVehicleBridge(adapter, factory).attach()
    registry = ProtocolRegistry()
    registry.register(adapter)
    app = build_app(factory, registry)
    with TestClient(app) as client:
        _vehicle(client, id_tag="BAD")
        with client.websocket_connect("/ocpp/CP1", subprotocols=SUBPROTOCOL) as ws:
            cp = ChargePointSim(ws)
            cp.boot()
            cp.start(1, "BAD")
        assert client.get("/api/v1/v2g/vehicles/ev-1").json()["charge_point_id"] is None
        assert client.get("/api/v1/v2g/sessions").json() == []


def _post_in_thread(client: TestClient, path: str, body: dict):
    pool = ThreadPoolExecutor(max_workers=1)
    return pool, pool.submit(client.post, path, json=body)


def test_schedule_pushes_set_charging_profile_and_reports_results(setup):
    app, adapter, _ = setup
    with TestClient(app) as client:
        _vehicle(client, ev_id="ev-live", id_tag="TAG-LIVE", target_soc=0.9)
        _vehicle(client, ev_id="ev-reject", id_tag="TAG-REJ", target_soc=0.9)
        _vehicle(client, ev_id="ev-offline", charge_point_id="CP-GONE", connector_id=1)
        _vehicle(client, ev_id="ev-unbound")
        with client.websocket_connect("/ocpp/CP1", subprotocols=SUBPROTOCOL) as ws1:
            cp1 = ChargePointSim(ws1)
            cp1.boot()
            tx1 = cp1.start(1, "TAG-LIVE")
            with client.websocket_connect("/ocpp/CP2", subprotocols=SUBPROTOCOL) as ws2:
                cp2 = ChargePointSim(ws2)
                cp2.boot()
                cp2.start(1, "TAG-REJ")

                prices = [0.05] * 32 + [0.40] * 16 + [0.10] * 48
                pool, fut = _post_in_thread(
                    client, "/api/v1/v2g/schedule", {"prices": prices, "time_horizon_hours": 24}
                )
                uid1, p1 = cp1.expect_call("SetChargingProfile")
                uid2, _p2 = cp2.expect_call("SetChargingProfile")
                prof = p1["csChargingProfiles"]
                assert p1["connectorId"] == 1
                assert prof["chargingProfilePurpose"] == "TxProfile"
                assert prof["transactionId"] == tx1
                assert prof["chargingProfileId"] == 200
                periods = prof["chargingSchedule"]["chargingSchedulePeriod"]
                assert periods[0]["startPeriod"] == 0
                assert periods[0]["limit"] > 0  # charges first (cheap slots)
                assert len(periods) <= 48
                cp1.reply(uid1, {"status": "Accepted"})
                cp2.reply(uid2, {"status": "Rejected"})
                resp = fut.result(timeout=10)
                pool.shutdown()

    assert resp.status_code == 200, resp.text
    body = resp.json()
    by_ev = {d["ev_id"]: d for d in body["deliveries"]}
    assert by_ev["ev-live"]["status"] == "accepted"
    assert by_ev["ev-live"]["charge_point_id"] == "CP1"
    assert by_ev["ev-reject"]["status"] == "rejected"
    assert by_ev["ev-offline"]["status"] == "not_connected"
    assert by_ev["ev-unbound"]["status"] == "not_bound"
    assert body["delivery_summary"]["accepted"] == 1
    assert body["schedule_id"]
    assert adapter.get_charge_point("CP1").active_profile.profile_id == 200


def test_schedule_without_transaction_uses_tx_default_profile(setup):
    app, _, _ = setup
    with TestClient(app) as client:
        _vehicle(client, charge_point_id="CP1", connector_id=1, target_soc=0.9)
        with client.websocket_connect("/ocpp/CP1", subprotocols=SUBPROTOCOL) as ws:
            cp = ChargePointSim(ws)
            cp.boot()
            pool, fut = _post_in_thread(client, "/api/v1/v2g/schedule", {})
            uid, payload = cp.expect_call("SetChargingProfile")
            prof = payload["csChargingProfiles"]
            assert prof["chargingProfilePurpose"] == "TxDefaultProfile"
            assert "transactionId" not in prof
            cp.reply(uid, {"status": "NotSupported"})
            resp = fut.result(timeout=10)
            pool.shutdown()
    [delivery] = resp.json()["deliveries"]
    assert delivery["status"] == "not_supported"


def test_dispatch_pushes_setpoints(setup):
    app, _, _ = setup
    with TestClient(app) as client:
        _vehicle(client, id_tag="T1", current_soc=0.8)
        with client.websocket_connect("/ocpp/CP1", subprotocols=SUBPROTOCOL) as ws:
            cp = ChargePointSim(ws)
            cp.boot()
            cp.start(1, "T1")
            pool, fut = _post_in_thread(
                client, "/api/v1/v2g/dispatch", {"target_power_kw": -5, "duration_seconds": 600}
            )
            uid, payload = cp.expect_call("SetChargingProfile")
            prof = payload["csChargingProfiles"]
            assert prof["chargingProfileId"] == 250
            assert prof["stackLevel"] == 1
            [period] = prof["chargingSchedule"]["chargingSchedulePeriod"]
            assert period["limit"] == pytest.approx(-5000.0)
            assert prof["chargingSchedule"]["duration"] == 600
            cp.reply(uid, {"status": "Accepted"})
            resp = fut.result(timeout=10)
            pool.shutdown()
    body = resp.json()
    assert body["achieved_power_kw"] == pytest.approx(-5.0)
    assert body["deliveries"][0]["status"] == "accepted"


def test_simulated_ocpp_reports_simulated_delivery(tmp_path):
    factory = tmp_session_factory(tmp_path / "v2g.db")
    adapter = OCPPAdapter()
    asyncio.run(adapter.connect())  # no Central System -> SIMULATED
    adapter.register_charge_point(ChargePoint(charge_point_id="SIM-1"))
    registry = ProtocolRegistry()
    registry.register(adapter)
    app = build_app(factory, registry)
    with TestClient(app) as client:
        _vehicle(client, charge_point_id="SIM-1", connector_id=1, target_soc=0.9)
        body = client.post("/api/v1/v2g/schedule", json={}).json()
        [delivery] = body["deliveries"]
        assert delivery["status"] == "simulated"
        assert "simulated" in delivery["detail"]
        body = client.post("/api/v1/v2g/schedule", json={"push_to_chargers": False}).json()
        assert body["deliveries"][0]["status"] == "skipped"
        history = client.get("/api/v1/v2g/schedules").json()
        assert [h["kind"] for h in history] == ["schedule", "schedule"]


def test_binding_adopts_live_charger_state(setup):
    app, _, _ = setup
    with TestClient(app) as client:
        with client.websocket_connect("/ocpp/CP1", subprotocols=SUBPROTOCOL) as ws:
            cp = ChargePointSim(ws)
            cp.boot()
            cp.status(1, "Preparing")
            tx = cp.start(1, "ANY")
            _vehicle(client, connection_state="disconnected")
            ev = client.put(
                "/api/v1/v2g/vehicles/ev-1/binding", json={"charge_point_id": "CP1"}
            ).json()
            assert ev["active_transaction_id"] == tx
            assert ev["charger_status"] == "Preparing"
            assert ev["connection_state"] == "connected_idle"


# ---------------------------------------------------------------------------
# Profile compaction
# ---------------------------------------------------------------------------


def test_compact_slots_fills_gaps_merges_and_truncates():
    slots = [
        {"start_time": 0, "end_time": 900, "power_kw": 7.0},
        {"start_time": 900, "end_time": 1800, "power_kw": 7.0},
        {"start_time": 3600, "end_time": 4500, "power_kw": -5.0},
    ]
    merged, truncated = compact_slots(slots, 48)
    assert not truncated
    assert merged == [
        {"start_time": 0, "end_time": 1800, "power_kw": 7.0},
        {"start_time": 1800, "end_time": 3600, "power_kw": 0.0},  # explicit idle gap
        {"start_time": 3600, "end_time": 4500, "power_kw": -5.0},
    ]
    many = [
        {"start_time": i * 900, "end_time": (i + 1) * 900, "power_kw": float(i % 2)}
        for i in range(10)
    ]
    capped, truncated = compact_slots(many, 4)
    assert truncated and len(capped) == 4


# ---------------------------------------------------------------------------
# Protocol data API + operator actions against a live charger
# ---------------------------------------------------------------------------


def test_charge_point_read_api_and_remote_start_stop(setup):
    app, _, _ = setup
    with TestClient(app) as client:
        with client.websocket_connect("/ocpp/CP1", subprotocols=SUBPROTOCOL) as ws:
            cp = ChargePointSim(ws)
            cp.boot()
            cp.status(1, "Charging")
            tx = cp.start(1, "T1")
            cp.meter(1, tx, power_w=11000, soc=40)

            listing = client.get("/api/v1/protocols/ocpp/charge-points").json()
            assert listing["mode"] == "live" and listing["simulated"] is False
            [info] = listing["charge_points"]
            assert info["charge_point_id"] == "CP1"
            assert info["connected"] is True
            [conn] = info["connectors"]
            assert conn["status"] == "Charging"
            assert conn["meter"]["power_kw"] == pytest.approx(11.0)
            assert conn["meter"]["soc"] == pytest.approx(0.4)
            assert info["transactions"][0]["transaction_id"] == tx
            assert client.get("/api/v1/protocols/ocpp/transactions").json()["transactions"]

            pool, fut = _post_in_thread(
                client,
                "/api/v1/protocols/ocpp/charge-points/CP1/remote-stop",
                {"transaction_id": tx},
            )
            uid, payload = cp.expect_call("RemoteStopTransaction")
            assert payload == {"transactionId": tx}
            cp.reply(uid, {"status": "Accepted"})
            body = fut.result(timeout=10).json()
            assert body["accepted"] is True and body["live"] is True

            fut = pool.submit(
                client.post,
                "/api/v1/protocols/ocpp/charge-points/CP1/remote-start",
                json={"connector_id": 2, "id_tag": "OPS"},
            )
            uid, payload = cp.expect_call("RemoteStartTransaction")
            assert payload == {"connectorId": 2, "idTag": "OPS"}
            cp.reply(uid, {"status": "Rejected"})
            assert fut.result(timeout=10).json()["accepted"] is False
            pool.shutdown()

        assert client.get("/api/v1/protocols/ocpp/charge-points/NOPE").status_code == 404
        assert (
            client.post("/api/v1/protocols/ocpp/charge-points/NOPE/remote-start").status_code
            == 404
        )


async def test_unconsumed_message_buffer_drops_oldest_without_counting_errors():
    adapter = OCPPAdapter()
    for i in range(510):
        await adapter._publish(f"ocpp/test/{i}", {"i": i})
    assert adapter.metrics.errors == 0
    assert adapter._message_queue.qsize() == 500
    assert (await adapter.receive()).payload == {"i": 10}
