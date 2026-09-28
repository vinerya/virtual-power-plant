# ruff: noqa: SIM117  (nested pytest.raises/websocket contexts read clearer)
"""OCPP 1.6-J Central System tests.

A simulated charge point talks to the real ``/ocpp/{id}`` WebSocket route
through Starlette's TestClient, exchanging raw OCPP-J frames.
"""

from __future__ import annotations

import asyncio
import base64
import json

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from vpp.api.routes import ocpp as ocpp_routes
from vpp.api.routes.protocols import get_registry
from vpp.events import EventType, get_event_bus
from vpp.protocols.base import ProtocolMode, ProtocolRegistry, ProtocolStatus
from vpp.protocols.ocpp import (
    ChargePoint,
    ChargePointStatus,
    ChargingProfilePurpose,
    OCPPAdapter,
    parse_meter_values,
    schedule_to_charging_profile,
)
from vpp.protocols.ocpp_j import (
    OCPPCallError,
    OCPPError,
    OCPPErrorCode,
    OCPPJSession,
    OCPPTimeoutError,
    parse_frame,
)

SUBPROTOCOL = ["ocpp1.6"]


def _make_app(**config) -> tuple[FastAPI, OCPPAdapter]:
    adapter = OCPPAdapter()
    adapter.configure(central_system_enabled=True, call_timeout_s=5, **config)
    asyncio.run(adapter.connect())
    registry = ProtocolRegistry()
    registry.register(adapter)
    app = FastAPI()
    app.include_router(ocpp_routes.router)
    app.dependency_overrides[get_registry] = lambda: registry
    return app, adapter


class ChargePointSim:
    """Minimal OCPP 1.6-J charge point over a TestClient websocket."""

    def __init__(self, ws) -> None:
        self.ws = ws
        self._n = 0

    def call(self, action: str, payload: dict) -> list:
        self._n += 1
        uid = f"cp-{self._n}"
        self.ws.send_text(json.dumps([2, uid, action, payload]))
        reply = json.loads(self.ws.receive_text())
        assert reply[1] == uid
        return reply

    def expect_call(self, action: str) -> tuple[str, dict]:
        frame = json.loads(self.ws.receive_text())
        assert frame[0] == 2, frame
        assert frame[2] == action, frame
        return frame[1], frame[3]

    def reply(self, uid: str, payload: dict) -> None:
        self.ws.send_text(json.dumps([3, uid, payload]))


# ---------------------------------------------------------------------------
# Mode / status honesty
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unconfigured_adapter_is_simulated_not_connected():
    adapter = OCPPAdapter()
    await adapter.connect()
    assert adapter.status == ProtocolStatus.SIMULATED
    assert adapter.mode == ProtocolMode.SIMULATED
    assert not adapter.is_connected
    assert adapter.is_operational


@pytest.mark.asyncio
async def test_central_system_enabled_is_connected_live():
    adapter = OCPPAdapter()
    adapter.configure(central_system_enabled=True)
    await adapter.connect()
    assert adapter.status == ProtocolStatus.CONNECTED
    assert adapter.mode == ProtocolMode.LIVE
    await adapter.disconnect()
    assert adapter.status == ProtocolStatus.DISCONNECTED


# ---------------------------------------------------------------------------
# Handshake
# ---------------------------------------------------------------------------

def test_rejects_when_central_system_not_running():
    adapter = OCPPAdapter()
    asyncio.run(adapter.connect())  # simulated
    registry = ProtocolRegistry()
    registry.register(adapter)
    app = FastAPI()
    app.include_router(ocpp_routes.router)
    app.dependency_overrides[get_registry] = lambda: registry
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ocpp/CP1", subprotocols=SUBPROTOCOL):
                pass


def test_rejects_missing_subprotocol():
    app, _ = _make_app()
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ocpp/CP1"):
                pass


def test_negotiates_ocpp16_subprotocol():
    app, adapter = _make_app()
    with TestClient(app) as client:
        with client.websocket_connect("/ocpp/CP1", subprotocols=SUBPROTOCOL) as ws:
            assert ws.accepted_subprotocol == "ocpp1.6"
            assert adapter.is_charge_point_connected("CP1")
    assert not adapter.is_charge_point_connected("CP1")


def test_basic_auth_security_profile_1():
    app, _ = _make_app(basic_auth_password="s3cret")
    good = "Basic " + base64.b64encode(b"CP1:s3cret").decode()
    bad = "Basic " + base64.b64encode(b"CP1:nope").decode()
    other = "Basic " + base64.b64encode(b"CP2:s3cret").decode()
    with TestClient(app) as client:
        for header in (None, bad, other):
            headers = {"Authorization": header} if header else {}
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect(
                    "/ocpp/CP1", subprotocols=SUBPROTOCOL, headers=headers
                ):
                    pass
        with client.websocket_connect(
            "/ocpp/CP1", subprotocols=SUBPROTOCOL, headers={"Authorization": good}
        ) as ws:
            assert ws.accepted_subprotocol == "ocpp1.6"


def test_allow_list():
    app, _ = _make_app(allowed_charge_points=["CP-OK"])
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ocpp/CP-EVIL", subprotocols=SUBPROTOCOL):
                pass
        with client.websocket_connect("/ocpp/CP-OK", subprotocols=SUBPROTOCOL):
            pass


# ---------------------------------------------------------------------------
# Charge point -> Central System
# ---------------------------------------------------------------------------

def test_full_charging_session_flow():
    app, adapter = _make_app(heartbeat_interval_s=120, authorized_id_tags=["TAG1"])
    events = []

    async def on_event(event):
        events.append(event)

    sub = get_event_bus().subscribe(on_event, event_types={EventType.RESOURCE_UPDATED})
    try:
        with TestClient(app) as client:
            with client.websocket_connect("/ocpp/CP1", subprotocols=SUBPROTOCOL) as ws:
                cp = ChargePointSim(ws)

                boot = cp.call("BootNotification", {
                    "chargePointVendor": "ABB", "chargePointModel": "Terra AC",
                    "chargePointSerialNumber": "SN-1", "firmwareVersion": "1.2.3",
                })
                assert boot[0] == 3
                assert boot[2]["status"] == "Accepted"
                assert boot[2]["interval"] == 120
                assert boot[2]["currentTime"].endswith("Z")

                hb = cp.call("Heartbeat", {})
                assert "currentTime" in hb[2]

                assert cp.call("Authorize", {"idTag": "TAG1"})[2] == {
                    "idTagInfo": {"status": "Accepted"}
                }
                assert cp.call("Authorize", {"idTag": "BAD"})[2]["idTagInfo"]["status"] == "Invalid"

                st = cp.call("StatusNotification", {
                    "connectorId": 1, "errorCode": "NoError", "status": "Preparing",
                })
                assert st == [3, st[1], {}]

                start = cp.call("StartTransaction", {
                    "connectorId": 1, "idTag": "TAG1", "meterStart": 1000,
                    "timestamp": "2026-09-28T10:00:00Z",
                })
                tx_id = start[2]["transactionId"]
                assert isinstance(tx_id, int)
                assert start[2]["idTagInfo"]["status"] == "Accepted"

                cp.call("StatusNotification", {
                    "connectorId": 1, "errorCode": "NoError", "status": "Charging",
                })
                mv = cp.call("MeterValues", {
                    "connectorId": 1, "transactionId": tx_id,
                    "meterValue": [{
                        "timestamp": "2026-09-28T10:05:00Z",
                        "sampledValue": [
                            {"value": "7400", "measurand": "Power.Active.Import", "unit": "W"},
                            {"value": "2500", "measurand": "Energy.Active.Import.Register",
                             "unit": "Wh"},
                            {"value": "63", "measurand": "SoC", "unit": "Percent"},
                        ],
                    }],
                })
                assert mv[2] == {}

                charger = adapter.get_charge_point("CP1")
                assert charger is not None
                assert charger.vendor == "ABB"
                assert charger.serial_number == "SN-1"
                assert charger.connected
                assert charger.status == ChargePointStatus.CHARGING
                assert charger.active_transaction_id == str(tx_id)
                assert charger.current_power_kw == pytest.approx(7.4)
                assert charger.current_soc == pytest.approx(0.63)
                assert charger.energy_import_kwh == pytest.approx(2.5)

                stop = cp.call("StopTransaction", {
                    "transactionId": tx_id, "meterStop": 9000,
                    "timestamp": "2026-09-28T11:00:00Z", "idTag": "TAG1",
                })
                assert stop[2]["idTagInfo"]["status"] == "Accepted"
                assert charger.active_transaction_id is None
                assert charger.metadata["last_session_kwh"] == pytest.approx(8.0)
                assert charger.current_power_kw == 0.0
    finally:
        get_event_bus().unsubscribe(sub)

    assert any(e.data.get("charge_point_id") == "CP1" and e.data.get("power_kw") == pytest.approx(7.4)
               for e in events)
    assert not adapter.get_charge_point("CP1").connected
    # 9 inbound CALLs, each counted exactly once
    assert adapter.metrics.messages_received == 9


def test_boot_rejected_when_auto_accept_disabled():
    app, adapter = _make_app(auto_accept_boot=False)
    adapter.register_charge_point(ChargePoint(charge_point_id="KNOWN"))
    with TestClient(app) as client:
        with client.websocket_connect("/ocpp/STRANGER", subprotocols=SUBPROTOCOL) as ws:
            reply = ChargePointSim(ws).call(
                "BootNotification", {"chargePointVendor": "X", "chargePointModel": "Y"}
            )
            assert reply[2]["status"] == "Rejected"
        with client.websocket_connect("/ocpp/KNOWN", subprotocols=SUBPROTOCOL) as ws:
            reply = ChargePointSim(ws).call(
                "BootNotification", {"chargePointVendor": "X", "chargePointModel": "Y"}
            )
            assert reply[2]["status"] == "Accepted"
    assert adapter.get_charge_point("STRANGER") is None


def test_call_errors_for_bad_input():
    app, _ = _make_app()
    with TestClient(app) as client:
        with client.websocket_connect("/ocpp/CP1", subprotocols=SUBPROTOCOL) as ws:
            cp = ChargePointSim(ws)
            unknown = cp.call("ReserveNow", {})
            assert unknown[0] == 4 and unknown[2] == "NotImplemented"

            missing = cp.call("BootNotification", {"chargePointVendor": "X"})
            assert missing[0] == 4 and missing[2] == "OccurenceConstraintViolation"

            bad_status = cp.call("StatusNotification", {
                "connectorId": 1, "errorCode": "NoError", "status": "Exploded",
            })
            assert bad_status[2] == "PropertyConstraintViolation"

            bad_type = cp.call("StartTransaction", {
                "connectorId": "one", "idTag": "T", "meterStart": 0, "timestamp": "x",
            })
            assert bad_type[2] == "TypeConstraintViolation"

            # Payload not an object, but uniqueId recoverable -> CALLERROR
            ws.send_text(json.dumps([2, "u-9", "Heartbeat", []]))
            err = json.loads(ws.receive_text())
            assert err[:3] == [4, "u-9", "FormationViolation"]

            # Garbage is dropped, the session survives
            ws.send_text("not json")
            assert cp.call("Heartbeat", {})[0] == 3


# ---------------------------------------------------------------------------
# Central System -> Charge point
# ---------------------------------------------------------------------------

def _boot_and_start(cp: ChargePointSim) -> int:
    cp.call("BootNotification", {"chargePointVendor": "V", "chargePointModel": "M"})
    return cp.call("StartTransaction", {
        "connectorId": 1, "idTag": "T", "meterStart": 0, "timestamp": "2026-09-28T10:00:00Z",
    })[2]["transactionId"]


def test_remote_start_and_stop_over_the_wire():
    app, adapter = _make_app(remote_id_tag="VPP-DISPATCH")
    with TestClient(app) as client:
        with client.websocket_connect("/ocpp/CP1", subprotocols=SUBPROTOCOL) as ws:
            cp = ChargePointSim(ws)
            cp.call("BootNotification", {"chargePointVendor": "V", "chargePointModel": "M"})

            fut = client.portal.start_task_soon(adapter.remote_start, "CP1", 2)
            uid, payload = cp.expect_call("RemoteStartTransaction")
            assert payload == {"connectorId": 2, "idTag": "VPP-DISPATCH"}
            cp.reply(uid, {"status": "Accepted"})
            assert fut.result(timeout=5) is True

            tx_id = cp.call("StartTransaction", {
                "connectorId": 2, "idTag": "VPP-DISPATCH", "meterStart": 0,
                "timestamp": "2026-09-28T10:00:00Z",
            })[2]["transactionId"]

            fut = client.portal.start_task_soon(adapter.remote_stop, "CP1")
            uid, payload = cp.expect_call("RemoteStopTransaction")
            assert payload == {"transactionId": tx_id}
            cp.reply(uid, {"status": "Rejected"})
            assert fut.result(timeout=5) is False


def test_set_charging_profile_from_v2g_schedule():
    app, adapter = _make_app()
    slots = [
        {"start_time": 1_800_000_000, "end_time": 1_800_000_900, "power_kw": 7.0},
        {"start_time": 1_800_000_900, "end_time": 1_800_001_800, "power_kw": -5.0},
    ]
    with TestClient(app) as client:
        with client.websocket_connect("/ocpp/CP1", subprotocols=SUBPROTOCOL) as ws:
            cp = ChargePointSim(ws)
            tx_id = _boot_and_start(cp)

            fut = client.portal.start_task_soon(adapter.apply_v2g_schedule, "CP1", slots)
            uid, payload = cp.expect_call("SetChargingProfile")
            assert payload["connectorId"] == 1
            prof = payload["csChargingProfiles"]
            assert prof["chargingProfilePurpose"] == "TxProfile"
            assert prof["chargingProfileKind"] == "Absolute"
            assert prof["transactionId"] == tx_id
            sched = prof["chargingSchedule"]
            assert sched["chargingRateUnit"] == "W"
            assert sched["duration"] == 1800
            assert sched["startSchedule"] == "2027-01-15T08:00:00Z"
            assert [p["startPeriod"] for p in sched["chargingSchedulePeriod"]] == [0, 900]
            assert [p["limit"] for p in sched["chargingSchedulePeriod"]] == [7000.0, -5000.0]
            cp.reply(uid, {"status": "Accepted"})
            assert fut.result(timeout=5) is True
            assert adapter.get_charge_point("CP1").active_profile.profile_id == 200

            fut = client.portal.start_task_soon(adapter.clear_charging_profile, "CP1")
            uid, payload = cp.expect_call("ClearChargingProfile")
            assert payload == {}
            cp.reply(uid, {"status": "Accepted"})
            assert fut.result(timeout=5) is True
            assert adapter.get_charge_point("CP1").active_profile is None


def test_charge_point_callerror_and_not_supported():
    app, adapter = _make_app()
    with TestClient(app) as client:
        with client.websocket_connect("/ocpp/CP1", subprotocols=SUBPROTOCOL) as ws:
            cp = ChargePointSim(ws)
            _boot_and_start(cp)
            profile = schedule_to_charging_profile(
                [{"start_time": 0, "end_time": 60, "power_kw": 1.0}]
            )
            fut = client.portal.start_task_soon(adapter.set_charging_profile, "CP1", profile)
            uid, _ = cp.expect_call("SetChargingProfile")
            ws.send_text(json.dumps([4, uid, "NotSupported", "no smart charging", {}]))
            assert fut.result(timeout=5) is False
            assert adapter.metrics.errors == 1

            fut = client.portal.start_task_soon(
                adapter.call, "CP1", "GetConfiguration", {"key": []}
            )
            uid, _ = cp.expect_call("GetConfiguration")
            ws.send_text(json.dumps([4, uid, "NotImplemented", "", {}]))
            with pytest.raises(OCPPCallError) as info:
                fut.result(timeout=5)
            assert info.value.code == "NotImplemented"


def test_live_operations_fail_for_offline_charge_point():
    adapter = OCPPAdapter()
    adapter.configure(central_system_enabled=True)

    async def scenario():
        await adapter.connect()
        adapter.register_charge_point(ChargePoint(charge_point_id="OFF", v2g_capable=True))
        assert await adapter.remote_start("OFF") is False
        assert await adapter.set_v2g_discharge_profile("OFF", 5.0) is False
        with pytest.raises(ConnectionError):
            await adapter.call("OFF", "Reset", {"type": "Soft"})

    asyncio.run(scenario())


def test_reconnect_replaces_session():
    app, adapter = _make_app()
    with TestClient(app) as client:
        with client.websocket_connect("/ocpp/CP1", subprotocols=SUBPROTOCOL) as ws1:
            first = adapter._sessions["CP1"]
            with client.websocket_connect("/ocpp/CP1", subprotocols=SUBPROTOCOL) as ws2:
                assert adapter._sessions["CP1"] is not first
                assert first.closed
                assert ChargePointSim(ws2).call("Heartbeat", {})[0] == 3
                with pytest.raises(WebSocketDisconnect):
                    ws1.receive_text()
            assert not adapter.is_charge_point_connected("CP1")


# ---------------------------------------------------------------------------
# OCPP-J session unit tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_session_timeout_and_close():
    sent: list[str] = []

    async def send(text: str) -> None:
        sent.append(text)

    async def handler(cp_id, action, payload):
        return {}

    session = OCPPJSession("CP", send, handler, default_timeout=0.05)
    with pytest.raises(OCPPTimeoutError):
        await session.call("Reset", {"type": "Soft"})
    assert json.loads(sent[0])[2] == "Reset"

    # Late response for the timed-out call is dropped harmlessly
    await session.handle_text(json.dumps([3, json.loads(sent[0])[1], {"status": "Accepted"}]))

    task = asyncio.create_task(session.call("Reset", {"type": "Hard"}, timeout=5))
    await asyncio.sleep(0.01)
    session.close()
    with pytest.raises(ConnectionError):
        await task
    with pytest.raises(ConnectionError):
        await session.call("Reset", {})


@pytest.mark.asyncio
async def test_session_handler_crash_becomes_internal_error():
    sent: list[str] = []

    async def send(text: str) -> None:
        sent.append(text)

    async def handler(cp_id, action, payload):
        raise RuntimeError("boom")

    session = OCPPJSession("CP", send, handler)
    await session.handle_text(json.dumps([2, "x1", "Heartbeat", {}]))
    assert json.loads(sent[0])[:3] == [4, "x1", "InternalError"]


def test_parse_frame_validation():
    with pytest.raises(OCPPError) as info:
        parse_frame("[9, \"id\", {}]")
    assert info.value.code == OCPPErrorCode.PROTOCOL_ERROR
    with pytest.raises(OCPPError):
        parse_frame("{}")
    with pytest.raises(OCPPError):
        parse_frame(json.dumps([2, "x" * 37, "Heartbeat", {}]))


def test_parse_meter_values_phases_and_units():
    readings = parse_meter_values([
        {"timestamp": "t", "sampledValue": [
            {"value": "1.0", "measurand": "Power.Active.Import", "unit": "kW", "phase": "L1"},
            {"value": "1.5", "measurand": "Power.Active.Import", "unit": "kW", "phase": "L2"},
            {"value": "12.5", "unit": "kWh"},  # default measurand = energy register
            {"value": "abc", "measurand": "SoC"},
            {"value": "1", "measurand": "Power.Active.Import", "format": "SignedData"},
        ]},
    ])
    assert readings == {"power_kw": pytest.approx(2.5), "energy_import_kwh": 12.5}
    export = parse_meter_values([{"sampledValue": [
        {"value": "3000", "measurand": "Power.Active.Export", "unit": "W"},
    ]}])
    assert export["power_kw"] == pytest.approx(-3.0)


def test_charging_profile_serialisation():
    profile = schedule_to_charging_profile(
        [{"start_time": 100, "end_time": 200, "power_kw": 3.33333}],
        purpose=ChargingProfilePurpose.TX_DEFAULT,
    )
    body = profile.to_dict()
    assert body["chargingSchedule"]["chargingSchedulePeriod"][0]["limit"] == 3333.3
    assert "transactionId" not in body
    with pytest.raises(ValueError):
        schedule_to_charging_profile([])
