"""DR orchestrator: OpenADR events / IEEE 2030.5 controls -> recorded fleet dispatch."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

import pytest
import respx
from _v2g_helpers import tmp_session_factory
from sqlalchemy import select
from test_openadr_ven import (
    NS,
    VTN,
    _body,
    _xml,
    created_party_registration,
    distribute_event,
    oadr_event,
    oadr_response,
)

from vpp.db.models import (
    DREventResponseModel,
    OptimizationRunModel,
    ResourceModel,
    V2GVehicleModel,
)
from vpp.dr.orchestrator import DROrchestrator
from vpp.dr.translate import (
    DRPolicy,
    FleetCapability,
    combine,
    opt_decision,
    translate_ieee2030_5,
    translate_openadr,
)
from vpp.events import EventType, get_event_bus
from vpp.protocols import openadr_xml as ox
from vpp.protocols.base import ProtocolRegistry
from vpp.protocols.ieee2030_5 import DERControl, DERProgram, EventStatusCode, IEEE2030_5Adapter
from vpp.protocols.ocpp import ChargePoint, OCPPAdapter
from vpp.protocols.openadr import DREvent, DREventStatus, DRSignalType, OpenADRAdapter

NOW = 1_900_000_000.0


class Clock:
    def __init__(self, t: float = NOW) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _event(
    event_id: str = "E1",
    *,
    signal: DRSignalType = DRSignalType.LOAD_DISPATCH,
    signal_type: str = "delta",
    level: float = 30.0,
    start: float = NOW - 60,
    duration: int = 3600,
    **meta,
) -> DREvent:
    return DREvent(
        event_id=event_id,
        signal_type=signal,
        signal_level=level,
        start_time=start,
        duration_seconds=duration,
        metadata={
            "signals": [
                {
                    "signal_name": signal.value,
                    "signal_type": signal_type,
                    "signal_id": "S",
                    "current_value": None,
                    "intervals": [
                        {"start_time": start, "duration_seconds": duration, "value": level}
                    ],
                }
            ],
            **meta,
        },
    )


# ---------------------------------------------------------------------------
# Translation rules (pure)
# ---------------------------------------------------------------------------

CAP = FleetCapability(export_kw=100.0, import_kw=80.0)
ON = DRPolicy(auto_response=True)


def test_openadr_translation_rules():
    assert translate_openadr(_event(level=30), CAP, ON, now=NOW).target_kw == 30.0
    setpoint = translate_openadr(_event(signal_type="setpoint", level=-20), CAP, ON, now=NOW)
    assert setpoint.target_kw == 20.0  # net-load setpoint -20 kW = export 20 kW
    simple = translate_openadr(
        _event(signal=DRSignalType.SIMPLE, signal_type="level", level=2), CAP, ON, now=NOW
    )
    assert simple.target_kw == pytest.approx(75.0)  # level 2 -> 75% of 100 kW
    top = translate_openadr(
        _event(signal=DRSignalType.SIMPLE, signal_type="level", level=9), CAP, ON, now=NOW
    )
    assert top.target_kw == pytest.approx(100.0)  # clamped to the last level
    control = translate_openadr(
        _event(signal=DRSignalType.LOAD_CONTROL, signal_type="x-loadControlSetpoint", level=5),
        CAP,
        ON,
        now=NOW,
    )
    assert control.target_kw == -5.0
    price = translate_openadr(
        _event(signal=DRSignalType.ELECTRICITY_PRICE, signal_type="price", level=0.4),
        CAP,
        ON,
        now=NOW,
    )
    assert not price.supported and price.target_kw is None


def test_openadr_uses_current_interval_value():
    event = _event(level=10)
    event.metadata["signals"][0]["intervals"] = [
        {"start_time": NOW - 60, "duration_seconds": 600, "value": 10.0},
        {"start_time": NOW + 540, "duration_seconds": 600, "value": 25.0},
    ]
    assert translate_openadr(event, CAP, ON, now=NOW).target_kw == 10.0
    assert translate_openadr(event, CAP, ON, now=NOW + 600).target_kw == 25.0


def test_ieee_translation_priority_and_limits():
    high = DERControl(control_id="HI", max_limit_pct=10.0, creation_time=2)
    low = DERControl(control_id="LO", set_watts=40_000, gen_limit_w=50_000, creation_time=1)
    d = translate_ieee2030_5([high, low], CAP, DRPolicy(set_max_w=200_000))
    assert d.target_kw == 40.0
    assert d.max_export_kw == pytest.approx(20.0)  # 10% of setMaxW 200 kW (< GenLim 50 kW)
    effective = combine(d, None, DRPolicy())
    assert effective.target_kw == pytest.approx(20.0)  # own target clamped by own limit

    cease = translate_ieee2030_5([DERControl(control_id="X", energize=False)], CAP, ON)
    assert cease.target_kw == 0.0
    fixed = translate_ieee2030_5([DERControl(control_id="F", fixed_w_pct=-50)], CAP, ON)
    assert fixed.target_kw == pytest.approx(-50.0)  # % of fleet capability, absorbing
    assert translate_ieee2030_5([], CAP, ON) is None
    assert not translate_ieee2030_5([DERControl(control_id="V", set_var=100)], CAP, ON).supported


def test_combine_ieee_limits_clamp_openadr_and_operator_caps_apply_last():
    oadr = translate_openadr(_event(level=60), CAP, ON, now=NOW)
    limit = translate_ieee2030_5([DERControl(control_id="L", gen_limit_w=25_000)], CAP, ON)
    effective = combine(limit, oadr, DRPolicy(max_export_kw=15))
    assert effective.protocol == "openadr"
    assert effective.target_kw == 15.0
    assert "IEEE 2030.5 export limit" in effective.reason
    assert "dr_max_export_kw" in effective.reason
    absorb = translate_openadr(_event(signal_type="setpoint", level=90), CAP, ON, now=NOW)
    assert combine(None, absorb, DRPolicy(max_import_kw=40)).target_kw == -40.0


def test_opt_decision_rules():
    d = translate_openadr(_event(level=30), CAP, ON, now=NOW)
    assert opt_decision(d, CAP, DRPolicy(auto_response=False))[:2] == ("optIn", "observed")
    assert opt_decision(d, CAP, DRPolicy(auto_response=False, auto_opt_in=False))[0] == "optOut"
    assert opt_decision(d, CAP, ON)[:2] == ("optIn", "accepted")
    big = translate_openadr(_event(level=500), CAP, ON, now=NOW)
    assert opt_decision(big, CAP, ON)[:2] == ("optOut", "declined")  # 100 < 50% of 500
    assert opt_decision(big, CAP, DRPolicy(auto_response=True, min_opt_in_fraction=0.1))[0] == (
        "optIn"
    )
    assert opt_decision(d, FleetCapability(), ON)[0] == "optOut"  # nothing to dispatch
    assert opt_decision(d, CAP, ON, test_event=True)[1] == "test_event"


# ---------------------------------------------------------------------------
# Orchestrator against a database
# ---------------------------------------------------------------------------


async def _seed_battery(factory, *, rated=50.0, capacity=200.0, soc=0.5, name="bess-1"):
    async with factory() as session:
        row = ResourceModel(
            name=name,
            resource_type="battery",
            rated_power=rated,
            online=True,
            nominal_energy_kwh=capacity,
            config_json=json.dumps({"state_of_charge": soc}),
            metadata_json="{}",
        )
        session.add(row)
        await session.commit()
        return row.id


async def _rows(factory, model, **where):
    async with factory() as session:
        stmt = select(model)
        for k, v in where.items():
            stmt = stmt.where(getattr(model, k) == v)
        return list((await session.execute(stmt)).scalars().all())


async def _setup(tmp_path, policy: DRPolicy, clock: Clock | None = None):
    factory = tmp_session_factory(tmp_path / "dr.db")
    registry = ProtocolRegistry()
    openadr = OpenADRAdapter()
    openadr.configure(role="ven", poll_interval_s=0)
    await openadr.connect()  # simulated VEN
    registry.register(openadr)
    orch = DROrchestrator(policy, registry, session_factory=factory, clock=clock or Clock())
    orch.attach_openadr(openadr)
    return factory, registry, openadr, orch


@pytest.fixture
def captured_events():
    events: list = []

    async def on_event(event):
        events.append(event)

    sub = get_event_bus().subscribe(on_event)
    yield events
    get_event_bus().unsubscribe(sub)


async def test_auto_response_off_records_but_never_dispatches(tmp_path, captured_events):
    factory, _, openadr, orch = await _setup(tmp_path, DRPolicy(auto_response=False))
    await _seed_battery(factory)
    response = await openadr.handle_incoming_event(_event())
    assert response.opt_type == "optIn"  # unchanged legacy behaviour (auto_opt_in)
    assert await orch.tick() is None
    [row] = await _rows(factory, DREventResponseModel)
    assert row.action == "received"
    assert json.loads(row.details_json)["decision"] == "observed"
    assert "operator action required" in row.reason
    assert await _rows(factory, OptimizationRunModel) == []
    types = {e.event_type for e in captured_events}
    assert EventType.DR_EVENT_RECEIVED in types and EventType.DR_RESPONSE_SENT in types
    assert EventType.DISPATCH_EXECUTED not in types


async def test_event_window_dispatch_redispatch_and_release(tmp_path, captured_events):
    clock = Clock()
    policy = DRPolicy(auto_response=True, redispatch_interval_s=600)
    factory, _, openadr, orch = await _setup(tmp_path, policy, clock)
    battery = await _seed_battery(factory)
    event = _event(level=30, start=NOW - 60, duration=1800)
    assert (await openadr.handle_incoming_event(event)).opt_type == "optIn"

    summary = await orch.tick()
    assert summary["action"] == "dispatched"
    assert summary["target_kw"] == 30.0
    assert summary["delivered_kw"] == pytest.approx(30.0)
    assert summary["allocations"][battery] == pytest.approx(30.0)
    assert summary["interval_minutes"] == 10
    [run] = await _rows(factory, OptimizationRunModel)
    assert run.problem_type == "dr_dispatch"
    assert json.loads(run.parameters_json)["dr"]["source_id"] == "E1"
    [dispatched] = await _rows(factory, DREventResponseModel, action="dispatched")
    assert dispatched.run_id == run.id
    assert dispatched.delivered_kw == pytest.approx(30.0)

    assert await orch.tick() is None  # unchanged target, within the interval
    clock.t += 601
    again = await orch.tick()
    assert again["redispatch"] is True
    assert len(await _rows(factory, OptimizationRunModel)) == 2

    clock.t = NOW + 1800  # event over
    released = await orch.tick()
    assert released["action"] == "released"
    assert orch.status()["active"] is None
    assert len(await _rows(factory, DREventResponseModel, action="released")) == 1
    dispatch_events = [e for e in captured_events if e.event_type == EventType.DISPATCH_EXECUTED]
    assert [e.data["action"] for e in dispatch_events] == ["dispatched", "dispatched", "released"]
    completed = [
        e
        for e in captured_events
        if e.event_type == EventType.OPTIMIZATION_COMPLETED
        and e.data["problem_type"] == "dr_dispatch"
    ]
    assert len(completed) == 2


async def test_never_exceeds_resource_limits(tmp_path):
    factory, _, openadr, orch = await _setup(
        tmp_path, DRPolicy(auto_response=True, min_opt_in_fraction=0.1)
    )
    await _seed_battery(factory, rated=20.0)
    await openadr.handle_incoming_event(_event(level=100))
    summary = await orch.tick()
    assert summary["delivered_kw"] == pytest.approx(20.0)
    assert summary["shortfall_kw"] == pytest.approx(80.0)
    assert summary["status"] == "shortfall"


async def test_opted_out_test_and_unsupported_events_are_not_dispatched(tmp_path):
    factory, _, openadr, orch = await _setup(tmp_path, DRPolicy(auto_response=True))
    await _seed_battery(factory, rated=10.0)
    too_big = await openadr.handle_incoming_event(_event("BIG", level=500))
    assert too_big.opt_type == "optOut"
    await openadr.handle_incoming_event(_event("TEST", level=5, test_event=True))
    await openadr.handle_incoming_event(
        _event("PRICE", signal=DRSignalType.ELECTRICITY_PRICE, signal_type="price", level=1)
    )
    cancelled = _event("CANC", level=5)
    await openadr.handle_incoming_event(cancelled)
    cancelled.status = DREventStatus.CANCELLED
    assert await orch.tick() is None
    assert await _rows(factory, OptimizationRunModel) == []
    rows = {r.source_id: r for r in await _rows(factory, DREventResponseModel)}
    assert rows["BIG"].opt_type == "optOut"
    assert json.loads(rows["TEST"].details_json)["decision"] == "test_event"
    assert json.loads(rows["PRICE"].details_json)["decision"] == "unsupported"


async def test_operator_override_opt_out_releases_dispatch(tmp_path):
    factory, _, openadr, orch = await _setup(tmp_path, DRPolicy(auto_response=True))
    await _seed_battery(factory)
    await openadr.handle_incoming_event(_event(level=10))
    assert (await orch.tick())["action"] == "dispatched"
    result = await orch.override_opt("E1", "optOut", user="ops")
    assert result == {
        "event_id": "E1",
        "opt_type": "optOut",
        "sent_to_vtn": False,
        "released_dispatch": True,
    }
    assert openadr.get_response("E1").opt_type == "optOut"
    assert await orch.tick() is None  # stays out
    actions = [r.action for r in await _rows(factory, DREventResponseModel)]
    assert actions.count("opt_override") == 1 and actions.count("released") == 1


async def test_ieee2030_5_control_target_wins_and_limits_apply(tmp_path):
    clock = Clock(time.time())
    factory, registry, openadr, orch = await _setup(
        tmp_path, DRPolicy(auto_response=True, set_max_w=100_000), clock
    )
    await _seed_battery(factory, rated=80.0, capacity=400.0)
    ieee = IEEE2030_5Adapter()
    ieee.configure(poll_interval_s=0)
    await ieee.connect()  # simulated
    registry.register(ieee)
    ieee.register_program(DERProgram(program_id="P1", primacy=1))
    await openadr.handle_incoming_event(_event(level=60, start=clock.t - 60))

    limit = DERControl(
        control_id="LIMIT",
        max_limit_pct=25.0,  # 25% of setMaxW 100 kW
        start_time=clock.t - 30,
        duration_seconds=900,
        event_status=EventStatusCode.ACTIVE,
        program_id="P1",
    )
    await ieee.apply_control("P1", limit)
    summary = await orch.tick()
    assert summary["protocol"] == "openadr"
    assert summary["target_kw"] == pytest.approx(25.0)  # OpenADR 60 kW clamped by 2030.5

    target = DERControl(
        control_id="TARGET",
        set_watts=-15_000,  # charge 15 kW
        start_time=clock.t - 10,
        duration_seconds=900,
        creation_time=5,
        program_id="P1",
    )
    await ieee.apply_control("P1", target)
    summary = await orch.tick()
    assert summary["protocol"] == "ieee2030_5"
    assert summary["target_kw"] == pytest.approx(-15.0)
    assert summary["delivered_kw"] == pytest.approx(-15.0)
    received = await _rows(factory, DREventResponseModel, protocol="ieee2030_5")
    assert {r.action for r in received} >= {"received", "dispatched"}


async def test_connected_evs_take_part_and_get_setpoints(tmp_path, captured_events):
    factory, registry, openadr, orch = await _setup(tmp_path, DRPolicy(auto_response=True))
    ocpp = OCPPAdapter()
    await ocpp.connect()  # simulated Central System
    ocpp.register_charge_point(ChargePoint(charge_point_id="SIM-1", v2g_capable=True))
    registry.register(ocpp)
    async with factory() as session:
        session.add(
            V2GVehicleModel(
                id="ev-1",
                capacity_kwh=60,
                current_soc=0.9,
                min_soc=0.2,
                target_soc=0.5,
                max_charge_kw=11,
                max_discharge_kw=11,
                v2g_capable=True,
                connection_state="connected_idle",
                charge_point_id="SIM-1",
                connector_id=1,
            )
        )
        session.add(
            V2GVehicleModel(
                id="ev-away",
                capacity_kwh=60,
                current_soc=0.9,
                min_soc=0.2,
                target_soc=0.5,
                max_charge_kw=11,
                max_discharge_kw=11,
                v2g_capable=True,
                connection_state="disconnected",
            )
        )
        await session.commit()
    await openadr.handle_incoming_event(_event(level=8))  # only the EV can deliver
    summary = await orch.tick()
    assert summary["allocations"] == {"ev:ev-1": pytest.approx(8.0)}
    assert summary["ev_deliveries"] == {"simulated": 1}
    profile = ocpp.get_charge_point("SIM-1").active_profile
    assert profile.profile_id == 300
    assert profile.schedule[0].limit == pytest.approx(-8000.0)  # discharge
    v2g = [e for e in captured_events if e.event_type == EventType.V2G_DISPATCH]
    assert v2g and v2g[-1].data["dispatch_kw"] == pytest.approx(8.0)

    await orch.override_opt("E1", "optOut")
    assert ocpp.get_charge_point("SIM-1").active_profile is None  # DR profile cleared


async def test_ev_without_flexibility_never_exports(tmp_path):
    factory, _, _, orch = await _setup(tmp_path, DRPolicy(auto_response=True))
    async with factory() as session:
        session.add(
            V2GVehicleModel(
                id="ev-1",
                capacity_kwh=60,
                current_soc=0.5,
                min_soc=0.2,
                target_soc=0.9,
                max_charge_kw=11,
                max_discharge_kw=11,
                v2g_capable=True,
                connection_state="connected_idle",
                # leaving in 30 min and still needs ~20 kWh: no slack for V2G
                departure_time=datetime.now(timezone.utc) + timedelta(minutes=30),
            )
        )
        await session.commit()
    cap = await orch.fleet_capability(900)
    assert cap.export_kw == 0.0
    assert cap.import_kw > 0


# ---------------------------------------------------------------------------
# Live VEN against a mocked VTN
# ---------------------------------------------------------------------------


@pytest.fixture
def vtn():
    with respx.mock(base_url=VTN, assert_all_called=False) as mock:
        mock.post("/EiRegisterParty").mock(return_value=_xml(created_party_registration()))
        mock.post("/EiEvent").mock(return_value=_xml(oadr_response()))
        mock.post("/OadrPoll").mock(return_value=_xml(oadr_response()))
        yield mock


@pytest.mark.parametrize(("kw", "expected"), [(20, "optIn"), (900, "optOut")])
async def test_live_ven_answers_vtn_with_orchestrator_decision(tmp_path, vtn, kw, expected):
    factory = tmp_session_factory(tmp_path / "dr.db")
    await _seed_battery(factory, rated=40.0)
    adapter = OpenADRAdapter()
    adapter.configure(role="ven", vtn_url=VTN, ven_name="vpp-test", poll_interval_s=0)
    registry = ProtocolRegistry()
    registry.register(adapter)
    orch = DROrchestrator(DRPolicy(auto_response=True), registry, session_factory=factory)
    orch.attach_openadr(adapter)
    start = datetime.now(timezone.utc) + timedelta(minutes=5)
    vtn.post("/OadrPoll").mock(
        side_effect=[
            _xml(
                distribute_event(
                    oadr_event(
                        "EVT-DR",
                        start=start,
                        signal_name="LOAD_DISPATCH",
                        signal_type="delta",
                        values=(kw, kw),
                    )
                )
            ),
            _xml(oadr_response()),
        ]
    )
    await adapter.connect()
    try:
        await adapter.poll_once()
    finally:
        await adapter.disconnect()
    created = [c for c in vtn.calls if c.request.url.path.endswith("/EiEvent")]
    assert len(created) == 1
    body = _body(created[0].request)
    assert ox.message_name(body) == "oadrCreatedEvent"
    assert body.findtext(".//ei:optType", namespaces=NS) == expected
    [row] = await _rows(factory, DREventResponseModel)
    assert row.opt_type == expected and row.source_id == "EVT-DR"


# ---------------------------------------------------------------------------
# Startup wiring
# ---------------------------------------------------------------------------


async def test_bootstrap_wires_vehicle_bridge_and_orchestrator():
    import asyncio

    from vpp.dr.orchestrator import get_dr_orchestrator
    from vpp.protocols.bootstrap import start_protocol_adapters, stop_protocol_adapters
    from vpp.settings import Settings
    from vpp.v2g.ocpp_bridge import OCPPVehicleBridge

    registry = ProtocolRegistry()
    settings = Settings(ocpp_enabled=True, openadr_enabled=True, dr_tick_interval_s=60)
    tasks = start_protocol_adapters(settings, registry)
    try:
        assert len(tasks) == 3  # ocpp, openadr, dr orchestrator
        orch = get_dr_orchestrator()
        assert orch is not None and orch.policy.auto_response is False
        for _ in range(50):
            if len(registry) == 2:
                break
            await asyncio.sleep(0.01)
        ocpp = registry.get("ocpp")
        assert any(
            isinstance(getattr(cb, "__self__", None), OCPPVehicleBridge)
            for cb in ocpp._subscribers.get("*", [])
        )
        assert registry.get("openadr")._event_handlers == [orch.on_openadr_event]
        assert orch.status()["protocols"] == {"openadr": True, "ieee2030_5": False, "ocpp": True}
    finally:
        await stop_protocol_adapters(tasks)
    assert get_dr_orchestrator() is None
