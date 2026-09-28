"""IEEE 2030.5 client tests against a respx-mocked utility server."""

from __future__ import annotations

import time

import httpx
import pytest
import respx

from vpp.protocols.base import ProtocolMode, ProtocolStatus
from vpp.protocols.ieee2030_5 import (
    DERControl,
    DERControlMode,
    DERProgram,
    EventStatusCode,
    IEEE2030_5Adapter,
    lfdi_from_cert_der,
    sfdi_from_lfdi,
)

SERVER = "https://2030-5.utility.example.com"
LFDI = "3E4F45AB31EDFE5B67E343E5E4562E31984E23E5"
NS = 'xmlns="urn:ieee:std:2030.5:ns"'


def sep(text: str) -> httpx.Response:
    return httpx.Response(
        200, content=text.encode(), headers={"Content-Type": "application/sep+xml"}
    )


def dcap(poll_rate: int = 300) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<DeviceCapability {NS} href="/dcap" pollRate="{poll_rate}">
  <TimeLink href="/tm"/>
  <EndDeviceListLink all="2" href="/edev"/>
  <MirrorUsagePointListLink all="0" href="/mup"/>
  <SelfDeviceLink href="/sdev"/>
</DeviceCapability>"""


def end_devices(lfdi: str = LFDI) -> str:
    return f"""<EndDeviceList {NS} all="2" results="2" href="/edev" subscribable="0">
  <EndDevice href="/edev/0" subscribable="0">
    <sFDI>111111111</sFDI>
    <lFDI>AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA</lFDI>
    <FunctionSetAssignmentsListLink all="0" href="/edev/0/fsa"/>
  </EndDevice>
  <EndDevice href="/edev/1" subscribable="0">
    <DERListLink all="1" href="/edev/1/der"/>
    <sFDI>{sfdi_from_lfdi(lfdi)}</sFDI>
    <lFDI>{lfdi.lower()}</lFDI>
    <changedTime>1700000000</changedTime>
    <FunctionSetAssignmentsListLink all="1" href="/edev/1/fsa"/>
    <RegistrationLink href="/edev/1/reg"/>
  </EndDevice>
</EndDeviceList>"""


FSA = f"""<FunctionSetAssignmentsList {NS} all="1" results="1" href="/edev/1/fsa">
  <FunctionSetAssignments href="/edev/1/fsa/0">
    <mRID>A1000000000000000000000000000001</mRID>
    <description>Feeder 12 assignments</description>
    <DERProgramListLink all="2" href="/derp"/>
  </FunctionSetAssignments>
</FunctionSetAssignmentsList>"""


DERP = f"""<DERProgramList {NS} all="2" results="2" href="/derp" pollRate="60">
  <DERProgram href="/derp/1">
    <mRID>B1000000000000000000000000000002</mRID>
    <description>Emergency curtailment</description>
    <ActiveDERControlListLink all="1" href="/derp/1/actderc"/>
    <DERControlListLink all="1" href="/derp/1/derc"/>
    <primacy>1</primacy>
  </DERProgram>
  <DERProgram href="/derp/0">
    <mRID>B1000000000000000000000000000001</mRID>
    <description>Default program</description>
    <DefaultDERControlLink href="/derp/0/dderc"/>
    <DERControlListLink all="3" href="/derp/0/derc"/>
    <primacy>89</primacy>
  </DERProgram>
</DERProgramList>"""


DDERC = f"""<DefaultDERControl {NS} href="/derp/0/dderc">
  <mRID>D1000000000000000000000000000000</mRID>
  <description>Default</description>
  <DERControlBase>
    <opModConnect>true</opModConnect>
    <opModMaxLimW>10000</opModMaxLimW>
  </DERControlBase>
  <setGradW>200</setGradW>
</DefaultDERControl>"""


def derc_item(
    mrid: str, start: int, duration: int, status: int, base: str, created: int = 1700000000
) -> str:
    return f"""
  <DERControl href="/derc/{mrid}" replyTo="/rsps/1/rsp" responseRequired="03">
    <mRID>{mrid}</mRID>
    <description>ctl {mrid}</description>
    <creationTime>{created}</creationTime>
    <EventStatus>
      <currentStatus>{status}</currentStatus>
      <dateTime>{created}</dateTime>
      <potentiallySuperseded>false</potentiallySuperseded>
    </EventStatus>
    <interval><duration>{duration}</duration><start>{start}</start></interval>
    <randomizeDuration>0</randomizeDuration>
    <randomizeStart>30</randomizeStart>
    <DERControlBase>{base}</DERControlBase>
  </DERControl>"""


def derc_list(href: str, items: list[str]) -> str:
    return (
        f'<DERControlList {NS} all="{len(items)}" results="{len(items)}" '
        f'href="{href}" subscribable="0">' + "".join(items) + "</DERControlList>"
    )


@pytest.fixture
def server():
    now = int(time.time())
    with respx.mock(base_url=SERVER, assert_all_called=False) as mock:
        mock.get("/dcap").mock(return_value=sep(dcap()))
        mock.get("/tm").mock(
            return_value=sep(
                f'<Time {NS} href="/tm"><currentTime>{now + 120}</currentTime>'
                "<dstEndTime>0</dstEndTime><dstOffset>0</dstOffset><dstStartTime>0</dstStartTime>"
                "<quality>7</quality><tzOffset>0</tzOffset></Time>"
            )
        )
        mock.get("/edev").mock(return_value=sep(end_devices()))
        mock.get("/edev/1/fsa").mock(return_value=sep(FSA))
        mock.get("/derp").mock(return_value=sep(DERP))
        mock.post("/rsps/1/rsp", name="responses").mock(return_value=httpx.Response(201))
        mock.get("/derp/0/dderc").mock(return_value=sep(DDERC))
        mock.get("/derp/1/derc").mock(
            return_value=sep(
                derc_list(
                    "/derp/1/derc",
                    [
                        derc_item(
                            "C-CURTAIL",
                            now - 60,
                            3600,
                            EventStatusCode.ACTIVE,
                            "<opModGenLimW><multiplier>3</multiplier><value>5</value></opModGenLimW>"
                            "<opModMaxLimW>5000</opModMaxLimW>",
                        ),
                    ],
                )
            )
        )
        mock.get("/derp/0/derc").mock(
            return_value=sep(
                derc_list(
                    "/derp/0/derc",
                    [
                        derc_item(
                            "C-TARGET",
                            now - 30,
                            1800,
                            EventStatusCode.SCHEDULED,
                            "<opModTargetW><multiplier>0</multiplier><value>-3000</value></opModTargetW>"
                            "<opModFixedPFInjectW><displacement>95</displacement>"
                            "<multiplier>-2</multiplier></opModFixedPFInjectW>"
                            "<rampTms>500</rampTms>",
                        ),
                        derc_item(
                            "C-FUTURE",
                            now + 7200,
                            600,
                            EventStatusCode.SCHEDULED,
                            "<opModEnergize>false</opModEnergize>",
                        ),
                        derc_item(
                            "C-CANCELLED",
                            now - 60,
                            3600,
                            EventStatusCode.CANCELLED,
                            "<opModConnect>false</opModConnect>",
                        ),
                    ],
                )
            )
        )
        yield mock


def _adapter(**config) -> IEEE2030_5Adapter:
    adapter = IEEE2030_5Adapter()
    adapter.configure(server_url=SERVER, lfdi=LFDI, poll_interval_s=0, **config)
    return adapter


@pytest.mark.asyncio
async def test_without_server_url_is_simulated():
    adapter = IEEE2030_5Adapter()
    adapter.configure(poll_interval_s=0)
    await adapter.connect()
    assert adapter.status == ProtocolStatus.SIMULATED
    assert adapter.mode == ProtocolMode.SIMULATED
    assert not adapter.is_connected
    program = DERProgram(program_id="p")
    adapter.register_program(program)
    await adapter.apply_control("p", DERControl(control_id="c", set_watts=100))
    assert [c.control_id for c in adapter.get_active_controls()] == ["c"]
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_discovery_walks_resource_tree(server):
    adapter = _adapter()
    announced = []

    async def on_msg(msg):
        announced.append(msg.topic)

    adapter.subscribe("*", on_msg)
    await adapter.connect()
    try:
        assert adapter.status == ProtocolStatus.CONNECTED
        assert adapter.mode == ProtocolMode.LIVE
        assert adapter.end_device_href == "/edev/1"
        assert adapter.poll_interval_s == 0  # configured override
        assert adapter._server_poll_rate_s == 60.0  # DERProgramList pollRate
        assert 100 < adapter.server_now() - time.time() < 140

        programs = adapter.list_programs()
        assert [p.program_id for p in programs] == [
            "B1000000000000000000000000000002",
            "B1000000000000000000000000000001",
        ]
        default = adapter.get_program("B1000000000000000000000000000001").default_control
        assert default.connect is True
        assert default.max_limit_pct == 100.0
        assert default.set_gradient_w_per_s == 2.0

        active = adapter.get_active_controls()
        assert [c.control_id for c in active] == ["C-CURTAIL", "C-TARGET"]
        curtail, target = active
        assert curtail.gen_limit_w == 5000
        assert curtail.max_limit_pct == 50.0
        assert curtail.event_status == EventStatusCode.ACTIVE
        assert curtail.response_required == 3
        assert curtail.reply_to == "/rsps/1/rsp"
        assert target.set_watts == -3000
        assert target.set_pf == pytest.approx(0.95)
        assert target.ramp_time_s == 5.0
        assert target.randomize_start_s == 30
        assert DERControlMode.OP_MOD_FIXED_PF in target.modes
        assert DERControlMode.CHARGE in target.modes

        assert sorted(announced) == sorted(
            [
                "ieee2030_5/control/B1000000000000000000000000000002/C-CURTAIL",
                "ieee2030_5/control/B1000000000000000000000000000001/C-TARGET",
            ]
        )

        # Lists are fetched with explicit paging parameters, sep+xml accepted
        edev_call = next(c for c in server.calls if c.request.url.path == "/edev")
        assert edev_call.request.url.params["s"] == "0"
        assert edev_call.request.url.params["l"] == "255"
        assert edev_call.request.headers["accept"] == "application/sep+xml"

        # A second discovery with nothing new does not re-announce
        await adapter.discover()
        assert len(announced) == 2
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_future_control_becomes_active(server):
    adapter = _adapter()
    await adapter.connect()
    try:
        later = adapter.server_now() + 7300
        ids = [c.control_id for c in adapter.get_active_controls(now=later)]
        assert ids == ["C-FUTURE"]
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_list_paging(server):
    page1 = derc_list("/derp/1/derc", [derc_item("P1", 0, 10, 0, "")]).replace(
        'all="1"', 'all="2"'
    )
    page2 = derc_list("/derp/1/derc", [derc_item("P2", 0, 10, 0, "")]).replace(
        'all="1"', 'all="2"'
    )
    route = server.get("/derp/1/derc").mock(side_effect=[sep(page1), sep(page2)])
    adapter = _adapter(list_page_size=1)
    await adapter.connect()
    try:
        program = adapter.get_program("B1000000000000000000000000000002")
        assert [c.control_id for c in program.active_controls] == ["P1", "P2"]
        assert [c.request.url.params["s"] for c in route.calls] == ["0", "1"]
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_end_device_identity_required_when_ambiguous(server):
    adapter = IEEE2030_5Adapter()
    adapter.configure(server_url=SERVER, poll_interval_s=0)
    with pytest.raises(ConnectionError, match="LFDI"):
        await adapter.connect()
    assert adapter.status == ProtocolStatus.ERROR


@pytest.mark.asyncio
async def test_match_by_sfdi(server):
    adapter = IEEE2030_5Adapter()
    adapter.configure(server_url=SERVER, sfdi=sfdi_from_lfdi(LFDI), poll_interval_s=0)
    await adapter.connect()
    assert adapter.end_device_href == "/edev/1"
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_server_errors_are_reported():
    with respx.mock(base_url=SERVER) as mock:
        mock.get("/dcap").mock(return_value=httpx.Response(403))
        adapter = _adapter()
        with pytest.raises(ConnectionError, match="HTTP 403"):
            await adapter.connect()
    with respx.mock(base_url=SERVER) as mock:
        mock.get("/dcap").mock(side_effect=httpx.ConnectTimeout("slow"))
        adapter = _adapter()
        with pytest.raises(ConnectionError, match="failed"):
            await adapter.connect()
    with respx.mock(base_url=SERVER) as mock:
        mock.get("/dcap").mock(return_value=sep("<broken"))
        adapter = _adapter()
        with pytest.raises(ConnectionError, match="Malformed"):
            await adapter.connect()


@pytest.mark.asyncio
async def test_poll_loop_backoff_and_recovery(server, monkeypatch):
    import vpp.protocols.ieee2030_5 as mod

    adapter = _adapter(max_backoff_s=100)
    await adapter.connect()
    adapter.configure(poll_interval_s=10)
    server.get("/dcap").mock(
        side_effect=[
            httpx.ConnectError("down"),
            httpx.ConnectError("down"),
            sep(dcap()),
            sep(dcap()),
        ]
    )
    sleeps: list[float] = []
    real_sleep = mod.asyncio.sleep

    async def fake_sleep(delay):
        sleeps.append(delay)
        if len(sleeps) > 4:
            raise mod.asyncio.CancelledError
        await real_sleep(0)

    monkeypatch.setattr(mod.asyncio, "sleep", fake_sleep)
    statuses = []
    orig = adapter.discover

    async def tracking():
        try:
            return await orig()
        finally:
            statuses.append(adapter.status)

    adapter.discover = tracking
    with pytest.raises(mod.asyncio.CancelledError):
        await adapter._poll_loop()
    monkeypatch.setattr(mod.asyncio, "sleep", real_sleep)
    assert sleeps == [10.0, 10.0, 20.0, 10.0, 10.0]
    assert adapter.status == ProtocolStatus.CONNECTED
    assert adapter.metrics.reconnect_count == 1
    assert ProtocolStatus.RECONNECTING in statuses
    await adapter.disconnect()


def test_identity_helpers():
    lfdi = lfdi_from_cert_der(b"not really a certificate")
    assert len(lfdi) == 40 and lfdi == lfdi.upper()
    sfdi = sfdi_from_lfdi("3E4F45AB31EDFE5B67E343E5E4562E31984E23E5")
    # Worked example from IEEE 2030.5-2018 section 6.3.4
    assert sfdi == 167261211391


def test_control_activity_rules():
    now = 1_000_000.0
    assert DERControl(start_time=now - 10, duration_seconds=60).is_active_at(now)
    assert not DERControl(start_time=now + 10, duration_seconds=60).is_active_at(now)
    assert not DERControl(start_time=now - 100, duration_seconds=60).is_active_at(now)
    assert not DERControl(
        start_time=now - 10, duration_seconds=60, event_status=EventStatusCode.SUPERSEDED
    ).is_active_at(now)


# ---------------------------------------------------------------------------
# Response resources (received / started / completed / cancelled)
# ---------------------------------------------------------------------------


def _posted(server) -> list[tuple[str, int]]:
    from lxml import etree

    out = []
    for call in server.routes["responses"].calls:
        assert call.request.headers["content-type"] == "application/sep+xml"
        doc = etree.fromstring(call.request.content)
        assert doc.tag == "{urn:ieee:std:2030.5:ns}DERControlResponse"
        values = {etree.QName(child).localname: child.text for child in doc}
        assert list(values) == ["createdDateTime", "endDeviceLFDI", "status", "subject"]
        assert values["endDeviceLFDI"] == LFDI
        out.append((values["subject"], int(values["status"])))
    return out


def _server_time(server, offset_s: int) -> None:
    now = int(time.time())
    server.get("/tm").mock(
        return_value=sep(
            f'<Time {NS} href="/tm"><currentTime>{now + offset_s}</currentTime>'
            "<quality>7</quality><tzOffset>0</tzOffset></Time>"
        )
    )


@pytest.mark.asyncio
async def test_responses_posted_for_received_started_cancelled(server):
    adapter = _adapter()
    await adapter.connect()
    try:
        posted = _posted(server)
        assert sorted(posted) == sorted(
            [
                ("C-CURTAIL", 1),
                ("C-CURTAIL", 2),
                ("C-TARGET", 1),
                ("C-TARGET", 2),
                ("C-FUTURE", 1),  # received, not started yet
                ("C-CANCELLED", 1),
                ("C-CANCELLED", 6),
            ]
        )
        # Each status is posted exactly once across polls.
        await adapter.discover()
        assert len(_posted(server)) == len(posted)
        assert adapter.metrics.errors == 0
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_completed_response_after_interval_even_if_server_drops_control(server):
    adapter = _adapter()
    await adapter.connect()
    try:
        # C-TARGET (1800 s) is over; C-CURTAIL (3600 s) still running.
        _server_time(server, 2400)
        server.get("/derp/0/derc").mock(return_value=sep(derc_list("/derp/0/derc", [])))
        await adapter.discover()
        new = _posted(server)[7:]
        assert new == [("C-TARGET", 3)]
        _server_time(server, 4000)
        await adapter.discover()
        assert _posted(server)[-1] == ("C-CURTAIL", 3)
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_failed_response_post_is_retried_next_poll(server):
    server.routes["responses"].mock(side_effect=[httpx.Response(500)] + [httpx.Response(201)] * 20)
    adapter = _adapter()
    await adapter.connect()
    try:
        assert adapter.metrics.errors == 1
        first = len(adapter.responses_sent)
        await adapter.discover()
        assert len(adapter.responses_sent) == first + 1  # the failed one, retried
        assert len({(r["control_id"], r["status"]) for r in adapter.responses_sent}) == 7
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_no_responses_in_simulated_mode_or_when_not_required():
    adapter = IEEE2030_5Adapter()
    adapter.configure(poll_interval_s=0)
    await adapter.connect()
    control = DERControl(control_id="c", reply_to="/rsp", response_required=3)
    assert await adapter.post_response(control, 1) is False  # no server
    assert adapter.responses_sent == []


@pytest.mark.asyncio
async def test_full_buffer_drops_oldest_instead_of_counting_errors():
    adapter = IEEE2030_5Adapter()
    adapter.configure(poll_interval_s=0)
    await adapter.connect()
    adapter.register_program(DERProgram(program_id="p"))
    for i in range(510):
        await adapter.apply_control("p", DERControl(control_id=f"c{i}"))
    assert adapter.metrics.errors == 0
    first = await adapter.receive()
    assert first is not None and first.payload["control_id"] == "c10"
