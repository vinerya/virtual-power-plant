"""OpenADR 2.0b VEN tests against a respx-mocked VTN with realistic payloads."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import httpx
import pytest
import respx

from vpp.protocols import openadr_xml as ox
from vpp.protocols._transport import (
    backoff_delay,
    build_ssl_context,
    format_iso8601_duration,
    parse_iso8601_datetime,
    parse_iso8601_duration,
)
from vpp.protocols.base import ProtocolMode, ProtocolStatus
from vpp.protocols.openadr import (
    DREvent,
    DREventStatus,
    DRResponse,
    DRSignalType,
    OpenADRAdapter,
)

VTN = "https://vtn.example.com/OpenADR2/Simple/2.0b"
NS = {"oadr": ox.OADR, "ei": ox.EI, "pyld": ox.PYLD, "emix": ox.EMIX, "xcal": ox.XCAL}

HEADER = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<oadr:oadrPayload xmlns:oadr="http://openadr.org/oadr-2.0b/2012/07" '
    'xmlns:ei="http://docs.oasis-open.org/ns/energyinterop/201110" '
    'xmlns:pyld="http://docs.oasis-open.org/ns/energyinterop/201110/payloads" '
    'xmlns:emix="http://docs.oasis-open.org/ns/emix/2011/06" '
    'xmlns:xcal="urn:ietf:params:xml:ns:icalendar-2.0" '
    'xmlns:strm="urn:ietf:params:xml:ns:icalendar-2.0:stream">'
    "<oadr:oadrSignedObject>"
)
FOOTER = "</oadr:oadrSignedObject></oadr:oadrPayload>"


def created_party_registration(
    code: int = 200, ven_id: str | None = "VEN-123", registration_id: str | None = "REG-9"
) -> str:
    ids = ""
    if registration_id:
        ids += f"<ei:registrationID>{registration_id}</ei:registrationID>"
    if ven_id:
        ids += f"<ei:venID>{ven_id}</ei:venID>"
    return (
        HEADER
        + f"""
<oadr:oadrCreatedPartyRegistration ei:schemaVersion="2.0b">
  <ei:eiResponse>
    <ei:responseCode>{code}</ei:responseCode>
    <ei:responseDescription>{"OK" if code == 200 else "ERROR"}</ei:responseDescription>
    <pyld:requestID>req-1</pyld:requestID>
  </ei:eiResponse>
  {ids}
  <ei:vtnID>VTN-UTILITY</ei:vtnID>
  <oadr:oadrProfiles>
    <oadr:oadrProfile>
      <oadr:oadrProfileName>2.0b</oadr:oadrProfileName>
      <oadr:oadrTransports><oadr:oadrTransport>
        <oadr:oadrTransportName>simpleHttp</oadr:oadrTransportName>
      </oadr:oadrTransport></oadr:oadrTransports>
    </oadr:oadrProfile>
  </oadr:oadrProfiles>
  <oadr:oadrRequestedOadrPollFreq><xcal:duration>PT15S</xcal:duration></oadr:oadrRequestedOadrPollFreq>
</oadr:oadrCreatedPartyRegistration>"""
        + FOOTER
    )


def oadr_response(code: int = 200) -> str:
    return (
        HEADER
        + f"""
<oadr:oadrResponse ei:schemaVersion="2.0b">
  <ei:eiResponse>
    <ei:responseCode>{code}</ei:responseCode>
    <ei:responseDescription>OK</ei:responseDescription>
    <pyld:requestID/>
  </ei:eiResponse>
  <ei:venID>VEN-123</ei:venID>
</oadr:oadrResponse>"""
        + FOOTER
    )


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def oadr_event(
    event_id: str,
    *,
    mod: int = 0,
    status: str = "far",
    start: datetime | None = None,
    duration: str = "PT1H",
    signal_name: str = "SIMPLE",
    signal_type: str = "level",
    values: tuple[float, ...] = (1.0, 2.0),
    current: float | None = None,
    response_required: str = "always",
) -> str:
    start = start or datetime.now(timezone.utc) + timedelta(hours=1)
    intervals = "".join(
        f"""<ei:interval>
              <xcal:duration><xcal:duration>PT30M</xcal:duration></xcal:duration>
              <xcal:uid><xcal:text>{i}</xcal:text></xcal:uid>
              <ei:signalPayload><ei:payloadFloat><ei:value>{v}</ei:value></ei:payloadFloat></ei:signalPayload>
            </ei:interval>"""
        for i, v in enumerate(values)
    )
    current_xml = (
        f"<ei:currentValue><ei:payloadFloat><ei:value>{current}</ei:value></ei:payloadFloat></ei:currentValue>"
        if current is not None
        else ""
    )
    return f"""
<oadr:oadrEvent>
  <ei:eiEvent>
    <ei:eventDescriptor>
      <ei:eventID>{event_id}</ei:eventID>
      <ei:modificationNumber>{mod}</ei:modificationNumber>
      <ei:priority>1</ei:priority>
      <ei:eiMarketContext><emix:marketContext>http://utility.example.com/dr</emix:marketContext></ei:eiMarketContext>
      <ei:createdDateTime>{_iso(datetime.now(timezone.utc))}</ei:createdDateTime>
      <ei:eventStatus>{status}</ei:eventStatus>
      <ei:testEvent>false</ei:testEvent>
    </ei:eventDescriptor>
    <ei:eiActivePeriod>
      <xcal:properties>
        <xcal:dtstart><xcal:date-time>{_iso(start)}</xcal:date-time></xcal:dtstart>
        <xcal:duration><xcal:duration>{duration}</xcal:duration></xcal:duration>
      </xcal:properties>
      <xcal:components xsi:nil="true" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"/>
    </ei:eiActivePeriod>
    <ei:eiEventSignals>
      <ei:eiEventSignal>
        <strm:intervals>{intervals}</strm:intervals>
        <ei:signalName>{signal_name}</ei:signalName>
        <ei:signalType>{signal_type}</ei:signalType>
        <ei:signalID>SIG-1</ei:signalID>
        {current_xml}
      </ei:eiEventSignal>
    </ei:eiEventSignals>
    <ei:eiTarget>
      <ei:venID>VEN-123</ei:venID>
      <ei:resourceID>battery-1</ei:resourceID>
    </ei:eiTarget>
  </ei:eiEvent>
  <oadr:oadrResponseRequired>{response_required}</oadr:oadrResponseRequired>
</oadr:oadrEvent>"""


def distribute_event(*events: str, request_id: str = "DIST-1") -> str:
    return (
        HEADER
        + f"""
<oadr:oadrDistributeEvent ei:schemaVersion="2.0b">
  <ei:eiResponse>
    <ei:responseCode>200</ei:responseCode>
    <ei:responseDescription>OK</ei:responseDescription>
    <pyld:requestID/>
  </ei:eiResponse>
  <pyld:requestID>{request_id}</pyld:requestID>
  <ei:vtnID>VTN-UTILITY</ei:vtnID>
  {"".join(events)}
</oadr:oadrDistributeEvent>"""
        + FOOTER
    )


def simple_vtn_message(tag: str) -> str:
    return (
        HEADER
        + f"""
<oadr:{tag} ei:schemaVersion="2.0b">
  <pyld:requestID>VTN-REQ-7</pyld:requestID>
  <ei:registrationID>REG-9</ei:registrationID>
  <ei:venID>VEN-123</ei:venID>
</oadr:{tag}>"""
        + FOOTER
    )


def _body(request: httpx.Request):
    return ox.parse_payload(request.content)


def _xml(text: str) -> httpx.Response:
    return httpx.Response(200, content=text.encode(), headers={"Content-Type": "application/xml"})


@pytest.fixture
def vtn():
    with respx.mock(base_url=VTN, assert_all_called=False) as mock:
        mock.post("/EiRegisterParty").mock(return_value=_xml(created_party_registration()))
        mock.post("/EiEvent").mock(return_value=_xml(oadr_response()))
        mock.post("/EiReport").mock(return_value=_xml(oadr_response()))
        mock.post("/OadrPoll").mock(return_value=_xml(oadr_response()))
        yield mock


def _adapter(**config) -> OpenADRAdapter:
    adapter = OpenADRAdapter()
    adapter.configure(role="ven", vtn_url=VTN, ven_name="vpp-test", poll_interval_s=0, **config)
    return adapter


# ---------------------------------------------------------------------------
# Mode honesty
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_without_vtn_url_is_simulated():
    adapter = OpenADRAdapter()
    adapter.configure(role="ven", poll_interval_s=0)
    await adapter.connect()
    assert adapter.status == ProtocolStatus.SIMULATED
    assert adapter.mode == ProtocolMode.SIMULATED
    assert not adapter.is_connected
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_vtn_role_is_simulated_even_with_url():
    adapter = OpenADRAdapter()
    adapter.configure(role="vtn", vtn_url=VTN)
    await adapter.connect()
    assert adapter.status == ProtocolStatus.SIMULATED


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_registration_flow(vtn):
    adapter = _adapter()
    await adapter.connect()
    try:
        assert adapter.status == ProtocolStatus.CONNECTED
        assert adapter.mode == ProtocolMode.LIVE
        assert adapter.ven_id == "VEN-123"
        assert adapter.registration_id == "REG-9"
        assert adapter.vtn_id == "VTN-UTILITY"

        calls = list(vtn.calls)
        assert [c.request.url.path.rsplit("/", 1)[-1] for c in calls] == [
            "EiRegisterParty",
            "EiRegisterParty",
        ]
        query = _body(calls[0].request)
        assert ox.message_name(query) == "oadrQueryRegistration"
        assert query.get(f"{{{ox.EI}}}schemaVersion") == "2.0b"
        create = _body(calls[1].request)
        assert ox.message_name(create) == "oadrCreatePartyRegistration"
        assert create.findtext("oadr:oadrVenName", namespaces=NS) == "vpp-test"
        assert create.findtext("oadr:oadrProfileName", namespaces=NS) == "2.0b"
        assert create.findtext("oadr:oadrTransportName", namespaces=NS) == "simpleHttp"
        assert create.findtext("oadr:oadrHttpPullModel", namespaces=NS) == "true"
        assert calls[1].request.headers["content-type"] == "application/xml"
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_poll_interval_uses_vtn_requested_frequency(vtn):
    adapter = OpenADRAdapter()
    adapter.configure(role="ven", vtn_url=VTN)
    adapter._client = httpx.AsyncClient()
    try:
        await adapter.register()
        assert adapter.poll_interval_s == 15.0
        adapter.configure(poll_interval_s=5)
        assert adapter.poll_interval_s == 5.0
    finally:
        await adapter._client.aclose()


@pytest.mark.asyncio
async def test_registration_rejected_sets_error(vtn):
    vtn.post("/EiRegisterParty").mock(return_value=_xml(created_party_registration(code=452)))
    adapter = _adapter()
    with pytest.raises(ox.OpenADRError) as info:
        await adapter.connect()
    assert info.value.code == 452
    assert adapter.status == ProtocolStatus.ERROR
    assert adapter._client is None


@pytest.mark.asyncio
async def test_unreachable_vtn_raises_transport_error():
    with respx.mock(base_url=VTN) as mock:
        mock.post("/EiRegisterParty").mock(side_effect=httpx.ConnectError("refused"))
        adapter = _adapter()
        with pytest.raises(ConnectionError):
            await adapter.connect()
        assert adapter.status == ProtocolStatus.ERROR


@pytest.mark.asyncio
async def test_http_error_status_raises(vtn):
    vtn.post("/EiRegisterParty").mock(return_value=httpx.Response(503))
    adapter = _adapter()
    with pytest.raises(ConnectionError, match="HTTP 503"):
        await adapter.connect()


# ---------------------------------------------------------------------------
# Polling + events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_poll_distribute_event_and_opt_in(vtn):
    start = datetime.now(timezone.utc) + timedelta(hours=2)
    vtn.post("/OadrPoll").mock(
        side_effect=[
            _xml(
                distribute_event(
                    oadr_event(
                        "EVT-1",
                        start=start,
                        signal_name="ELECTRICITY_PRICE",
                        signal_type="price",
                        values=(0.45, 0.60),
                    ),
                    oadr_event("EVT-2", start=start, response_required="never"),
                )
            ),
            _xml(oadr_response()),
        ]
    )
    adapter = _adapter()
    received = []

    async def on_msg(msg):
        received.append(msg)

    adapter.subscribe("*", on_msg)
    await adapter.connect()
    try:
        handled = await adapter.poll_once()
        assert handled == 1

        evt = adapter.get_event("EVT-1")
        assert evt is not None
        assert evt.signal_type == DRSignalType.ELECTRICITY_PRICE
        assert evt.signal_level == pytest.approx(0.45)
        assert evt.status == DREventStatus.PENDING
        assert evt.start_time == pytest.approx(int(start.timestamp()))
        assert evt.duration_seconds == 3600
        assert evt.market_context == "http://utility.example.com/dr"
        assert evt.resource_ids == ["battery-1"]
        intervals = evt.metadata["signals"][0]["intervals"]
        assert [i["value"] for i in intervals] == [0.45, 0.60]
        assert intervals[1]["start_time"] - intervals[0]["start_time"] == 1800
        assert adapter.get_event("EVT-2") is not None
        assert {m.topic for m in received} >= {"openadr/event/EVT-1", "openadr/event/EVT-2"}

        # Exactly one oadrCreatedEvent, answering only the response-required event
        created_calls = [c for c in vtn.calls if c.request.url.path.endswith("/EiEvent")]
        assert len(created_calls) == 1
        created = _body(created_calls[0].request)
        assert ox.message_name(created) == "oadrCreatedEvent"
        responses = created.findall(".//ei:eventResponse", NS)
        assert len(responses) == 1
        er = responses[0]
        assert er.findtext("ei:qualifiedEventID/ei:eventID", namespaces=NS) == "EVT-1"
        assert er.findtext("ei:qualifiedEventID/ei:modificationNumber", namespaces=NS) == "0"
        assert er.findtext("ei:optType", namespaces=NS) == "optIn"
        assert er.findtext("pyld:requestID", namespaces=NS) == "DIST-1"
        assert created.findtext("pyld:eiCreatedEvent/ei:venID", namespaces=NS) == "VEN-123"

        poll_calls = [c for c in vtn.calls if c.request.url.path.endswith("/OadrPoll")]
        poll = _body(poll_calls[0].request)
        assert poll.findtext("ei:venID", namespaces=NS) == "VEN-123"
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_handler_opt_out_and_modification_and_implicit_cancel(vtn):
    now = datetime.now(timezone.utc)
    vtn.post("/OadrPoll").mock(
        side_effect=[
            _xml(
                distribute_event(
                    oadr_event(
                        "EVT-A",
                        start=now - timedelta(minutes=5),
                        status="active",
                        current=3.0,
                        signal_name="LOAD_DISPATCH",
                        signal_type="setpoint",
                    )
                )
            ),
            _xml(oadr_response()),
            # identical revision: no new response
            _xml(
                distribute_event(
                    oadr_event(
                        "EVT-A",
                        start=now - timedelta(minutes=5),
                        status="active",
                        current=3.0,
                        signal_name="LOAD_DISPATCH",
                        signal_type="setpoint",
                    )
                )
            ),
            _xml(oadr_response()),
            # modified revision -> handled again
            _xml(
                distribute_event(
                    oadr_event(
                        "EVT-A",
                        mod=1,
                        start=now - timedelta(minutes=5),
                        status="active",
                        current=1.0,
                        signal_name="LOAD_DISPATCH",
                        signal_type="setpoint",
                    )
                )
            ),
            _xml(oadr_response()),
            # event gone -> implicitly cancelled
            _xml(distribute_event()),
            _xml(oadr_response()),
        ]
    )
    adapter = _adapter()
    seen = []

    async def handler(event: DREvent) -> DRResponse:
        seen.append((event.event_id, event.metadata["modification_number"]))
        return DRResponse(event_id=event.event_id, opt_type="optOut")

    adapter.register_event_handler(handler)
    await adapter.connect()
    try:
        await adapter.poll_once()
        evt = adapter.get_event("EVT-A")
        assert evt.status == DREventStatus.ACTIVE
        assert evt.is_active
        assert evt.signal_type == DRSignalType.LOAD_DISPATCH
        assert evt.signal_level == 3.0
        assert adapter.get_active_events() == [evt]

        await adapter.poll_once()
        await adapter.poll_once()
        assert seen == [("EVT-A", 0), ("EVT-A", 1)]
        assert adapter.get_event("EVT-A").signal_level == 1.0

        created = [c for c in vtn.calls if c.request.url.path.endswith("/EiEvent")]
        assert len(created) == 2
        last = _body(created[-1].request)
        assert last.findtext(".//ei:optType", namespaces=NS) == "optOut"
        assert last.findtext(".//ei:modificationNumber", namespaces=NS) == "1"

        await adapter.poll_once()
        assert adapter.get_event("EVT-A").status == DREventStatus.CANCELLED
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_send_opt_changes_opt_state(vtn):
    vtn.post("/OadrPoll").mock(
        side_effect=[
            _xml(distribute_event(oadr_event("EVT-X"))),
            _xml(oadr_response()),
        ]
    )
    adapter = _adapter()
    await adapter.connect()
    try:
        await adapter.poll_once()
        await adapter.send_opt("EVT-X", "optOut")
        assert adapter.get_response("EVT-X").opt_type == "optOut"
        created = [c for c in vtn.calls if c.request.url.path.endswith("/EiEvent")]
        assert _body(created[-1].request).findtext(".//ei:optType", namespaces=NS) == "optOut"
        with pytest.raises(ValueError):
            await adapter.send_opt("EVT-X", "maybe")
        with pytest.raises(ValueError):
            await adapter.send_opt("NOPE", "optIn")
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_reregistration_and_cancel_and_register_report(vtn):
    vtn.post("/OadrPoll").mock(
        side_effect=[
            _xml(simple_vtn_message("oadrRegisterReport")),
            _xml(simple_vtn_message("oadrRequestReregistration")),
            _xml(simple_vtn_message("oadrCancelPartyRegistration")),
            _xml(oadr_response()),
        ]
    )
    adapter = _adapter()
    await adapter.connect()
    try:
        before = len([c for c in vtn.calls if c.request.url.path.endswith("/EiRegisterParty")])
        assert await adapter.poll_once() == 3
        report = [c for c in vtn.calls if c.request.url.path.endswith("/EiReport")]
        assert ox.message_name(_body(report[0].request)) == "oadrRegisteredReport"
        reg_calls = [c for c in vtn.calls if c.request.url.path.endswith("/EiRegisterParty")]
        names = [ox.message_name(_body(c.request)) for c in reg_calls[before:]]
        assert names == [
            "oadrResponse",
            "oadrQueryRegistration",
            "oadrCreatePartyRegistration",
            "oadrCanceledPartyRegistration",
        ]
        canceled = _body(reg_calls[-1].request)
        assert canceled.findtext("ei:eiResponse/pyld:requestID", namespaces=NS) == "VTN-REQ-7"
        assert not adapter.registered
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_poll_loop_backs_off_and_recovers(vtn, monkeypatch):
    import vpp.protocols.openadr as openadr_mod

    sleeps: list[float] = []
    real_sleep = openadr_mod.asyncio.sleep

    async def fake_sleep(delay):
        sleeps.append(delay)
        if len(sleeps) >= 4:
            raise openadr_mod.asyncio.CancelledError
        await real_sleep(0)

    vtn.post("/OadrPoll").mock(
        side_effect=[
            httpx.ConnectError("down"),
            httpx.ConnectError("down"),
            _xml(oadr_response()),
            _xml(oadr_response()),
        ]
    )
    adapter = _adapter(max_backoff_s=60)
    await adapter.connect()
    adapter.configure(poll_interval_s=10)
    monkeypatch.setattr(openadr_mod.asyncio, "sleep", fake_sleep)
    statuses = []

    orig_poll = adapter.poll_once

    async def tracking_poll(*a, **kw):
        try:
            return await orig_poll(*a, **kw)
        finally:
            statuses.append(adapter.status)

    adapter.poll_once = tracking_poll
    with pytest.raises(openadr_mod.asyncio.CancelledError):
        await adapter._ven_poll_loop()
    monkeypatch.setattr(openadr_mod.asyncio, "sleep", real_sleep)
    assert sleeps == [10.0, 20.0, 10.0, 10.0]
    assert adapter.status == ProtocolStatus.CONNECTED
    assert adapter.metrics.reconnect_count == 1
    assert adapter.metrics.errors == 2
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_poll_error_code_triggers_reregistration(vtn):
    vtn.post("/OadrPoll").mock(side_effect=[_xml(oadr_response(code=452)), _xml(oadr_response())])
    adapter = _adapter()
    await adapter.connect()
    try:
        with pytest.raises(ox.OpenADRError) as info:
            await adapter.poll_once()
        assert info.value.code == 452
    finally:
        await adapter.disconnect()


# ---------------------------------------------------------------------------
# XML + helpers
# ---------------------------------------------------------------------------


def test_parse_rejects_malformed_and_xxe():
    with pytest.raises(ox.OpenADRError):
        ox.parse_payload(b"<not-closed")
    with pytest.raises(ox.OpenADRError):
        ox.parse_payload(b"<foo/>")
    xxe = (
        b'<?xml version="1.0"?><!DOCTYPE r [<!ENTITY x SYSTEM "file:///etc/passwd">]>'
        + (
            HEADER.split("\n", 1)[1] + "<oadr:oadrResponse><ei:venID>&x;</ei:venID>"
            "</oadr:oadrResponse>" + FOOTER
        ).encode()
    )
    msg = ox.parse_payload(xxe)
    assert "root:" not in (msg.findtext("ei:venID", namespaces=NS) or "")


def test_open_ended_event_and_load_percentage():
    from vpp.protocols.openadr import dr_event_from_parsed

    msg = ox.parse_payload(
        distribute_event(
            oadr_event(
                "OPEN",
                duration="PT0S",
                values=(),
                signal_name="LOAD_CONTROL",
                signal_type="x-loadControlPercentOffset",
            )
        )
    )
    parsed = ox.parse_distribute_event(msg).events[0]
    evt = dr_event_from_parsed(parsed)
    assert evt.metadata["open_ended"] is True
    assert evt.duration_seconds == ox.OPEN_ENDED_DURATION_S
    assert evt.signal_type == DRSignalType.LOAD_PERCENTAGE


def test_duration_and_datetime_helpers():
    assert parse_iso8601_duration("PT1H30M") == 5400
    assert parse_iso8601_duration("P1DT1S") == 86401
    assert parse_iso8601_duration("-PT5M") == -300
    assert parse_iso8601_duration("PT0.5S") == 0.5
    for bad in ("", "P", "PT", "1H", "P1Y"):
        with pytest.raises(ValueError):
            parse_iso8601_duration(bad)
    assert format_iso8601_duration(5400) == "PT1H30M"
    assert format_iso8601_duration(0) == "PT0S"
    assert parse_iso8601_datetime("2026-01-01T00:00:00Z") == 1767225600
    assert parse_iso8601_datetime("2026-01-01T00:00:00.1234Z") == pytest.approx(1767225600.1234)
    assert parse_iso8601_datetime("2026-01-01T01:00:00+01:00") == 1767225600
    assert backoff_delay(0) == 0
    assert backoff_delay(1, base=2) == 2
    assert backoff_delay(10, base=2, maximum=30) == 30
    assert build_ssl_context(verify=False) is False
    assert build_ssl_context() is not False


def test_simulated_event_state_advances():
    adapter = OpenADRAdapter()
    evt = DREvent(event_id="S", start_time=time.time() - 10, duration_seconds=5)
    adapter._events["S"] = evt
    adapter._advance_event_states()
    assert evt.status == DREventStatus.COMPLETED


@pytest.mark.asyncio
async def test_full_buffer_drops_oldest_instead_of_counting_errors():
    adapter = OpenADRAdapter()
    adapter.configure(role="ven", poll_interval_s=0)
    await adapter.connect()
    for i in range(510):
        await adapter.handle_incoming_event(DREvent(event_id=f"E{i}", start_time=time.time()))
    assert adapter.metrics.errors == 0
    oldest = await adapter.receive()
    assert oldest is not None and oldest.payload["event_id"] == "E10"
