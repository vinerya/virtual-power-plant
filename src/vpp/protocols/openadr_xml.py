"""OpenADR 2.0b XML payload building and parsing (VEN side).

Covers the messages a simple-HTTP *pull* VEN exchanges with a VTN:

* VEN -> VTN: ``oadrQueryRegistration``, ``oadrCreatePartyRegistration``,
  ``oadrPoll``, ``oadrCreatedEvent``, ``oadrResponse``,
  ``oadrCanceledPartyRegistration``, ``oadrRegisteredReport``
* VTN -> VEN: ``oadrCreatedPartyRegistration``, ``oadrDistributeEvent``,
  ``oadrResponse``, ``oadrRequestReregistration``,
  ``oadrCancelPartyRegistration``, ``oadrRegisterReport``

XML signatures (``oadrXmlSignature``) are not supported; TLS client
certificates provide authentication, as in most 2.0b deployments.
Parsing uses a hardened lxml parser (no entity resolution, no network).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from lxml import etree

from vpp.protocols._transport import parse_iso8601_datetime, parse_iso8601_duration

OADR = "http://openadr.org/oadr-2.0b/2012/07"
EI = "http://docs.oasis-open.org/ns/energyinterop/201110"
PYLD = "http://docs.oasis-open.org/ns/energyinterop/201110/payloads"
EMIX = "http://docs.oasis-open.org/ns/emix/2011/06"
XCAL = "urn:ietf:params:xml:ns:icalendar-2.0"
STRM = "urn:ietf:params:xml:ns:icalendar-2.0:stream"

NSMAP = {"oadr": OADR, "ei": EI, "pyld": PYLD, "emix": EMIX, "xcal": XCAL, "strm": STRM}

SCHEMA_VERSION = "2.0b"
PROFILE_NAME = "2.0b"
TRANSPORT_NAME = "simpleHttp"

# Seconds used for events whose duration is "PT0S" (open-ended).
OPEN_ENDED_DURATION_S = 10 * 365 * 86400

_PARSER = etree.XMLParser(
    resolve_entities=False, no_network=True, remove_blank_text=True, huge_tree=False
)


class OpenADRError(Exception):
    """A VTN answered with a non-2xx ``ei:responseCode`` or a malformed payload."""

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message if code is None else f"{message} (responseCode {code})")
        self.code = code


def _q(ns: str, tag: str) -> str:
    return f"{{{ns}}}{tag}"


def _sub(parent: etree._Element, ns: str, tag: str, text: str | None = None) -> etree._Element:
    el = etree.SubElement(parent, _q(ns, tag))
    if text is not None:
        el.text = text
    return el


def new_request_id() -> str:
    return uuid.uuid4().hex


def _envelope(message_tag: str) -> tuple[etree._Element, etree._Element]:
    root = etree.Element(_q(OADR, "oadrPayload"), nsmap=NSMAP)
    signed = _sub(root, OADR, "oadrSignedObject")
    inner = _sub(signed, OADR, message_tag)
    inner.set(_q(EI, "schemaVersion"), SCHEMA_VERSION)
    return root, inner


def _serialise(root: etree._Element) -> bytes:
    return bytes(etree.tostring(root, xml_declaration=True, encoding="UTF-8"))


def _ei_response(parent: etree._Element, request_id: str, code: int = 200,
                 description: str = "OK") -> None:
    resp = _sub(parent, EI, "eiResponse")
    _sub(resp, EI, "responseCode", str(code))
    _sub(resp, EI, "responseDescription", description)
    _sub(resp, PYLD, "requestID", request_id)


# ---------------------------------------------------------------------------
# VEN -> VTN builders
# ---------------------------------------------------------------------------

def build_query_registration(request_id: str | None = None) -> bytes:
    root, inner = _envelope("oadrQueryRegistration")
    _sub(inner, PYLD, "requestID", request_id or new_request_id())
    return _serialise(root)


def build_create_party_registration(
    ven_name: str,
    *,
    ven_id: str | None = None,
    registration_id: str | None = None,
    request_id: str | None = None,
    report_only: bool = False,
) -> bytes:
    root, inner = _envelope("oadrCreatePartyRegistration")
    _sub(inner, PYLD, "requestID", request_id or new_request_id())
    if registration_id:
        _sub(inner, EI, "registrationID", registration_id)
    if ven_id:
        _sub(inner, EI, "venID", ven_id)
    _sub(inner, OADR, "oadrProfileName", PROFILE_NAME)
    _sub(inner, OADR, "oadrTransportName", TRANSPORT_NAME)
    _sub(inner, OADR, "oadrReportOnly", "true" if report_only else "false")
    _sub(inner, OADR, "oadrXmlSignature", "false")
    _sub(inner, OADR, "oadrVenName", ven_name)
    _sub(inner, OADR, "oadrHttpPullModel", "true")
    return _serialise(root)


def build_poll(ven_id: str) -> bytes:
    root, inner = _envelope("oadrPoll")
    _sub(inner, EI, "venID", ven_id)
    return _serialise(root)


@dataclass(frozen=True)
class EventOptResponse:
    event_id: str
    modification_number: int
    opt_type: str  # optIn | optOut
    request_id: str  # requestID of the oadrDistributeEvent being answered
    code: int = 200
    description: str = "OK"


def build_created_event(ven_id: str, responses: list[EventOptResponse]) -> bytes:
    root, inner = _envelope("oadrCreatedEvent")
    created = _sub(inner, PYLD, "eiCreatedEvent")
    _ei_response(created, "")
    event_responses = _sub(created, EI, "eventResponses")
    for r in responses:
        er = _sub(event_responses, EI, "eventResponse")
        _sub(er, EI, "responseCode", str(r.code))
        _sub(er, EI, "responseDescription", r.description)
        _sub(er, PYLD, "requestID", r.request_id)
        qid = _sub(er, EI, "qualifiedEventID")
        _sub(qid, EI, "eventID", r.event_id)
        _sub(qid, EI, "modificationNumber", str(r.modification_number))
        _sub(er, EI, "optType", r.opt_type)
    _sub(created, EI, "venID", ven_id)
    return _serialise(root)


def build_response(ven_id: str, request_id: str, code: int = 200, description: str = "OK") -> bytes:
    root, inner = _envelope("oadrResponse")
    _ei_response(inner, request_id, code, description)
    _sub(inner, EI, "venID", ven_id)
    return _serialise(root)


def build_canceled_party_registration(ven_id: str, registration_id: str, request_id: str) -> bytes:
    root, inner = _envelope("oadrCanceledPartyRegistration")
    _ei_response(inner, request_id)
    _sub(inner, EI, "registrationID", registration_id)
    _sub(inner, EI, "venID", ven_id)
    return _serialise(root)


def build_registered_report(ven_id: str, request_id: str) -> bytes:
    root, inner = _envelope("oadrRegisteredReport")
    _ei_response(inner, request_id)
    _sub(inner, EI, "venID", ven_id)
    return _serialise(root)


# ---------------------------------------------------------------------------
# VTN -> VEN parsing
# ---------------------------------------------------------------------------

def parse_payload(data: bytes | str) -> etree._Element:
    """Parse an ``oadrPayload`` document and return the signed message element."""
    if isinstance(data, str):
        data = data.encode()
    try:
        root = etree.fromstring(data, _PARSER)
    except etree.XMLSyntaxError as exc:
        raise OpenADRError(f"Malformed OpenADR XML: {exc}") from exc
    if root.tag != _q(OADR, "oadrPayload"):
        raise OpenADRError(f"Unexpected root element {root.tag}")
    signed = root.find("oadr:oadrSignedObject", NSMAP)
    if signed is None or len(signed) == 0:
        raise OpenADRError("oadrPayload has no oadrSignedObject message")
    return signed[0]


def message_name(message: etree._Element) -> str:
    return str(etree.QName(message).localname)


def _text(el: etree._Element | None, path: str, default: str | None = None) -> str | None:
    if el is None:
        return default
    found = el.find(path, NSMAP)
    if found is None or found.text is None:
        return default
    return str(found.text).strip()


def check_ei_response(message: etree._Element) -> None:
    """Raise :class:`OpenADRError` if the message's eiResponse is not 2xx."""
    code_text = _text(message, "ei:eiResponse/ei:responseCode")
    if code_text is None:
        return
    try:
        code = int(code_text)
    except ValueError as exc:
        raise OpenADRError(f"Non-numeric responseCode {code_text!r}") from exc
    if not 200 <= code < 300:
        desc = _text(message, "ei:eiResponse/ei:responseDescription", "") or ""
        raise OpenADRError(f"VTN rejected {message_name(message)}: {desc}", code)


def request_id_of(message: etree._Element) -> str:
    return (
        _text(message, "pyld:requestID")
        or _text(message, "ei:eiResponse/pyld:requestID")
        or ""
    )


@dataclass
class RegistrationInfo:
    registration_id: str | None
    ven_id: str | None
    vtn_id: str | None
    poll_interval_s: float | None


def parse_created_party_registration(message: etree._Element) -> RegistrationInfo:
    if message_name(message) != "oadrCreatedPartyRegistration":
        raise OpenADRError(f"Expected oadrCreatedPartyRegistration, got {message_name(message)}")
    check_ei_response(message)
    freq = _text(message, "oadr:oadrRequestedOadrPollFreq/xcal:duration")
    return RegistrationInfo(
        registration_id=_text(message, "ei:registrationID"),
        ven_id=_text(message, "ei:venID"),
        vtn_id=_text(message, "ei:vtnID"),
        poll_interval_s=parse_iso8601_duration(freq) if freq else None,
    )


def _payload_value(el: etree._Element | None) -> float | None:
    """Value of an ``ei:signalPayload`` / ``ei:currentValue`` payload element."""
    if el is None:
        return None
    for path in ("ei:payloadFloat/ei:value", "oadr:oadrPayloadResourceStatus"):
        text = _text(el, path)
        if text is not None:
            try:
                return float(text)
            except ValueError:
                return None
    return None


@dataclass
class ParsedSignal:
    signal_name: str
    signal_type: str
    signal_id: str
    current_value: float | None
    intervals: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ParsedEvent:
    event_id: str
    modification_number: int
    status: str  # far | near | active | completed | cancelled | none
    market_context: str
    start_time: float
    duration_s: float
    priority: int | None
    test_event: bool
    response_required: str  # always | never
    created_at: float | None
    signals: list[ParsedSignal]
    targets: dict[str, list[str]]


def _parse_signal(sig: etree._Element, start_time: float) -> ParsedSignal:
    intervals: list[dict[str, Any]] = []
    cursor = start_time
    for interval in sig.findall("strm:intervals/ei:interval", NSMAP):
        dur_text = _text(interval, "xcal:duration/xcal:duration")
        duration = parse_iso8601_duration(dur_text) if dur_text else 0.0
        value = _payload_value(interval.find("ei:signalPayload", NSMAP))
        intervals.append({
            "uid": _text(interval, "xcal:uid/xcal:text"),
            "start_time": cursor,
            "duration_seconds": duration,
            "value": value,
        })
        cursor += duration
    return ParsedSignal(
        signal_name=_text(sig, "ei:signalName", "simple") or "simple",
        signal_type=_text(sig, "ei:signalType", "level") or "level",
        signal_id=_text(sig, "ei:signalID", "") or "",
        current_value=_payload_value(sig.find("ei:currentValue", NSMAP)),
        intervals=intervals,
    )


def _parse_event(oadr_event: etree._Element) -> ParsedEvent:
    ei_event = oadr_event.find("ei:eiEvent", NSMAP)
    if ei_event is None:
        raise OpenADRError("oadrEvent without ei:eiEvent")
    desc = ei_event.find("ei:eventDescriptor", NSMAP)
    event_id = _text(desc, "ei:eventID")
    if not event_id:
        raise OpenADRError("eiEvent without eventID")
    props = ei_event.find("ei:eiActivePeriod/xcal:properties", NSMAP)
    dtstart = _text(props, "xcal:dtstart/xcal:date-time")
    if dtstart is None:
        raise OpenADRError(f"Event {event_id} has no dtstart")
    start_time = parse_iso8601_datetime(dtstart)
    duration_text = _text(props, "xcal:duration/xcal:duration", "PT0S") or "PT0S"
    duration = parse_iso8601_duration(duration_text)

    signals = [
        _parse_signal(sig, start_time)
        for sig in ei_event.findall("ei:eiEventSignals/ei:eiEventSignal", NSMAP)
    ]
    if duration <= 0:
        interval_total = sum(
            i["duration_seconds"] for s in signals[:1] for i in s.intervals
        )
        duration = interval_total if interval_total > 0 else 0.0

    targets: dict[str, list[str]] = {}
    target_el = ei_event.find("ei:eiTarget", NSMAP)
    if target_el is not None:
        for child in target_el:
            if isinstance(child.tag, str) and child.text and child.text.strip():
                targets.setdefault(etree.QName(child).localname, []).append(child.text.strip())

    created = _text(desc, "ei:createdDateTime")
    priority = _text(desc, "ei:priority")
    return ParsedEvent(
        event_id=event_id,
        modification_number=int(_text(desc, "ei:modificationNumber", "0") or 0),
        status=(_text(desc, "ei:eventStatus", "none") or "none").lower(),
        market_context=_text(desc, "ei:eiMarketContext/emix:marketContext", "") or "",
        start_time=start_time,
        duration_s=duration,
        priority=int(priority) if priority and priority.isdigit() else None,
        test_event=(_text(desc, "ei:testEvent", "false") or "false").lower() == "true",
        response_required=_text(oadr_event, "oadr:oadrResponseRequired", "always") or "always",
        created_at=parse_iso8601_datetime(created) if created else None,
        signals=signals,
        targets=targets,
    )


@dataclass
class DistributeEvent:
    request_id: str
    vtn_id: str
    events: list[ParsedEvent]
    received_at: float = field(default_factory=time.time)


def parse_distribute_event(message: etree._Element) -> DistributeEvent:
    if message_name(message) != "oadrDistributeEvent":
        raise OpenADRError(f"Expected oadrDistributeEvent, got {message_name(message)}")
    check_ei_response(message)
    return DistributeEvent(
        request_id=_text(message, "pyld:requestID", "") or "",
        vtn_id=_text(message, "ei:vtnID", "") or "",
        events=[_parse_event(e) for e in message.findall("oadr:oadrEvent", NSMAP)],
    )
