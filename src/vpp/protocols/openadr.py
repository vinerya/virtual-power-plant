"""OpenADR 2.0b adapter -- Virtual End Node (VEN) with a real HTTP pull client.

Modes (see :class:`~vpp.protocols.base.ProtocolMode`):

* **Live VEN** (``role="ven"`` and ``vtn_url`` configured): registers with
  the VTN (``oadrQueryRegistration`` + ``oadrCreatePartyRegistration`` on
  ``EiRegisterParty``), then polls ``OadrPoll`` on the VTN-requested (or
  configured) interval. ``oadrDistributeEvent`` payloads are parsed into
  :class:`DREvent` objects, run through the registered handlers, and
  answered with ``oadrCreatedEvent`` (opt-in/opt-out) when the VTN asks for
  a response. Transient failures back off exponentially; a VTN that forgets
  the VEN (``oadrRequestReregistration``, 452/463 response codes) triggers
  re-registration. Optional client TLS certificates enable mutual TLS.
  Status is ``CONNECTED`` while the VTN is reachable, ``RECONNECTING``
  while backing off.
* **Simulated** (no ``vtn_url``, or ``role="vtn"``): events are managed in
  memory only -- ``handle_incoming_event``/``publish_event`` still work,
  but nothing is exchanged with a real VTN. Status is ``SIMULATED``.
  (This package does not implement a VTN server.)

``vtn_url`` is the VTN's service prefix, conventionally ending in
``/OpenADR2/Simple/2.0b``; service names (``EiRegisterParty``, ``EiEvent``,
``OadrPoll``, ``EiReport``) are appended to it.

Not implemented: VEN report registration (``oadrRegisterReport`` from the
VEN), XML signatures, and the push (VTN->VEN HTTP) transport.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from vpp.protocols._transport import (
    ProtocolTransportError,
    backoff_delay,
    build_ssl_context,
)
from vpp.protocols.base import (
    ProtocolAdapter,
    ProtocolMessage,
    ProtocolMode,
    ProtocolStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenADR value objects
# ---------------------------------------------------------------------------


class DREventStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class DRSignalType(str, Enum):
    SIMPLE = "SIMPLE"  # 0/1/2/3 levels
    ELECTRICITY_PRICE = "ELECTRICITY_PRICE"
    LOAD_DISPATCH = "LOAD_DISPATCH"  # absolute kW target
    LOAD_CONTROL = "LOAD_CONTROL"  # delta kW
    LOAD_PERCENTAGE = "LOAD_PERCENTAGE"


@dataclass
class DREvent:
    """A demand-response event from or for the grid operator."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    signal_type: DRSignalType = DRSignalType.SIMPLE
    signal_level: float = 0.0
    start_time: float = 0.0
    duration_seconds: int = 3600
    status: DREventStatus = DREventStatus.PENDING
    market_context: str = "default"
    resource_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration_seconds

    @property
    def is_active(self) -> bool:
        now = time.time()
        return self.start_time <= now < self.end_time and self.status == DREventStatus.ACTIVE

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "signal_type": self.signal_type.value,
            "signal_level": self.signal_level,
            "start_time": self.start_time,
            "duration_seconds": self.duration_seconds,
            "status": self.status.value,
            "market_context": self.market_context,
            "resource_ids": self.resource_ids,
        }


@dataclass
class DRResponse:
    """VEN response to a DR event."""

    event_id: str
    opt_type: str = "optIn"  # optIn | optOut
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "opt_type": self.opt_type,
            "created_at": self.created_at,
        }


DREventHandler = Callable[[DREvent], Awaitable[DRResponse]]


# VTN responseCodes meaning "I don't know this VEN / registration any more".
_REREGISTER_CODES = {452, 463}

_SIGNAL_NAME_MAP = {
    "simple": DRSignalType.SIMPLE,
    "electricity_price": DRSignalType.ELECTRICITY_PRICE,
    "load_dispatch": DRSignalType.LOAD_DISPATCH,
    "load_control": DRSignalType.LOAD_CONTROL,
}

_STATUS_MAP = {
    "none": DREventStatus.PENDING,
    "far": DREventStatus.PENDING,
    "near": DREventStatus.PENDING,
    "active": DREventStatus.ACTIVE,
    "completed": DREventStatus.COMPLETED,
    "cancelled": DREventStatus.CANCELLED,
}


def dr_event_from_parsed(parsed: Any, *, vtn_id: str = "", request_id: str = "") -> DREvent:
    """Convert an :class:`vpp.protocols.openadr_xml.ParsedEvent` into a :class:`DREvent`.

    The first signal drives ``signal_type``/``signal_level`` (its
    ``currentValue`` if the VTN sent one, else the first interval's value);
    every signal and interval is kept in ``metadata["signals"]``.
    """
    from vpp.protocols.openadr_xml import OPEN_ENDED_DURATION_S

    primary = parsed.signals[0] if parsed.signals else None
    signal_type = DRSignalType.SIMPLE
    level = 0.0
    if primary is not None:
        signal_type = _SIGNAL_NAME_MAP.get(primary.signal_name.lower(), DRSignalType.SIMPLE)
        if (
            signal_type == DRSignalType.LOAD_CONTROL
            and primary.signal_type == "x-loadControlPercentOffset"
        ):
            signal_type = DRSignalType.LOAD_PERCENTAGE
        if primary.current_value is not None:
            level = primary.current_value
        elif primary.intervals and primary.intervals[0]["value"] is not None:
            level = primary.intervals[0]["value"]

    open_ended = parsed.duration_s <= 0
    return DREvent(
        event_id=parsed.event_id,
        signal_type=signal_type,
        signal_level=level,
        start_time=parsed.start_time,
        duration_seconds=OPEN_ENDED_DURATION_S if open_ended else int(parsed.duration_s),
        status=_STATUS_MAP.get(parsed.status, DREventStatus.PENDING),
        market_context=parsed.market_context or "default",
        resource_ids=list(parsed.targets.get("resourceID", [])),
        metadata={
            "source": "vtn",
            "vtn_id": vtn_id,
            "request_id": request_id,
            "modification_number": parsed.modification_number,
            "vtn_status": parsed.status,
            "priority": parsed.priority,
            "test_event": parsed.test_event,
            "response_required": parsed.response_required,
            "open_ended": open_ended,
            "targets": parsed.targets,
            "signals": [
                {
                    "signal_name": s.signal_name,
                    "signal_type": s.signal_type,
                    "signal_id": s.signal_id,
                    "current_value": s.current_value,
                    "intervals": s.intervals,
                }
                for s in parsed.signals
            ],
        },
    )


class OpenADRAdapter(ProtocolAdapter):
    """OpenADR 2.0b adapter (live VEN, or simulated VEN/VTN).

    Configuration keys:
        role ("ven" | "vtn"), vtn_url, ven_name, ven_id, registration_id,
        poll_interval_s (overrides the VTN-requested frequency; 0 disables
        polling), market_context, auto_opt_in, timeout_s, max_backoff_s,
        verify_tls, ca_path, cert_path, key_path
    """

    def __init__(self) -> None:
        super().__init__("openadr", "2.0b")
        self._events: dict[str, DREvent] = {}
        self._responses: dict[str, DRResponse] = {}
        self._event_handlers: list[DREventHandler] = []
        self._poll_task: asyncio.Task | None = None
        self._message_queue: asyncio.Queue[ProtocolMessage] = asyncio.Queue(maxsize=500)
        self._client: Any = None  # httpx.AsyncClient in live mode
        self.ven_id: str | None = None
        self.registration_id: str | None = None
        self.vtn_id: str | None = None
        self._vtn_poll_interval_s: float | None = None
        self._consecutive_failures = 0

    # -- Mode ----------------------------------------------------------------

    @property
    def live(self) -> bool:
        return self._config.get("role", "ven") == "ven" and bool(self._config.get("vtn_url"))

    @property
    def mode(self) -> ProtocolMode:
        return ProtocolMode.LIVE if self.live else ProtocolMode.SIMULATED

    @property
    def registered(self) -> bool:
        return self.ven_id is not None and self.registration_id is not None

    @property
    def poll_interval_s(self) -> float:
        configured = self._config.get("poll_interval_s")
        if configured is not None:
            return float(configured)
        return self._vtn_poll_interval_s or 10.0

    # -- Lifecycle -----------------------------------------------------------

    async def connect(self) -> None:
        role = self._config.get("role", "ven")
        if role not in ("vtn", "ven"):
            raise ValueError(f"Invalid OpenADR role: {role}")
        self._status = ProtocolStatus.CONNECTING

        if not self.live:
            reason = "VTN role is simulated" if role == "vtn" else "no vtn_url configured"
            self._enter_simulated_mode(reason)
            poll_interval = float(self._config.get("poll_interval_s", 30))
            if role == "ven" and poll_interval > 0:
                self._poll_task = asyncio.create_task(self._simulated_tick_loop(poll_interval))
            return

        self._client = self._build_client()
        self.ven_id = self._config.get("ven_id") or self.ven_id
        self.registration_id = self._config.get("registration_id") or self.registration_id
        try:
            await self.register()
        except Exception:
            self._status = ProtocolStatus.ERROR
            self._metrics.errors += 1
            await self._close_client()
            raise

        self._status = ProtocolStatus.CONNECTED
        self._metrics.connected_since = time.time()
        if self.poll_interval_s > 0:
            self._poll_task = asyncio.create_task(self._ven_poll_loop())
        logger.info(
            "OpenADR VEN registered with VTN %s (venID=%s, poll every %.0fs)",
            self.vtn_id,
            self.ven_id,
            self.poll_interval_s,
        )

    async def disconnect(self) -> None:
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._poll_task
        self._poll_task = None
        await self._close_client()
        self._status = ProtocolStatus.DISCONNECTED
        self._metrics.connected_since = None
        logger.info("OpenADR adapter disconnected")

    async def send(self, message: ProtocolMessage) -> None:
        """Send a DR event (simulated VTN) or DR response (VEN)."""
        if not self.is_operational:
            raise ConnectionError("OpenADR adapter not started")

        if message.topic.startswith("openadr/event"):
            event = DREvent(**message.payload)
            self._events[event.event_id] = event
            logger.info(
                "Published DR event %s (signal=%s)", event.event_id, event.signal_type.value
            )
        elif message.topic.startswith("openadr/response"):
            response = DRResponse(**message.payload)
            self._responses[response.event_id] = response
            if self.is_connected:
                await self.send_opt(response.event_id, response.opt_type)
            logger.info("Sent DR response for event %s: %s", response.event_id, response.opt_type)

        self._metrics.messages_sent += 1
        self._metrics.last_message_at = time.time()

    async def receive(self) -> ProtocolMessage | None:
        try:
            return self._message_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    # -- DR event management -------------------------------------------------

    def register_event_handler(self, handler: DREventHandler) -> None:
        """Register an async callback for incoming DR events."""
        self._event_handlers.append(handler)

    async def publish_event(self, event: DREvent) -> None:
        """Publish a DR event (simulated VTN role; in-process only)."""
        self._events[event.event_id] = event
        msg = ProtocolMessage(
            topic=f"openadr/event/{event.event_id}",
            payload=event.to_dict(),
            source="openadr",
        )
        await self._dispatch(msg)
        self._metrics.messages_sent += 1

    async def handle_incoming_event(self, event: DREvent) -> DRResponse:
        """Process a received DR event (VEN role).

        Runs registered handlers; defaults to auto opt-in if configured.
        """
        self._events[event.event_id] = event

        # Notify subscribers
        msg = ProtocolMessage(
            topic=f"openadr/event/{event.event_id}",
            payload=event.to_dict(),
            source="openadr",
        )
        await self._dispatch(msg)
        try:
            self._message_queue.put_nowait(msg)
        except asyncio.QueueFull:
            self._metrics.errors += 1

        # Run handlers
        for handler in self._event_handlers:
            try:
                response = await handler(event)
                self._responses[event.event_id] = response
                return response
            except Exception:
                logger.exception("DR event handler failed for %s", event.event_id)
                self._metrics.errors += 1

        # Default: auto opt-in
        auto = self._config.get("auto_opt_in", True)
        response = DRResponse(
            event_id=event.event_id,
            opt_type="optIn" if auto else "optOut",
        )
        self._responses[event.event_id] = response
        return response

    def get_active_events(self) -> list[DREvent]:
        """Return currently active DR events."""
        return [e for e in self._events.values() if e.is_active]

    def get_event(self, event_id: str) -> DREvent | None:
        return self._events.get(event_id)

    def get_response(self, event_id: str) -> DRResponse | None:
        return self._responses.get(event_id)

    def list_events(self) -> list[DREvent]:
        return list(self._events.values())

    # -- HTTP transport (live VEN) -------------------------------------------

    def _build_client(self) -> Any:
        import httpx

        verify = build_ssl_context(
            verify=bool(self._config.get("verify_tls", True)),
            ca_path=self._config.get("ca_path"),
            cert_path=self._config.get("cert_path"),
            key_path=self._config.get("key_path"),
        )
        return httpx.AsyncClient(
            verify=verify,
            timeout=httpx.Timeout(float(self._config.get("timeout_s", 10.0))),
            headers={"Content-Type": "application/xml"},
        )

    async def _close_client(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            finally:
                self._client = None

    def _service_url(self, service: str) -> str:
        return f"{str(self._config['vtn_url']).rstrip('/')}/{service}"

    async def _post(self, service: str, body: bytes) -> Any:
        """POST an oadrPayload and return the parsed response message element."""
        import httpx

        from vpp.protocols.openadr_xml import parse_payload

        if self._client is None:
            raise ProtocolTransportError("OpenADR VEN HTTP client not started")
        try:
            response = await self._client.post(self._service_url(service), content=body)
        except httpx.HTTPError as exc:
            raise ProtocolTransportError(f"VTN {service} request failed: {exc}") from exc
        self._metrics.messages_sent += 1
        self._metrics.last_message_at = time.time()
        if response.status_code >= 400:
            raise ProtocolTransportError(f"VTN {service} returned HTTP {response.status_code}")
        self._metrics.messages_received += 1
        return parse_payload(response.content)

    # -- Registration --------------------------------------------------------

    async def register(self) -> None:
        """Query and (re-)create the party registration with the VTN."""
        from vpp.protocols import openadr_xml as ox

        ven_name = str(self._config.get("ven_name") or "vpp-ven")

        query = await self._post("EiRegisterParty", ox.build_query_registration())
        info = ox.parse_created_party_registration(query)
        if info.vtn_id:
            self.vtn_id = info.vtn_id

        created = await self._post(
            "EiRegisterParty",
            ox.build_create_party_registration(
                ven_name, ven_id=self.ven_id, registration_id=self.registration_id
            ),
        )
        info = ox.parse_created_party_registration(created)
        if not info.ven_id or not info.registration_id:
            raise ox.OpenADRError("VTN registration response lacks venID/registrationID")
        self.ven_id = info.ven_id
        self.registration_id = info.registration_id
        self.vtn_id = info.vtn_id or self.vtn_id
        if info.poll_interval_s:
            self._vtn_poll_interval_s = info.poll_interval_s

    # -- Polling -------------------------------------------------------------

    async def poll_once(self, max_messages: int = 10) -> int:
        """Poll the VTN until it has nothing queued; return messages handled."""
        from vpp.protocols import openadr_xml as ox

        if not self.registered:
            await self.register()
        handled = 0
        for _ in range(max_messages):
            message = await self._post("OadrPoll", ox.build_poll(str(self.ven_id)))
            if ox.message_name(message) == "oadrResponse":
                ox.check_ei_response(message)
                break
            await self._handle_vtn_message(message)
            handled += 1
        self._advance_event_states()
        return handled

    async def _handle_vtn_message(self, message: Any) -> None:
        from vpp.protocols import openadr_xml as ox

        name = ox.message_name(message)
        request_id = ox.request_id_of(message)
        if name == "oadrDistributeEvent":
            await self._on_distribute_event(ox.parse_distribute_event(message))
        elif name == "oadrRequestReregistration":
            await self._post("EiRegisterParty", ox.build_response(str(self.ven_id), request_id))
            self.registration_id = None
            await self.register()
        elif name == "oadrCancelPartyRegistration":
            await self._post(
                "EiRegisterParty",
                ox.build_canceled_party_registration(
                    str(self.ven_id), str(self.registration_id), request_id
                ),
            )
            logger.warning("VTN cancelled our OpenADR registration; will re-register")
            self.registration_id = None
        elif name == "oadrRegisterReport":
            await self._post("EiReport", ox.build_registered_report(str(self.ven_id), request_id))
        else:
            logger.info("Ignoring unsupported OpenADR message %s from VTN", name)

    async def _on_distribute_event(self, dist: Any) -> None:
        from vpp.protocols import openadr_xml as ox

        opt_responses: list[ox.EventOptResponse] = []
        seen: set[str] = set()
        for parsed in dist.events:
            seen.add(parsed.event_id)
            existing = self._events.get(parsed.event_id)
            if (
                existing is not None
                and existing.metadata.get("modification_number") == parsed.modification_number
            ):
                # Same revision: only the VTN-reported status may have moved.
                existing.status = _STATUS_MAP.get(parsed.status, existing.status)
                existing.metadata["vtn_status"] = parsed.status
                continue
            event = dr_event_from_parsed(parsed, vtn_id=dist.vtn_id, request_id=dist.request_id)
            response = await self.handle_incoming_event(event)
            if parsed.response_required == "always":
                opt_responses.append(
                    ox.EventOptResponse(
                        event_id=parsed.event_id,
                        modification_number=parsed.modification_number,
                        opt_type=response.opt_type,
                        request_id=dist.request_id,
                    )
                )

        # Events the VTN no longer lists are implicitly cancelled.
        for event in self._events.values():
            if (
                event.metadata.get("source") == "vtn"
                and event.event_id not in seen
                and event.status not in (DREventStatus.COMPLETED, DREventStatus.CANCELLED)
            ):
                event.status = DREventStatus.CANCELLED

        if opt_responses:
            await self._send_created_event(opt_responses)

    async def _send_created_event(self, responses: list[Any]) -> None:
        from vpp.protocols import openadr_xml as ox

        reply = await self._post("EiEvent", ox.build_created_event(str(self.ven_id), responses))
        ox.check_ei_response(reply)

    async def send_opt(self, event_id: str, opt_type: str) -> None:
        """Send (or change) the VEN's opt-in/opt-out for a VTN event."""
        from vpp.protocols import openadr_xml as ox

        if opt_type not in ("optIn", "optOut"):
            raise ValueError("opt_type must be 'optIn' or 'optOut'")
        event = self._events.get(event_id)
        if event is None or event.metadata.get("source") != "vtn":
            raise ValueError(f"Unknown VTN event {event_id}")
        self._responses[event_id] = DRResponse(event_id=event_id, opt_type=opt_type)
        await self._send_created_event(
            [
                ox.EventOptResponse(
                    event_id=event_id,
                    modification_number=int(event.metadata.get("modification_number", 0)),
                    opt_type=opt_type,
                    request_id=str(event.metadata.get("request_id", "")),
                )
            ]
        )

    def _advance_event_states(self) -> None:
        now = time.time()
        for event in self._events.values():
            if event.status == DREventStatus.PENDING and event.start_time <= now:
                event.status = DREventStatus.ACTIVE
            if event.status == DREventStatus.ACTIVE and now >= event.end_time:
                event.status = DREventStatus.COMPLETED

    async def _ven_poll_loop(self) -> None:
        """Live VEN: poll the VTN forever with exponential backoff on failure."""
        from vpp.protocols.openadr_xml import OpenADRError

        max_backoff = float(self._config.get("max_backoff_s", 300.0))
        while True:
            delay = self.poll_interval_s
            try:
                await self.poll_once()
                if self._consecutive_failures:
                    self._metrics.reconnect_count += 1
                    logger.info("OpenADR VTN reachable again")
                self._consecutive_failures = 0
                self._status = ProtocolStatus.CONNECTED
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._consecutive_failures += 1
                self._metrics.errors += 1
                self._status = ProtocolStatus.RECONNECTING
                if isinstance(exc, OpenADRError) and exc.code in _REREGISTER_CODES:
                    self.registration_id = None
                delay = backoff_delay(
                    self._consecutive_failures, base=max(1.0, delay), maximum=max_backoff
                )
                logger.warning("OpenADR poll failed (%s); retrying in %.0fs", exc, delay)
            await asyncio.sleep(delay)

    async def _simulated_tick_loop(self, interval: float) -> None:
        """Simulated VEN: advance in-memory event states on a timer."""
        while self.is_operational:
            try:
                self._advance_event_states()
            except Exception:
                logger.exception("OpenADR simulated tick error")
                self._metrics.errors += 1
            await asyncio.sleep(interval)
