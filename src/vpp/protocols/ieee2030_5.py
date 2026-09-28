"""IEEE 2030.5 (Smart Energy Profile 2.0) client adapter.

Modes (see :class:`~vpp.protocols.base.ProtocolMode`):

* **Live** (``server_url`` configured): an HTTPS client (mutual TLS with the
  device certificate, as 2030.5 requires) walks the standard resource tree

      DeviceCapability -> EndDeviceList -> EndDevice
        -> FunctionSetAssignmentsList -> DERProgramList
        -> DERControlList (+ DefaultDERControl)

  parses the ``application/sep+xml`` documents, and re-polls on the
  server's ``pollRate`` (or ``poll_interval_s``). Active DER controls,
  ordered by program primacy, are exposed via
  :meth:`IEEE2030_5Adapter.get_active_controls`; newly (in)active controls
  are dispatched to subscribers on ``ieee2030_5/control/...``. Status is
  ``CONNECTED`` while the server is reachable, ``RECONNECTING`` while
  backing off.
* **Simulated** (no ``server_url``): programs and controls live in memory
  (``register_program``/``apply_control``); status is ``SIMULATED``.

The client identifies its own EndDevice by LFDI/SFDI (configured, or
derived from the client certificate per IEEE 2030.5 section 6.3.4).

Not implemented yet: posting DERStatus/DERCapability/Response resources
back to the server, subscription/notification, and DERCurve parsing.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import ssl
import time
import uuid
from dataclasses import dataclass, field
from enum import IntFlag
from typing import Any
from urllib.parse import urljoin

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

SEP_NS = "urn:ieee:std:2030.5:ns"
SEP_CONTENT_TYPE = "application/sep+xml"


# ---------------------------------------------------------------------------
# IEEE 2030.5 resource types
# ---------------------------------------------------------------------------


class DERControlMode(IntFlag):
    """DER operating modes as per IEEE 2030.5 DERControlBase."""

    CHARGE = 0x0001
    DISCHARGE = 0x0002
    OP_MOD_CONNECT = 0x0004
    OP_MOD_ENERGIZE = 0x0008
    OP_MOD_FIXED_PF = 0x0010
    OP_MOD_FIXED_W = 0x0020
    OP_MOD_FREQ_DROOP = 0x0040
    OP_MOD_FREQ_WATT = 0x0080
    OP_MOD_VOLT_VAR = 0x0100
    OP_MOD_VOLT_WATT = 0x0200
    OP_MOD_WATT_PF = 0x0400


NO_MODES = DERControlMode(0)


class EventStatusCode:
    """``EventStatus.currentStatus`` values."""

    SCHEDULED = 0
    ACTIVE = 1
    CANCELLED = 2
    CANCELLED_WITH_RANDOMIZATION = 3
    SUPERSEDED = 4


@dataclass
class DERProgram:
    """A DER program defining default and active controls."""

    program_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    primacy: int = 0  # lower = higher priority
    default_control: DERControl | None = None
    active_controls: list[DERControl] = field(default_factory=list)
    href: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "description": self.description,
            "primacy": self.primacy,
            "href": self.href,
            "default_control": self.default_control.to_dict() if self.default_control else None,
            "active_controls": [c.to_dict() for c in self.active_controls],
        }


@dataclass
class DERControl:
    """A DER control event (e.g. curtailment, frequency response)."""

    control_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    modes: DERControlMode = NO_MODES
    set_watts: float | None = None  # target W (opModTargetW)
    set_var: float | None = None  # target var (opModTargetVar)
    set_pf: float | None = None  # power factor (opModFixedPFInjectW)
    set_gradient_w_per_s: float | None = None  # ramp rate
    start_time: float = 0.0
    duration_seconds: int = 3600
    randomize_start_s: int = 0
    randomize_duration_s: int = 0
    # Fields populated from a 2030.5 server
    description: str = ""
    program_id: str | None = None
    event_status: int | None = None  # EventStatus.currentStatus
    creation_time: float | None = None
    connect: bool | None = None  # opModConnect
    energize: bool | None = None  # opModEnergize
    max_limit_pct: float | None = None  # opModMaxLimW, % of setMaxW
    fixed_w_pct: float | None = None  # opModFixedW, % of setMaxW (signed)
    gen_limit_w: float | None = None  # opModGenLimW
    load_limit_w: float | None = None  # opModLoadLimW
    ramp_time_s: float | None = None  # rampTms
    href: str | None = None
    response_required: int = 0
    reply_to: str | None = None

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration_seconds

    def is_active_at(self, now: float) -> bool:
        """Whether the control applies at *now* (server-clock seconds)."""
        if self.event_status in (
            EventStatusCode.CANCELLED,
            EventStatusCode.CANCELLED_WITH_RANDOMIZATION,
            EventStatusCode.SUPERSEDED,
        ):
            return False
        if self.start_time and now >= self.end_time:
            return False
        if self.event_status == EventStatusCode.ACTIVE:
            return True
        return self.start_time <= now

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_id": self.control_id,
            "modes": int(self.modes),
            "set_watts": self.set_watts,
            "set_var": self.set_var,
            "set_pf": self.set_pf,
            "set_gradient_w_per_s": self.set_gradient_w_per_s,
            "start_time": self.start_time,
            "duration_seconds": self.duration_seconds,
            "description": self.description,
            "program_id": self.program_id,
            "event_status": self.event_status,
            "connect": self.connect,
            "energize": self.energize,
            "max_limit_pct": self.max_limit_pct,
            "fixed_w_pct": self.fixed_w_pct,
            "gen_limit_w": self.gen_limit_w,
            "load_limit_w": self.load_limit_w,
            "ramp_time_s": self.ramp_time_s,
        }


@dataclass
class DERStatus:
    """DER status report from a device."""

    device_id: str
    state_of_charge: float | None = None  # 0.0-1.0
    real_power_w: float = 0.0
    reactive_power_var: float = 0.0
    voltage_v: float = 0.0
    frequency_hz: float = 0.0
    op_mode: DERControlMode = NO_MODES
    alarm_status: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "state_of_charge": self.state_of_charge,
            "real_power_w": self.real_power_w,
            "reactive_power_var": self.reactive_power_var,
            "voltage_v": self.voltage_v,
            "frequency_hz": self.frequency_hz,
            "op_mode": int(self.op_mode),
            "timestamp": self.timestamp,
        }


@dataclass
class DERCapability:
    """Device nameplate ratings and capabilities."""

    device_id: str
    max_charge_rate_w: float = 0.0
    max_discharge_rate_w: float = 0.0
    max_apparent_power_va: float = 0.0
    nameplate_energy_wh: float = 0.0
    modes_supported: DERControlMode = NO_MODES

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "max_charge_rate_w": self.max_charge_rate_w,
            "max_discharge_rate_w": self.max_discharge_rate_w,
            "max_apparent_power_va": self.max_apparent_power_va,
            "nameplate_energy_wh": self.nameplate_energy_wh,
            "modes_supported": int(self.modes_supported),
        }


# ---------------------------------------------------------------------------
# Identity helpers (IEEE 2030.5 section 6.3.4)
# ---------------------------------------------------------------------------


def lfdi_from_cert_der(cert_der: bytes) -> str:
    """Long-form device identifier: first 160 bits of SHA-256(cert), hex."""
    return hashlib.sha256(cert_der).hexdigest()[:40].upper()


def sfdi_from_lfdi(lfdi: str) -> int:
    """Short-form device identifier: first 36 bits of the LFDI + check digit."""
    base = int(lfdi[:9], 16)
    digit_sum = sum(int(d) for d in str(base))
    check = (10 - digit_sum % 10) % 10
    return base * 10 + check


def lfdi_from_cert_file(path: str) -> str:
    with open(path, encoding="ascii") as fh:
        pem = fh.read()
    start = pem.index("-----BEGIN CERTIFICATE-----")
    end = pem.index("-----END CERTIFICATE-----", start) + len("-----END CERTIFICATE-----")
    return lfdi_from_cert_der(ssl.PEM_cert_to_DER_cert(pem[start:end]))


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------


def _q(tag: str) -> str:
    return f"{{{SEP_NS}}}{tag}"


def _parser() -> Any:
    from lxml import etree

    return etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False)


def parse_sep_xml(data: bytes) -> Any:
    from lxml import etree

    try:
        return etree.fromstring(data, _parser())
    except etree.XMLSyntaxError as exc:
        raise ProtocolTransportError(f"Malformed IEEE 2030.5 XML: {exc}") from exc


def _child(el: Any, *path: str) -> Any:
    for tag in path:
        if el is None:
            return None
        el = el.find(_q(tag))
    return el


def _text(el: Any, *path: str) -> str | None:
    node = _child(el, *path)
    if node is None or node.text is None:
        return None
    return str(node.text).strip()


def _int(el: Any, *path: str) -> int | None:
    text = _text(el, *path)
    try:
        return int(text) if text is not None else None
    except ValueError:
        return None


def _link(el: Any, tag: str) -> str | None:
    node = _child(el, tag)
    return node.get("href") if node is not None else None


def _bool(el: Any, *path: str) -> bool | None:
    text = _text(el, *path)
    return None if text is None else text.lower() in ("true", "1")


def _scaled(el: Any, tag: str, value_tag: str = "value") -> float | None:
    """Value x 10^multiplier for ActivePower/ReactivePower-style elements."""
    node = _child(el, tag)
    if node is None:
        return None
    value = _int(node, value_tag)
    if value is None:
        return None
    return float(value * 10.0 ** (_int(node, "multiplier") or 0))


def _parse_control_base(base: Any, ctrl: DERControl) -> None:
    if base is None:
        return
    modes = NO_MODES
    ctrl.connect = _bool(base, "opModConnect")
    if ctrl.connect is not None:
        modes |= DERControlMode.OP_MOD_CONNECT
    ctrl.energize = _bool(base, "opModEnergize")
    if ctrl.energize is not None:
        modes |= DERControlMode.OP_MOD_ENERGIZE
    pf = _scaled(base, "opModFixedPFInjectW", "displacement")
    if pf is not None:
        ctrl.set_pf = pf
        modes |= DERControlMode.OP_MOD_FIXED_PF
    fixed_w = _int(base, "opModFixedW")
    if fixed_w is not None:
        ctrl.fixed_w_pct = fixed_w / 100.0
        modes |= DERControlMode.OP_MOD_FIXED_W
    target_w = _scaled(base, "opModTargetW")
    if target_w is not None:
        ctrl.set_watts = target_w
        modes |= DERControlMode.OP_MOD_FIXED_W
        modes |= DERControlMode.CHARGE if target_w < 0 else DERControlMode.DISCHARGE
    ctrl.set_var = _scaled(base, "opModTargetVar")
    max_lim = _int(base, "opModMaxLimW")
    if max_lim is not None:
        ctrl.max_limit_pct = max_lim / 100.0
    ctrl.gen_limit_w = _scaled(base, "opModGenLimW")
    ctrl.load_limit_w = _scaled(base, "opModLoadLimW")
    for tag, flag in (
        ("opModFreqDroop", DERControlMode.OP_MOD_FREQ_DROOP),
        ("opModFreqWatt", DERControlMode.OP_MOD_FREQ_WATT),
        ("opModVoltVar", DERControlMode.OP_MOD_VOLT_VAR),
        ("opModVoltWatt", DERControlMode.OP_MOD_VOLT_WATT),
        ("opModWattPF", DERControlMode.OP_MOD_WATT_PF),
    ):
        if _child(base, tag) is not None:
            modes |= flag
    ramp = _int(base, "rampTms")
    if ramp is not None:
        ctrl.ramp_time_s = ramp / 100.0
    ctrl.modes = modes


def parse_der_control(el: Any, program_id: str | None = None) -> DERControl:
    """Parse a ``DERControl`` element."""
    ctrl = DERControl(
        control_id=_text(el, "mRID") or str(uuid.uuid4()),
        description=_text(el, "description") or "",
        program_id=program_id,
        start_time=float(_int(el, "interval", "start") or 0),
        duration_seconds=_int(el, "interval", "duration") or 0,
        randomize_start_s=_int(el, "randomizeStart") or 0,
        randomize_duration_s=_int(el, "randomizeDuration") or 0,
        event_status=_int(el, "EventStatus", "currentStatus"),
        href=el.get("href"),
        reply_to=el.get("replyTo"),
        response_required=int(el.get("responseRequired", "0"), 16)
        if el.get("responseRequired")
        else 0,
    )
    created = _int(el, "creationTime")
    ctrl.creation_time = float(created) if created is not None else None
    _parse_control_base(_child(el, "DERControlBase"), ctrl)
    return ctrl


def parse_default_der_control(el: Any, program_id: str | None = None) -> DERControl:
    """Parse a ``DefaultDERControl`` element (no interval: always applicable)."""
    ctrl = DERControl(
        control_id=_text(el, "mRID") or str(uuid.uuid4()),
        description=_text(el, "description") or "",
        program_id=program_id,
        start_time=0.0,
        duration_seconds=0,
        href=el.get("href"),
    )
    _parse_control_base(_child(el, "DERControlBase"), ctrl)
    grad = _int(el, "setGradW")
    if grad is not None:
        # hundredths of a percent of setMaxW per second
        ctrl.set_gradient_w_per_s = grad / 100.0
    return ctrl


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class IEEE2030_5Adapter(ProtocolAdapter):
    """IEEE 2030.5 (SEP 2.0) client adapter.

    Configuration keys:
        server_url, dcap_path (default "/dcap"), lfdi, sfdi, device_id,
        cert_path, key_path, ca_path, verify_tls, tls_ciphers,
        poll_interval_s (overrides the server pollRate), timeout_s,
        max_backoff_s, list_page_size
    """

    def __init__(self) -> None:
        super().__init__("ieee2030_5", "2.0")
        self._programs: dict[str, DERProgram] = {}
        self._server_programs: dict[str, DERProgram] = {}
        self._statuses: dict[str, DERStatus] = {}
        self._capabilities: dict[str, DERCapability] = {}
        self._poll_task: asyncio.Task | None = None
        self._message_queue: asyncio.Queue[ProtocolMessage] = asyncio.Queue(maxsize=500)
        self._client: Any = None
        self._server_poll_rate_s: float | None = None
        self._clock_offset_s = 0.0
        self._announced_active: set[str] = set()
        self._consecutive_failures = 0
        self.end_device_href: str | None = None
        self.lfdi: str | None = None

    # -- Mode ----------------------------------------------------------------

    @property
    def live(self) -> bool:
        return bool(self._config.get("server_url"))

    @property
    def mode(self) -> ProtocolMode:
        return ProtocolMode.LIVE if self.live else ProtocolMode.SIMULATED

    @property
    def poll_interval_s(self) -> float:
        configured = self._config.get("poll_interval_s")
        if configured is not None:
            return float(configured)
        return self._server_poll_rate_s or 900.0  # spec default pollRate

    def server_now(self) -> float:
        """Current time on the server's clock (from the Time resource)."""
        return time.time() + self._clock_offset_s

    # -- Lifecycle -----------------------------------------------------------

    async def connect(self) -> None:
        self._status = ProtocolStatus.CONNECTING

        if not self.live:
            self._enter_simulated_mode("no IEEE 2030.5 server_url configured")
            poll_interval = float(self._config.get("poll_interval_s", 60))
            if poll_interval > 0:
                self._poll_task = asyncio.create_task(self._simulated_tick_loop(poll_interval))
            return

        self._client = self._build_client()
        try:
            await self.discover()
        except Exception:
            self._status = ProtocolStatus.ERROR
            self._metrics.errors += 1
            await self._close_client()
            raise

        self._status = ProtocolStatus.CONNECTED
        self._metrics.connected_since = time.time()
        if self.poll_interval_s > 0:
            self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info(
            "IEEE 2030.5 client connected to %s (EndDevice %s, %d programs)",
            self._config.get("server_url"),
            self.end_device_href,
            len(self._server_programs),
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
        logger.info("IEEE 2030.5 adapter disconnected")

    async def send(self, message: ProtocolMessage) -> None:
        if not self.is_operational:
            raise ConnectionError("IEEE 2030.5 adapter not started")

        topic = message.topic
        if "status" in topic:
            status = DERStatus(**message.payload)
            self._statuses[status.device_id] = status
        elif "control" in topic:
            payload = dict(message.payload)
            program_id = payload.pop("program_id", "default")
            control = DERControl(**payload)
            program = self._programs.get(program_id)
            if program:
                program.active_controls.append(control)

        self._metrics.messages_sent += 1
        self._metrics.last_message_at = time.time()

    async def receive(self) -> ProtocolMessage | None:
        try:
            return self._message_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    # -- DER program management ----------------------------------------------

    def register_program(self, program: DERProgram) -> None:
        self._programs[program.program_id] = program
        logger.info("Registered DER program: %s", program.program_id)

    def get_program(self, program_id: str) -> DERProgram | None:
        return self._server_programs.get(program_id) or self._programs.get(program_id)

    def list_programs(self) -> list[DERProgram]:
        """All programs (server-discovered first), ordered by primacy."""
        merged = {**self._programs, **self._server_programs}
        return sorted(merged.values(), key=lambda p: p.primacy)

    def get_active_controls(self, now: float | None = None) -> list[DERControl]:
        """Controls in effect now, highest-priority program (lowest primacy) first.

        Within a program, the most recently created control wins ties.
        """
        at = self.server_now() if now is None else now
        out: list[DERControl] = []
        for program in self.list_programs():
            active = [c for c in program.active_controls if c.is_active_at(at)]
            active.sort(key=lambda c: -(c.creation_time or 0.0))
            out.extend(active)
        return out

    def get_default_controls(self) -> list[DERControl]:
        return [p.default_control for p in self.list_programs() if p.default_control]

    # -- Device registration -------------------------------------------------

    def register_capability(self, cap: DERCapability) -> None:
        self._capabilities[cap.device_id] = cap

    def get_capability(self, device_id: str) -> DERCapability | None:
        return self._capabilities.get(device_id)

    # -- Status reporting ----------------------------------------------------

    async def report_status(self, status: DERStatus) -> None:
        """Record DER status locally and notify subscribers.

        (Posting DERStatus to the server is not implemented yet.)
        """
        self._statuses[status.device_id] = status
        msg = ProtocolMessage(
            topic=f"ieee2030_5/status/{status.device_id}",
            payload=status.to_dict(),
            source="ieee2030_5",
        )
        await self._dispatch(msg)
        self._metrics.messages_sent += 1

    def get_status(self, device_id: str) -> DERStatus | None:
        return self._statuses.get(device_id)

    # -- Control application -------------------------------------------------

    async def apply_control(self, program_id: str, control: DERControl) -> None:
        """Apply a DER control to a program and notify subscribers."""
        program = self._programs.get(program_id)
        if program is None:
            raise ValueError(f"Unknown program: {program_id}")

        program.active_controls.append(control)
        await self._announce_control(program_id, control)

    async def _announce_control(self, program_id: str, control: DERControl) -> None:
        msg = ProtocolMessage(
            topic=f"ieee2030_5/control/{program_id}/{control.control_id}",
            payload={**control.to_dict(), "program_id": program_id},
            source="ieee2030_5",
        )
        await self._dispatch(msg)
        try:
            self._message_queue.put_nowait(msg)
        except asyncio.QueueFull:
            self._metrics.errors += 1

    # -- HTTP transport ------------------------------------------------------

    def _build_client(self) -> Any:
        import httpx

        verify = build_ssl_context(
            verify=bool(self._config.get("verify_tls", True)),
            ca_path=self._config.get("ca_path"),
            cert_path=self._config.get("cert_path"),
            key_path=self._config.get("key_path"),
            ciphers=self._config.get("tls_ciphers"),
        )
        return httpx.AsyncClient(
            verify=verify,
            timeout=httpx.Timeout(float(self._config.get("timeout_s", 10.0))),
            headers={"Accept": SEP_CONTENT_TYPE},
        )

    async def _close_client(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            finally:
                self._client = None

    def _url(self, href: str) -> str:
        return urljoin(str(self._config["server_url"]).rstrip("/") + "/", href)

    async def _get(self, href: str, params: dict[str, Any] | None = None) -> Any:
        import httpx

        if self._client is None:
            raise ProtocolTransportError("IEEE 2030.5 HTTP client not started")
        try:
            response = await self._client.get(self._url(href), params=params)
        except httpx.HTTPError as exc:
            raise ProtocolTransportError(f"GET {href} failed: {exc}") from exc
        self._metrics.messages_sent += 1
        self._metrics.last_message_at = time.time()
        if response.status_code >= 400:
            raise ProtocolTransportError(f"GET {href} returned HTTP {response.status_code}")
        self._metrics.messages_received += 1
        return parse_sep_xml(response.content)

    async def _get_list(self, href: str, item_tag: str) -> tuple[Any, list[Any]]:
        """Fetch every item of a 2030.5 list resource, following s/l paging."""
        page = int(self._config.get("list_page_size", 255))
        items: list[Any] = []
        start = 0
        root = None
        for _ in range(1000):  # hard stop against a misbehaving server
            root = await self._get(href, {"s": start, "l": page})
            batch = root.findall(_q(item_tag))
            items.extend(batch)
            start += len(batch)
            total = int(root.get("all", len(items)))
            if not batch or start >= total:
                break
        return root, items

    # -- Discovery -----------------------------------------------------------

    def _own_identity(self) -> tuple[str | None, str | None]:
        lfdi = self._config.get("lfdi")
        if not lfdi and self._config.get("cert_path"):
            try:
                lfdi = lfdi_from_cert_file(str(self._config["cert_path"]))
            except (OSError, ValueError):
                logger.warning("Could not derive LFDI from %s", self._config["cert_path"])
        sfdi = self._config.get("sfdi")
        if sfdi is None and lfdi:
            sfdi = sfdi_from_lfdi(str(lfdi))
        return (str(lfdi).upper() if lfdi else None, str(sfdi) if sfdi is not None else None)

    def _select_end_device(self, devices: list[Any]) -> Any:
        lfdi, sfdi = self._own_identity()
        self.lfdi = lfdi
        for dev in devices:
            if lfdi and (_text(dev, "lFDI") or "").upper() == lfdi:
                return dev
            if sfdi and _text(dev, "sFDI") == sfdi:
                return dev
        if len(devices) == 1 and not (lfdi or sfdi):
            return devices[0]
        if not devices:
            raise ProtocolTransportError("IEEE 2030.5 server lists no EndDevice for this client")
        raise ProtocolTransportError(
            "No EndDevice matches this client's LFDI/SFDI; configure lfdi or cert_path"
        )

    async def discover(self) -> list[DERProgram]:
        """Walk the resource tree and refresh programs/controls from the server."""
        dcap = await self._get(str(self._config.get("dcap_path", "/dcap")))
        poll_rate = dcap.get("pollRate")
        if poll_rate and poll_rate.isdigit():
            self._server_poll_rate_s = float(poll_rate)

        time_href = _link(dcap, "TimeLink")
        if time_href:
            tm = await self._get(time_href)
            server_time = _int(tm, "currentTime")
            if server_time is not None:
                self._clock_offset_s = server_time - time.time()

        edev_href = _link(dcap, "EndDeviceListLink")
        if not edev_href:
            raise ProtocolTransportError("DeviceCapability has no EndDeviceListLink")
        _, devices = await self._get_list(edev_href, "EndDevice")
        device = self._select_end_device(devices)
        self.end_device_href = device.get("href")

        fsa_href = _link(device, "FunctionSetAssignmentsListLink")
        programs: dict[str, DERProgram] = {}
        if fsa_href:
            _, fsas = await self._get_list(fsa_href, "FunctionSetAssignments")
            for fsa in fsas:
                derp_href = _link(fsa, "DERProgramListLink")
                if not derp_href:
                    continue
                derp_root, derps = await self._get_list(derp_href, "DERProgram")
                list_rate = derp_root.get("pollRate") if derp_root is not None else None
                if list_rate and list_rate.isdigit():
                    self._server_poll_rate_s = float(list_rate)
                for derp in derps:
                    program = await self._load_program(derp)
                    programs.setdefault(program.program_id, program)

        self._server_programs = programs
        await self._announce_changes()
        return list(programs.values())

    async def _load_program(self, derp: Any) -> DERProgram:
        program_id = _text(derp, "mRID") or derp.get("href") or str(uuid.uuid4())
        program = DERProgram(
            program_id=program_id,
            description=_text(derp, "description") or "",
            primacy=_int(derp, "primacy") or 0,
            href=derp.get("href"),
        )
        default_href = _link(derp, "DefaultDERControlLink")
        if default_href:
            program.default_control = parse_default_der_control(
                await self._get(default_href), program_id
            )
        controls_href = _link(derp, "DERControlListLink")
        if controls_href:
            _, controls = await self._get_list(controls_href, "DERControl")
            program.active_controls = [parse_der_control(c, program_id) for c in controls]
        return program

    async def _announce_changes(self) -> None:
        """Dispatch controls that became active since the last poll."""
        now = self.server_now()
        active_now: set[str] = set()
        for control in self.get_active_controls(now):
            key = f"{control.program_id}/{control.control_id}"
            active_now.add(key)
            if key not in self._announced_active:
                await self._announce_control(str(control.program_id), control)
        self._announced_active = active_now

    # -- Polling -------------------------------------------------------------

    async def _poll_loop(self) -> None:
        max_backoff = float(self._config.get("max_backoff_s", 300.0))
        while True:
            await asyncio.sleep(self._next_delay(max_backoff))
            try:
                await self.discover()
                if self._consecutive_failures:
                    self._metrics.reconnect_count += 1
                    logger.info("IEEE 2030.5 server reachable again")
                self._consecutive_failures = 0
                self._status = ProtocolStatus.CONNECTED
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._consecutive_failures += 1
                self._metrics.errors += 1
                self._status = ProtocolStatus.RECONNECTING
                logger.warning("IEEE 2030.5 poll failed: %s", exc)

    def _next_delay(self, max_backoff: float) -> float:
        if self._consecutive_failures:
            return backoff_delay(
                self._consecutive_failures,
                base=min(max(1.0, self.poll_interval_s), max_backoff),
                maximum=max_backoff,
            )
        return self.poll_interval_s

    async def _simulated_tick_loop(self, interval: float) -> None:
        while self.is_operational:
            try:
                now = time.time()
                for program in self._programs.values():
                    expired = [
                        c
                        for c in program.active_controls
                        if c.start_time + c.duration_seconds < now and c.start_time > 0
                    ]
                    for c in expired:
                        program.active_controls.remove(c)
            except Exception:
                logger.exception("IEEE 2030.5 simulated tick error")
                self._metrics.errors += 1
            await asyncio.sleep(interval)
