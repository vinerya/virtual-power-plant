"""IEEE 2030.5 (Smart Energy Profile 2.0) adapter.

Implements DERProgram, DERControl, and DERStatus resources for
utility-grade DER management with certificate-based authentication.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import IntFlag
from typing import Any

from vpp.protocols.base import (
    ProtocolAdapter,
    ProtocolMessage,
    ProtocolStatus,
)

logger = logging.getLogger(__name__)


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


@dataclass
class DERProgram:
    """A DER program defining default and active controls."""

    program_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    primacy: int = 0  # lower = higher priority
    default_control: DERControl | None = None
    active_controls: list[DERControl] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "description": self.description,
            "primacy": self.primacy,
            "default_control": self.default_control.to_dict() if self.default_control else None,
            "active_controls": [c.to_dict() for c in self.active_controls],
        }


@dataclass
class DERControl:
    """A DER control event (e.g. curtailment, frequency response)."""

    control_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    modes: DERControlMode = DERControlMode(0)
    set_watts: float | None = None           # target W
    set_var: float | None = None             # target var
    set_pf: float | None = None              # power factor
    set_gradient_w_per_s: float | None = None  # ramp rate
    start_time: float = 0.0
    duration_seconds: int = 3600
    randomize_start_s: int = 0
    randomize_duration_s: int = 0

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
        }


@dataclass
class DERStatus:
    """DER status report from a device."""

    device_id: str
    state_of_charge: float | None = None   # 0.0–1.0
    real_power_w: float = 0.0
    reactive_power_var: float = 0.0
    voltage_v: float = 0.0
    frequency_hz: float = 0.0
    op_mode: DERControlMode = DERControlMode(0)
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
    modes_supported: DERControlMode = DERControlMode(0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "max_charge_rate_w": self.max_charge_rate_w,
            "max_discharge_rate_w": self.max_discharge_rate_w,
            "max_apparent_power_va": self.max_apparent_power_va,
            "nameplate_energy_wh": self.nameplate_energy_wh,
            "modes_supported": int(self.modes_supported),
        }


class IEEE2030_5Adapter(ProtocolAdapter):
    """IEEE 2030.5 (SEP 2.0) adapter.

    Configuration keys:
        server_url, device_id, cert_path, key_path, ca_path,
        poll_interval_s, default_program_id
    """

    def __init__(self) -> None:
        super().__init__("ieee2030_5", "2.0")
        self._programs: dict[str, DERProgram] = {}
        self._statuses: dict[str, DERStatus] = {}
        self._capabilities: dict[str, DERCapability] = {}
        self._poll_task: asyncio.Task | None = None
        self._message_queue: asyncio.Queue[ProtocolMessage] = asyncio.Queue(maxsize=500)

    # -- Lifecycle -----------------------------------------------------------

    async def connect(self) -> None:
        self._status = ProtocolStatus.CONNECTING

        # In a full implementation we would establish a TLS-authenticated
        # HTTP connection to the utility server.
        self._status = ProtocolStatus.CONNECTED
        self._metrics.connected_since = time.time()

        poll_interval = float(self._config.get("poll_interval_s", 60))
        if poll_interval > 0:
            self._poll_task = asyncio.create_task(self._poll_loop(poll_interval))

        logger.info("IEEE 2030.5 adapter connected")

    async def disconnect(self) -> None:
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        self._status = ProtocolStatus.DISCONNECTED
        self._metrics.connected_since = None
        logger.info("IEEE 2030.5 adapter disconnected")

    async def send(self, message: ProtocolMessage) -> None:
        if not self.is_connected:
            raise ConnectionError("IEEE 2030.5 not connected")

        topic = message.topic
        if "status" in topic:
            status = DERStatus(**message.payload)
            self._statuses[status.device_id] = status
        elif "control" in topic:
            control = DERControl(**message.payload)
            program_id = message.payload.get("program_id", "default")
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
        return self._programs.get(program_id)

    def list_programs(self) -> list[DERProgram]:
        return list(self._programs.values())

    # -- Device registration -------------------------------------------------

    def register_capability(self, cap: DERCapability) -> None:
        self._capabilities[cap.device_id] = cap

    def get_capability(self, device_id: str) -> DERCapability | None:
        return self._capabilities.get(device_id)

    # -- Status reporting ----------------------------------------------------

    async def report_status(self, status: DERStatus) -> None:
        """Report DER status to the utility server."""
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

    # -- Polling -------------------------------------------------------------

    async def _poll_loop(self, interval: float) -> None:
        while self.is_connected:
            try:
                # In production this polls the utility server for new programs
                # and controls.  For the MVP we process in-memory data.
                now = time.time()
                for program in self._programs.values():
                    expired = [
                        c for c in program.active_controls
                        if c.start_time + c.duration_seconds < now and c.start_time > 0
                    ]
                    for c in expired:
                        program.active_controls.remove(c)
            except Exception:
                logger.exception("IEEE 2030.5 poll error")
                self._metrics.errors += 1
            await asyncio.sleep(interval)
