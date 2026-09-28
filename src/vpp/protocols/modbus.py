"""Modbus TCP/RTU adapter for inverter and meter communication.

Supports predefined register maps for common inverters (SMA, Fronius,
SolarEdge) and a generic mode for custom register definitions.

Registers are **read-only unless flagged** ``writable=True``: the setpoint
writer (:mod:`vpp.protocols.modbus_control`) refuses to write a named
register that is not flagged. The writable definitions shipped here are the
SunSpec *model-relative* control blocks built by
:func:`sunspec_model_123_registers` (Immediate Controls, ``WMaxLimPct``) and
:func:`sunspec_model_124_registers` (Storage, generic/unverified). SunSpec
models sit at a device-specific address (after the preceding models in the
chain), so these take the address of the model's ``ID`` register instead of
guessing a vendor's absolute addresses.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import struct
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from vpp.protocols.base import (
    ProtocolAdapter,
    ProtocolMessage,
    ProtocolStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Register map definitions
# ---------------------------------------------------------------------------


class RegisterType(str, Enum):
    HOLDING = "holding"
    INPUT = "input"
    COIL = "coil"
    DISCRETE = "discrete"


@dataclass
class RegisterDefinition:
    """A single Modbus register definition."""

    address: int
    count: int = 1
    register_type: RegisterType = RegisterType.HOLDING
    name: str = ""
    unit: str = ""
    scale: float = 1.0
    data_type: str = "uint16"  # uint16, int16, uint32, int32, float32
    writable: bool = False  # only flagged registers may be written by the setpoint writer


@dataclass
class RegisterMap:
    """Collection of registers for a device profile."""

    name: str
    registers: dict[str, RegisterDefinition]
    unit_id: int = 1


# Pre-built register maps for common inverters
INVERTER_MAPS: dict[str, RegisterMap] = {
    "sma_sunnyboy": RegisterMap(
        name="SMA Sunny Boy",
        registers={
            "ac_power": RegisterDefinition(
                30775, 2, RegisterType.INPUT, "AC Power", "W", 1.0, "int32"
            ),
            "dc_power": RegisterDefinition(
                30773, 2, RegisterType.INPUT, "DC Power", "W", 1.0, "int32"
            ),
            "daily_yield": RegisterDefinition(
                30517, 4, RegisterType.INPUT, "Daily Yield", "Wh", 1.0, "uint64"
            ),
            "total_yield": RegisterDefinition(
                30513, 4, RegisterType.INPUT, "Total Yield", "Wh", 1.0, "uint64"
            ),
            "grid_frequency": RegisterDefinition(
                30803, 2, RegisterType.INPUT, "Grid Freq", "Hz", 0.01, "uint32"
            ),
        },
    ),
    "fronius_symo": RegisterMap(
        name="Fronius Symo",
        registers={
            "ac_power": RegisterDefinition(
                40092, 1, RegisterType.HOLDING, "AC Power", "W", 1.0, "float32"
            ),
            "ac_energy": RegisterDefinition(
                40094, 2, RegisterType.HOLDING, "AC Energy", "Wh", 1.0, "float32"
            ),
            "dc_power": RegisterDefinition(
                40101, 1, RegisterType.HOLDING, "DC Power", "W", 1.0, "float32"
            ),
            "frequency": RegisterDefinition(
                40086, 1, RegisterType.HOLDING, "Frequency", "Hz", 1.0, "float32"
            ),
        },
    ),
    "solaredge_se": RegisterMap(
        name="SolarEdge SE",
        registers={
            "ac_power": RegisterDefinition(
                40084, 1, RegisterType.HOLDING, "AC Power", "W", 1.0, "int16"
            ),
            "ac_power_scale": RegisterDefinition(
                40085, 1, RegisterType.HOLDING, "AC Power Scale", "", 1.0, "int16"
            ),
            "dc_power": RegisterDefinition(
                40101, 1, RegisterType.HOLDING, "DC Power", "W", 1.0, "int16"
            ),
            "temperature": RegisterDefinition(
                40104, 1, RegisterType.HOLDING, "Temperature", "°C", 0.01, "int16"
            ),
            "ac_energy": RegisterDefinition(
                40094, 2, RegisterType.HOLDING, "AC Energy", "Wh", 1.0, "uint32"
            ),
        },
    ),
    "generic_meter": RegisterMap(
        name="Generic Power Meter",
        registers={
            "voltage_l1": RegisterDefinition(
                0, 2, RegisterType.INPUT, "Voltage L1", "V", 0.1, "float32"
            ),
            "voltage_l2": RegisterDefinition(
                2, 2, RegisterType.INPUT, "Voltage L2", "V", 0.1, "float32"
            ),
            "voltage_l3": RegisterDefinition(
                4, 2, RegisterType.INPUT, "Voltage L3", "V", 0.1, "float32"
            ),
            "current_l1": RegisterDefinition(
                6, 2, RegisterType.INPUT, "Current L1", "A", 0.01, "float32"
            ),
            "power_total": RegisterDefinition(
                12, 2, RegisterType.INPUT, "Total Power", "W", 1.0, "float32"
            ),
            "energy_total": RegisterDefinition(
                72, 2, RegisterType.INPUT, "Total Energy", "kWh", 0.1, "float32"
            ),
        },
    ),
}


# ---------------------------------------------------------------------------
# SunSpec control blocks (model-relative; offsets include the ID and L registers)
# ---------------------------------------------------------------------------


def sunspec_model_123_registers(base: int) -> dict[str, RegisterDefinition]:
    """SunSpec model 123 *Immediate Controls* at *base* (address of its ``ID``).

    Offsets per the SunSpec inverter controls model 123 (``ID``=123, ``L``=24):
    ``WMaxLimPct`` (+5, uint16, % of WMax scaled by ``WMaxLimPct_SF``),
    ``WMaxLimPct_RvrtTms`` (+7, uint16 s, device-side revert timeout),
    ``WMaxLim_Ena`` (+9, enum16: 0 disabled / 1 enabled) and
    ``WMaxLimPct_SF`` (+23, sunssf int16 power-of-ten exponent).
    """
    h = RegisterType.HOLDING
    return {
        "sunspec_123_id": RegisterDefinition(base, 1, h, "Model ID (123)", "", 1.0, "uint16"),
        "wmax_lim_pct": RegisterDefinition(
            base + 5, 1, h, "WMaxLimPct", "%", 1.0, "uint16", writable=True
        ),
        "wmax_lim_pct_rvrt_tms": RegisterDefinition(
            base + 7, 1, h, "WMaxLimPct_RvrtTms", "s", 1.0, "uint16", writable=True
        ),
        "wmax_lim_ena": RegisterDefinition(
            base + 9, 1, h, "WMaxLim_Ena", "", 1.0, "uint16", writable=True
        ),
        "wmax_lim_pct_sf": RegisterDefinition(base + 23, 1, h, "WMaxLimPct_SF", "", 1.0, "int16"),
    }


def sunspec_model_124_registers(base: int) -> dict[str, RegisterDefinition]:
    """SunSpec model 124 *Storage* at *base* (address of its ``ID``).

    **Generic/unverified**: the offsets follow the published model 124
    layout (``ID``=124, ``L``=24), but how devices interpret forced
    charge/discharge (negative ``InWRte``/``OutWRte``) differs between
    vendors and has not been verified against hardware. ``StorCtl_Mod``
    (+5, bitfield16: bit0 charge limit, bit1 discharge limit active),
    ``OutWRte`` (+12, int16 % of max discharge rate), ``InWRte`` (+13, int16 %
    of max charge rate), ``InOutWRte_RvrtTms`` (+15, uint16 s) and
    ``InOutWRte_SF`` (+25, sunssf).
    """
    h = RegisterType.HOLDING
    return {
        "sunspec_124_id": RegisterDefinition(base, 1, h, "Model ID (124)", "", 1.0, "uint16"),
        "stor_ctl_mod": RegisterDefinition(
            base + 5, 1, h, "StorCtl_Mod", "", 1.0, "uint16", writable=True
        ),
        "out_w_rte": RegisterDefinition(
            base + 12, 1, h, "OutWRte", "%", 1.0, "int16", writable=True
        ),
        "in_w_rte": RegisterDefinition(
            base + 13, 1, h, "InWRte", "%", 1.0, "int16", writable=True
        ),
        "in_out_w_rte_rvrt_tms": RegisterDefinition(
            base + 15, 1, h, "InOutWRte_RvrtTms", "s", 1.0, "uint16", writable=True
        ),
        "in_out_w_rte_sf": RegisterDefinition(base + 25, 1, h, "InOutWRte_SF", "", 1.0, "int16"),
    }


class ModbusWriteError(RuntimeError):
    """A Modbus write or read was rejected by the device."""


def _unit_kwarg(method: Any) -> str:
    """pymodbus renamed the unit-id keyword ``slave`` -> ``device_id`` (3.10)."""
    try:
        params = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return "device_id"
    return "slave" if "slave" in params and "device_id" not in params else "device_id"


_INT_RANGES: dict[str, tuple[int, int, int]] = {
    "uint16": (0, 0xFFFF, 1),
    "int16": (-0x8000, 0x7FFF, 1),
    "uint32": (0, 0xFFFFFFFF, 2),
    "int32": (-0x80000000, 0x7FFFFFFF, 2),
}


def encode_value(value: float, data_type: str) -> list[int]:
    """Encode an (already scaled) raw value into big-endian 16-bit registers.

    Raises ``ValueError`` when the value does not fit the type (never wraps).
    """
    if data_type == "float32":
        hi, lo = struct.unpack(">HH", struct.pack(">f", float(value)))
        return [hi, lo]
    if data_type not in _INT_RANGES:
        raise ValueError(f"cannot encode data_type {data_type!r}")
    lo_lim, hi_lim, words = _INT_RANGES[data_type]
    raw = round(float(value))
    if not lo_lim <= raw <= hi_lim:
        raise ValueError(f"value {raw} out of range for {data_type}")
    raw &= (1 << (16 * words)) - 1
    return [(raw >> (16 * (words - 1 - i))) & 0xFFFF for i in range(words)]


def register_words(data_type: str) -> int:
    """Number of 16-bit registers a value of *data_type* occupies."""
    return 2 if data_type in ("uint32", "int32", "float32") else 1


class ModbusAdapter(ProtocolAdapter):
    """Modbus TCP/RTU adapter implementing the VPP ``ProtocolAdapter`` ABC.

    Configuration keys:
        host, port (TCP) | serial_port, baudrate (RTU),
        mode ("tcp" | "rtu"), device_profile, unit_id, poll_interval_s,
        custom_registers
    """

    def __init__(self) -> None:
        super().__init__("modbus", "1.0")
        self._client: Any | None = None
        self._poll_task: asyncio.Task | None = None
        self._register_map: RegisterMap | None = None
        self._latest_values: dict[str, float] = {}

    # -- Lifecycle -----------------------------------------------------------

    async def connect(self) -> None:
        try:
            from pymodbus.client import AsyncModbusSerialClient, AsyncModbusTcpClient
        except ImportError as exc:
            raise RuntimeError(
                "pymodbus is required for Modbus support. "
                "Install with: pip install 'virtual-power-plant[protocols]'"
            ) from exc

        self._status = ProtocolStatus.CONNECTING
        mode = self._config.get("mode", "tcp")

        if mode == "tcp":
            host = self._config.get("host", "localhost")
            port = int(self._config.get("port", 502))
            self._client = AsyncModbusTcpClient(host, port=port)
        elif mode == "rtu":
            serial_port = self._config.get("serial_port", "/dev/ttyUSB0")
            baudrate = int(self._config.get("baudrate", 9600))
            self._client = AsyncModbusSerialClient(serial_port, baudrate=baudrate)
        else:
            raise ValueError(f"Unknown Modbus mode: {mode}")

        connected = await self._client.connect()
        if not connected:
            self._status = ProtocolStatus.ERROR
            raise ConnectionError(f"Modbus {mode} connection failed")

        # Load register map
        profile = self._config.get("device_profile", "generic_meter")
        if profile in INVERTER_MAPS:
            # Copy: custom registers / unit id must not leak into the shared profile.
            base = INVERTER_MAPS[profile]
            self._register_map = RegisterMap(base.name, dict(base.registers), base.unit_id)
        else:
            self._register_map = RegisterMap(name="custom", registers={})

        # Apply custom registers if provided
        custom = self._config.get("custom_registers", {})
        for name, reg_def in custom.items():
            self._register_map.registers[name] = RegisterDefinition(**reg_def)

        unit_id = int(self._config.get("unit_id", self._register_map.unit_id))
        self._register_map.unit_id = unit_id

        self._status = ProtocolStatus.CONNECTED
        self._metrics.connected_since = time.time()
        logger.info("Modbus %s connected (profile=%s)", mode, profile)

        # Start polling
        poll_interval = float(self._config.get("poll_interval_s", 5.0))
        if poll_interval > 0:
            self._poll_task = asyncio.create_task(self._poll_loop(poll_interval))

    async def disconnect(self) -> None:
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._poll_task
        if self._client is not None:
            self._client.close()
            self._client = None
        self._status = ProtocolStatus.DISCONNECTED
        self._metrics.connected_since = None
        logger.info("Modbus disconnected")

    async def send(self, message: ProtocolMessage) -> None:
        """Write a register value."""
        if not self.is_connected or self._client is None:
            raise ConnectionError("Modbus not connected")

        address = message.payload.get("address")
        value = message.payload.get("value")
        unit = self._register_map.unit_id if self._register_map else 1

        if address is None or value is None:
            raise ValueError("Modbus send requires 'address' and 'value' in payload")

        await self.write_registers(int(address), [int(value)], unit=unit)

    # -- Register I/O (used by the setpoint writer) --------------------------

    def register(self, name: str) -> RegisterDefinition | None:
        """The named register of the loaded map (``None`` before connect)."""
        return self._register_map.registers.get(name) if self._register_map else None

    def _unit(self, unit: int | None) -> int:
        if unit is not None:
            return int(unit)
        return self._register_map.unit_id if self._register_map else 1

    async def write_registers(
        self, address: int, values: list[int], *, unit: int | None = None
    ) -> None:
        """Write raw 16-bit *values* starting at holding register *address*."""
        if not self.is_connected or self._client is None:
            raise ConnectionError("Modbus not connected")
        method: Any
        if len(values) == 1:
            method = self._client.write_register
            args: tuple[Any, ...] = (address, int(values[0]))
        else:
            method = self._client.write_registers
            args = (address, [int(v) for v in values])
        try:
            result = await method(*args, **{_unit_kwarg(method): self._unit(unit)})
        except Exception:
            self._metrics.errors += 1
            raise
        if result is not None and hasattr(result, "isError") and result.isError():
            self._metrics.errors += 1
            raise ModbusWriteError(f"write to register {address} rejected: {result}")
        self._metrics.messages_sent += 1
        self._metrics.last_message_at = time.time()

    async def read_holding(
        self, address: int, count: int = 1, *, unit: int | None = None
    ) -> list[int]:
        """Read *count* raw holding registers starting at *address*."""
        if not self.is_connected or self._client is None:
            raise ConnectionError("Modbus not connected")
        method = self._client.read_holding_registers
        try:
            result = await method(address, count=count, **{_unit_kwarg(method): self._unit(unit)})
        except Exception:
            self._metrics.errors += 1
            raise
        if result.isError():
            self._metrics.errors += 1
            raise ModbusWriteError(f"read of register {address} rejected: {result}")
        self._metrics.messages_received += 1
        return list(result.registers)

    async def receive(self) -> ProtocolMessage | None:
        """Return the latest polled values as a message."""
        if not self._latest_values:
            return None
        return ProtocolMessage(
            topic=f"modbus/{self._register_map.name if self._register_map else 'unknown'}",
            payload=dict(self._latest_values),
            source="modbus",
        )

    # -- Polling -------------------------------------------------------------

    async def poll_once(self) -> dict[str, float]:
        """Read all configured registers once and return name→value dict."""
        if not self.is_connected or self._client is None or self._register_map is None:
            return {}

        values: dict[str, float] = {}
        unit = self._register_map.unit_id

        for name, reg in self._register_map.registers.items():
            try:
                method: Any
                if reg.register_type == RegisterType.HOLDING:
                    method = self._client.read_holding_registers
                elif reg.register_type == RegisterType.INPUT:
                    method = self._client.read_input_registers
                else:
                    continue
                result = await method(reg.address, count=reg.count, **{_unit_kwarg(method): unit})

                if result.isError():
                    logger.warning("Modbus read error for %s: %s", name, result)
                    self._metrics.errors += 1
                    continue

                raw = self._decode_registers(result.registers, reg)
                values[name] = raw * reg.scale
            except Exception:
                logger.exception("Error reading register %s", name)
                self._metrics.errors += 1

        self._latest_values = values
        self._metrics.messages_received += 1
        self._metrics.last_message_at = time.time()

        # Dispatch to subscribers
        msg = ProtocolMessage(
            topic=f"modbus/{self._register_map.name}",
            payload=values,
            source="modbus",
        )
        await self._dispatch(msg)

        return values

    async def _poll_loop(self, interval: float) -> None:
        """Continuously poll registers at a fixed interval."""
        while self.is_connected:
            try:
                await self.poll_once()
            except Exception:
                logger.exception("Modbus poll error")
                self._metrics.errors += 1
            await asyncio.sleep(interval)

    # -- Decode helpers ------------------------------------------------------

    @staticmethod
    def _decode_registers(regs: list[int], defn: RegisterDefinition) -> float:
        """Decode raw register values based on data type."""
        if defn.data_type == "uint16":
            return float(regs[0])
        elif defn.data_type == "int16":
            val = regs[0]
            return float(val - 65536 if val >= 32768 else val)
        elif defn.data_type in ("uint32", "uint64"):
            val = 0
            for r in regs:
                val = (val << 16) | r
            return float(val)
        elif defn.data_type == "int32":
            val = (regs[0] << 16) | regs[1]
            if val >= 0x80000000:
                val -= 0x100000000
            return float(val)
        elif defn.data_type == "float32":
            raw = struct.pack(">HH", regs[0], regs[1] if len(regs) > 1 else 0)
            return float(struct.unpack(">f", raw)[0])
        return float(regs[0])
