"""Modbus TCP/RTU adapter for inverter and meter communication.

Supports predefined register maps for common inverters (SMA, Fronius,
SolarEdge) and a generic mode for custom register definitions.

Registers are **read-only unless flagged** ``writable=True``: the setpoint
writer (:mod:`vpp.protocols.modbus_control`) refuses to write a named
register that is not flagged. The writable definitions shipped here are the
SunSpec *model-relative* control blocks built by
:func:`sunspec_model_123_registers` (Immediate Controls, ``WMaxLimPct``) and
:func:`sunspec_model_124_registers` (Storage). SunSpec models sit at a
device-specific address (after the preceding models in the chain), so these
take the address of the model's ``ID`` register instead of guessing a
vendor's absolute addresses; :func:`discover_sunspec_models` finds it by
walking the chain from the ``"SunS"`` marker.

Every SunSpec register shipped here (the model 123/124 control blocks and the
Fronius model 113 / SolarEdge model 103 read maps) records its SunSpec point
and is checked against the official SunSpec model definitions (as bundled
with pysunspec2 1.3.6) by ``tests/test_sunspec_models.py``: offset, size,
type, units, access and scale-factor pairing. They have **not** been tested
against physical hardware. Polling honours the SunSpec conventions: a value
paired with a ``sunssf`` register is multiplied by ``10 ** sf``, and
"not implemented" sentinels (0x8000 int16/sunssf, 0xFFFF uint16/enum16,
0x80000000 int32, 0xFFFFFFFF uint32, 0 acc32, NaN float32) are dropped
instead of being reported as readings.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import math
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
    data_type: str = "uint16"  # uint16, int16, uint32, int32, acc32, uint64, float32
    writable: bool = False  # only flagged registers may be written by the setpoint writer
    # Name (in the same map) of a SunSpec ``sunssf`` register: the polled
    # value is multiplied by ``10 ** sf`` (and by ``scale``).
    scale_factor: str | None = None
    # SunSpec point this register maps to; enables the SunSpec
    # "not implemented" sentinels (0x8000 int16, 0xFFFF uint16, ...).
    sunspec: SunSpecPointRef | None = None

    def __post_init__(self) -> None:
        if isinstance(self.sunspec, dict):  # custom_registers from JSON config
            self.sunspec = SunSpecPointRef(**self.sunspec)


@dataclass(frozen=True)
class SunSpecPointRef:
    """Where a register comes from in the SunSpec information model."""

    model: int  # model ID, e.g. 103
    base: int  # wire address of that model's ``ID`` register
    point: str  # SunSpec point name, e.g. ``"W_SF"``


@dataclass
class RegisterMap:
    """Collection of registers for a device profile."""

    name: str
    registers: dict[str, RegisterDefinition]
    unit_id: int = 1


def _sunspec_reg(
    model: int,
    base: int,
    point: str,
    offset: int,
    name: str,
    unit: str,
    data_type: str,
    *,
    count: int = 1,
    scale_factor: str | None = None,
    writable: bool = False,
) -> RegisterDefinition:
    """A holding register for SunSpec *point* at ``base + offset``.

    *base* is the wire (0-based) address of the model's ``ID`` register and
    *offset* the point's offset from it (``ID`` = 0, ``L`` = 1, first data
    point = 2), exactly as in the SunSpec model definitions.
    """
    return RegisterDefinition(
        base + offset,
        count,
        RegisterType.HOLDING,
        name,
        unit,
        1.0,
        data_type,
        writable=writable,
        scale_factor=scale_factor,
        sunspec=SunSpecPointRef(model, base, point),
    )


# Wire address of the inverter model's ``ID`` register on Fronius and
# SolarEdge devices: "SunS" at 40000-40001, common model 1 at 40002 with
# L = 65 (both vendors document L = 65), so the next model starts at 40069.
# Vendor register tables list this as register *number* 40070 (1-based).
# Other firmware may differ: confirm with :func:`discover_sunspec_models`.
SUNSPEC_INVERTER_BASE = 40069
_INV = SUNSPEC_INVERTER_BASE

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
    # Fronius Symo in its default "float" SunSpec mode: model 113
    # (inverter_three_phase_float). Offsets verified against the SunSpec
    # model 113 definition; the absolute base is per Fronius documentation.
    "fronius_symo": RegisterMap(
        name="Fronius Symo",
        registers={
            "ac_power": _sunspec_reg(113, _INV, "W", 22, "AC Power", "W", "float32", count=2),
            "frequency": _sunspec_reg(113, _INV, "Hz", 24, "Frequency", "Hz", "float32", count=2),
            "ac_energy": _sunspec_reg(113, _INV, "WH", 32, "AC Energy", "Wh", "float32", count=2),
            "dc_power": _sunspec_reg(113, _INV, "DCW", 38, "DC Power", "W", "float32", count=2),
        },
    ),
    # SolarEdge SE inverters: model 101/102/103 ("int + SF"; the three share
    # one layout). Offsets verified against the SunSpec model 103 definition;
    # every value is paired with its sunssf scale-factor register.
    "solaredge_se": RegisterMap(
        name="SolarEdge SE",
        registers={
            "ac_power": _sunspec_reg(
                103, _INV, "W", 14, "AC Power", "W", "int16", scale_factor="ac_power_scale"
            ),
            "ac_power_scale": _sunspec_reg(103, _INV, "W_SF", 15, "AC Power Scale", "", "int16"),
            "frequency": _sunspec_reg(
                103, _INV, "Hz", 16, "Frequency", "Hz", "uint16", scale_factor="frequency_scale"
            ),
            "frequency_scale": _sunspec_reg(
                103, _INV, "Hz_SF", 17, "Frequency Scale", "", "int16"
            ),
            "ac_energy": _sunspec_reg(
                103,
                _INV,
                "WH",
                24,
                "AC Energy",
                "Wh",
                "acc32",
                count=2,
                scale_factor="ac_energy_scale",
            ),
            "ac_energy_scale": _sunspec_reg(
                103, _INV, "WH_SF", 26, "AC Energy Scale", "", "int16"
            ),
            "dc_power": _sunspec_reg(
                103, _INV, "DCW", 31, "DC Power", "W", "int16", scale_factor="dc_power_scale"
            ),
            "dc_power_scale": _sunspec_reg(103, _INV, "DCW_SF", 32, "DC Power Scale", "", "int16"),
            "temperature": _sunspec_reg(
                103,
                _INV,
                "TmpSnk",
                34,
                "Heat Sink Temperature",
                "°C",
                "int16",
                scale_factor="temperature_scale",
            ),
            "temperature_scale": _sunspec_reg(
                103, _INV, "Tmp_SF", 37, "Temperature Scale", "", "int16"
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
#
# Offsets, sizes, types, units, access and scale-factor pairings below are
# checked point by point against the SunSpec information model definitions
# (the JSON models bundled with pysunspec2 1.3.6, i.e. sunspec/models) by
# tests/test_sunspec_models.py. enum16/bitfield16 points are read and written
# as uint16 and sunssf points as int16.

SUNSPEC_MODEL_123_LENGTH = 24  # L of model 123 (points after ID and L)
SUNSPEC_MODEL_124_LENGTH = 24  # L of model 124


def sunspec_model_123_registers(base: int) -> dict[str, RegisterDefinition]:
    """SunSpec model 123 *Immediate Controls* at *base* (address of its ``ID``).

    Offsets per the SunSpec model 123 definition (``ID``=123, ``L``=24):
    ``WMaxLimPct`` (+5, uint16 RW, % of ``WMax`` scaled by ``WMaxLimPct_SF``),
    ``WMaxLimPct_RvrtTms`` (+7, uint16 RW seconds, device-side revert
    timeout), ``WMaxLim_Ena`` (+9, enum16 RW: 0 DISABLED / 1 ENABLED) and
    ``WMaxLimPct_SF`` (+23, sunssf: int16 power-of-ten exponent, read-only).
    """
    return {
        "sunspec_123_id": _sunspec_reg(123, base, "ID", 0, "Model ID (123)", "", "uint16"),
        "sunspec_123_length": _sunspec_reg(123, base, "L", 1, "Model length", "", "uint16"),
        "wmax_lim_pct": _sunspec_reg(
            123,
            base,
            "WMaxLimPct",
            5,
            "WMaxLimPct",
            "%",
            "uint16",
            scale_factor="wmax_lim_pct_sf",
            writable=True,
        ),
        "wmax_lim_pct_rvrt_tms": _sunspec_reg(
            123, base, "WMaxLimPct_RvrtTms", 7, "WMaxLimPct_RvrtTms", "s", "uint16", writable=True
        ),
        "wmax_lim_ena": _sunspec_reg(
            123, base, "WMaxLim_Ena", 9, "WMaxLim_Ena", "", "uint16", writable=True
        ),
        "wmax_lim_pct_sf": _sunspec_reg(
            123, base, "WMaxLimPct_SF", 23, "WMaxLimPct_SF", "", "int16"
        ),
    }


def sunspec_model_124_registers(base: int) -> dict[str, RegisterDefinition]:
    """SunSpec model 124 *Storage* at *base* (address of its ``ID``).

    Offsets per the SunSpec model 124 definition (``ID``=124, ``L``=24):
    ``StorCtl_Mod`` (+5, bitfield16 RW: bit 0 CHARGE, bit 1 DISCHARGE limit
    active), ``OutWRte`` (+12, int16 RW, % of ``WDisChaMax``), ``InWRte``
    (+13, int16 RW, % of ``WChaMax``), ``InOutWRte_RvrtTms`` (+15, uint16 RW
    seconds) and ``InOutWRte_SF`` (+25, sunssf). The register layout is
    verified against the spec; how devices interpret *forced*
    charge/discharge (negative ``InWRte``/``OutWRte``) differs between
    vendors and is not verified, hence the profile stays flagged unverified.
    """
    return {
        "sunspec_124_id": _sunspec_reg(124, base, "ID", 0, "Model ID (124)", "", "uint16"),
        "sunspec_124_length": _sunspec_reg(124, base, "L", 1, "Model length", "", "uint16"),
        "stor_ctl_mod": _sunspec_reg(
            124, base, "StorCtl_Mod", 5, "StorCtl_Mod", "", "uint16", writable=True
        ),
        "out_w_rte": _sunspec_reg(
            124,
            base,
            "OutWRte",
            12,
            "OutWRte",
            "%",
            "int16",
            scale_factor="in_out_w_rte_sf",
            writable=True,
        ),
        "in_w_rte": _sunspec_reg(
            124,
            base,
            "InWRte",
            13,
            "InWRte",
            "%",
            "int16",
            scale_factor="in_out_w_rte_sf",
            writable=True,
        ),
        "in_out_w_rte_rvrt_tms": _sunspec_reg(
            124, base, "InOutWRte_RvrtTms", 15, "InOutWRte_RvrtTms", "s", "uint16", writable=True
        ),
        "in_out_w_rte_sf": _sunspec_reg(
            124, base, "InOutWRte_SF", 25, "InOutWRte_SF", "", "int16"
        ),
    }


# ---------------------------------------------------------------------------
# SunSpec conventions: "not implemented" values and model discovery
# ---------------------------------------------------------------------------

#: "SunS" (0x53756e53) marks the start of a SunSpec register map.
SUNSPEC_MARKER = (0x5375, 0x6E53)
#: Base addresses a SunSpec client probes for the marker, in order.
SUNSPEC_BASE_ADDRESSES = (40000, 0, 50000)
#: Model ID of the end model that terminates the chain.
SUNSPEC_END_MODEL_ID = 0xFFFF

# Raw register values that mean "not implemented" per SunSpec data type.
# (sunssf and int16 share 0x8000, enum16/bitfield16 share uint16's 0xFFFF;
# accumulators use 0; float32 uses NaN, checked separately.)
_SUNSPEC_NOT_IMPLEMENTED: dict[str, int] = {
    "int16": 0x8000,
    "uint16": 0xFFFF,
    "int32": 0x80000000,
    "uint32": 0xFFFFFFFF,
    "acc32": 0,
    "uint64": 0xFFFFFFFFFFFFFFFF,
}


def _raw_int(regs: list[int]) -> int:
    val = 0
    for r in regs:
        val = (val << 16) | (r & 0xFFFF)
    return val


def sunspec_not_implemented(regs: list[int], data_type: str) -> bool:
    """True when *regs* hold SunSpec's "not implemented" value for *data_type*."""
    if data_type == "float32":
        if len(regs) < 2:
            return True
        value = float(
            struct.unpack(">f", struct.pack(">HH", regs[0] & 0xFFFF, regs[1] & 0xFFFF))[0]
        )
        return math.isnan(value)  # 0x7FC00000 and other NaN encodings
    sentinel = _SUNSPEC_NOT_IMPLEMENTED.get(data_type)
    return sentinel is not None and _raw_int(regs) == sentinel


@dataclass(frozen=True)
class SunSpecModelHeader:
    """One model in a device's SunSpec chain."""

    model_id: int
    address: int  # wire address of the model's ID register
    length: int  # L: registers following ID and L


class SunSpecDiscoveryError(RuntimeError):
    """The device does not expose a readable SunSpec register map."""


async def discover_sunspec_models(
    io: Any,
    *,
    unit: int | None = None,
    base_addresses: tuple[int, ...] = SUNSPEC_BASE_ADDRESSES,
    max_models: int = 256,
) -> list[SunSpecModelHeader]:
    """Walk a device's SunSpec model chain (like pysunspec2's device scan).

    *io* needs ``read_holding(address, count, unit=...)`` (a connected
    :class:`ModbusAdapter` does). The ``"SunS"`` marker is looked for at
    40000, 0 and 50000; the chain then starts two registers later and each
    model header is ``ID`` then ``L``, the next model starting ``L + 2``
    registers on, until the end model ``0xFFFF``. A read failure after at
    least one model ends the walk (some devices omit the end model).
    """
    base: int | None = None
    errors: list[str] = []
    for candidate in base_addresses:
        try:
            marker = await io.read_holding(candidate, 2, unit=unit)
        except Exception as exc:  # no register there -> try the next base
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
            continue
        if tuple(marker[:2]) == SUNSPEC_MARKER:
            base = candidate
            break
        errors.append(f"{candidate}: no SunS marker")
    if base is None:
        raise SunSpecDiscoveryError("no SunSpec map found (" + "; ".join(errors) + ")")

    models: list[SunSpecModelHeader] = []
    addr = base + 2
    for _ in range(max_models):
        try:
            [model_id] = await io.read_holding(addr, 1, unit=unit)
        except Exception:
            if models:
                logger.warning("SunSpec chain ends without end model at %d", addr)
                return models
            raise
        if model_id == SUNSPEC_END_MODEL_ID:
            return models
        [length] = await io.read_holding(addr + 1, 1, unit=unit)
        models.append(SunSpecModelHeader(int(model_id), addr, int(length)))
        addr += int(length) + 2
        if addr > 0xFFFF:
            raise SunSpecDiscoveryError("SunSpec model chain runs past the register space")
    raise SunSpecDiscoveryError(f"SunSpec model chain longer than {max_models} models")


def find_sunspec_model(
    models: list[SunSpecModelHeader], model_id: int
) -> SunSpecModelHeader | None:
    """First model with *model_id* in a discovered chain (``None`` if absent)."""
    return next((m for m in models if m.model_id == model_id), None)


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
    "acc32": (0, 0xFFFFFFFF, 2),
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
    if data_type == "uint64":
        return 4
    return 2 if data_type in ("uint32", "int32", "acc32", "float32") else 1


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

        decoded: dict[str, float] = {}
        unit = self._register_map.unit_id
        registers = self._register_map.registers

        for name, reg in registers.items():
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

                regs = list(result.registers)
                if reg.sunspec is not None and sunspec_not_implemented(regs, reg.data_type):
                    logger.debug("SunSpec register %s not implemented by the device", name)
                    continue
                raw = self._decode_registers(regs, reg)
                if math.isnan(raw):  # float32 NaN: no reading
                    continue
                decoded[name] = raw
            except Exception:
                logger.exception("Error reading register %s", name)
                self._metrics.errors += 1

        values: dict[str, float] = {}
        for name, raw in decoded.items():
            reg = registers[name]
            if reg.scale_factor is not None:
                # SunSpec value = raw * 10**sf; without a valid sf there is no value.
                sf = decoded.get(reg.scale_factor)
                if sf is None or not -10 <= sf <= 10:
                    logger.debug("No valid scale factor %s for %s", reg.scale_factor, name)
                    continue
                raw = raw * 10.0 ** int(sf)
            values[name] = raw * reg.scale

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
        elif defn.data_type in ("uint32", "acc32", "uint64"):
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
