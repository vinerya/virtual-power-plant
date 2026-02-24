"""Modbus TCP/RTU adapter for inverter and meter communication.

Supports predefined register maps for common inverters (SMA, Fronius,
SolarEdge) and a generic mode for custom register definitions.
"""

from __future__ import annotations

import asyncio
import logging
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
            "ac_power": RegisterDefinition(30775, 2, RegisterType.INPUT, "AC Power", "W", 1.0, "int32"),
            "dc_power": RegisterDefinition(30773, 2, RegisterType.INPUT, "DC Power", "W", 1.0, "int32"),
            "daily_yield": RegisterDefinition(30517, 4, RegisterType.INPUT, "Daily Yield", "Wh", 1.0, "uint64"),
            "total_yield": RegisterDefinition(30513, 4, RegisterType.INPUT, "Total Yield", "Wh", 1.0, "uint64"),
            "grid_frequency": RegisterDefinition(30803, 2, RegisterType.INPUT, "Grid Freq", "Hz", 0.01, "uint32"),
        },
    ),
    "fronius_symo": RegisterMap(
        name="Fronius Symo",
        registers={
            "ac_power": RegisterDefinition(40092, 1, RegisterType.HOLDING, "AC Power", "W", 1.0, "float32"),
            "ac_energy": RegisterDefinition(40094, 2, RegisterType.HOLDING, "AC Energy", "Wh", 1.0, "float32"),
            "dc_power": RegisterDefinition(40101, 1, RegisterType.HOLDING, "DC Power", "W", 1.0, "float32"),
            "frequency": RegisterDefinition(40086, 1, RegisterType.HOLDING, "Frequency", "Hz", 1.0, "float32"),
        },
    ),
    "solaredge_se": RegisterMap(
        name="SolarEdge SE",
        registers={
            "ac_power": RegisterDefinition(40084, 1, RegisterType.HOLDING, "AC Power", "W", 1.0, "int16"),
            "ac_power_scale": RegisterDefinition(40085, 1, RegisterType.HOLDING, "AC Power Scale", "", 1.0, "int16"),
            "dc_power": RegisterDefinition(40101, 1, RegisterType.HOLDING, "DC Power", "W", 1.0, "int16"),
            "temperature": RegisterDefinition(40104, 1, RegisterType.HOLDING, "Temperature", "°C", 0.01, "int16"),
            "ac_energy": RegisterDefinition(40094, 2, RegisterType.HOLDING, "AC Energy", "Wh", 1.0, "uint32"),
        },
    ),
    "generic_meter": RegisterMap(
        name="Generic Power Meter",
        registers={
            "voltage_l1": RegisterDefinition(0, 2, RegisterType.INPUT, "Voltage L1", "V", 0.1, "float32"),
            "voltage_l2": RegisterDefinition(2, 2, RegisterType.INPUT, "Voltage L2", "V", 0.1, "float32"),
            "voltage_l3": RegisterDefinition(4, 2, RegisterType.INPUT, "Voltage L3", "V", 0.1, "float32"),
            "current_l1": RegisterDefinition(6, 2, RegisterType.INPUT, "Current L1", "A", 0.01, "float32"),
            "power_total": RegisterDefinition(12, 2, RegisterType.INPUT, "Total Power", "W", 1.0, "float32"),
            "energy_total": RegisterDefinition(72, 2, RegisterType.INPUT, "Total Energy", "kWh", 0.1, "float32"),
        },
    ),
}


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
            from pymodbus.client import AsyncModbusTcpClient, AsyncModbusSerialClient
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
            self._register_map = INVERTER_MAPS[profile]
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
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
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

        await self._client.write_register(address, int(value), slave=unit)
        self._metrics.messages_sent += 1
        self._metrics.last_message_at = time.time()

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
                if reg.register_type == RegisterType.HOLDING:
                    result = await self._client.read_holding_registers(
                        reg.address, reg.count, slave=unit,
                    )
                elif reg.register_type == RegisterType.INPUT:
                    result = await self._client.read_input_registers(
                        reg.address, reg.count, slave=unit,
                    )
                else:
                    continue

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
            import struct
            raw = struct.pack(">HH", regs[0], regs[1] if len(regs) > 1 else 0)
            return struct.unpack(">f", raw)[0]
        return float(regs[0])
