"""OCPP 1.6 / 2.0.1 adapter for EV charger management.

Manages ChargePoints, charging profiles, remote start/stop, and V2G
discharge profiles.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

from vpp.protocols.base import (
    ProtocolAdapter,
    ProtocolMessage,
    ProtocolStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OCPP value objects
# ---------------------------------------------------------------------------

class ChargePointStatus(str, Enum):
    AVAILABLE = "Available"
    PREPARING = "Preparing"
    CHARGING = "Charging"
    SUSPENDED_EVSE = "SuspendedEVSE"
    SUSPENDED_EV = "SuspendedEV"
    FINISHING = "Finishing"
    RESERVED = "Reserved"
    UNAVAILABLE = "Unavailable"
    FAULTED = "Faulted"


class ChargingProfilePurpose(str, Enum):
    CHARGE_POINT_MAX = "ChargePointMaxProfile"
    TX_DEFAULT = "TxDefaultProfile"
    TX_PROFILE = "TxProfile"


class ChargingRateUnit(str, Enum):
    WATTS = "W"
    AMPS = "A"


@dataclass
class ChargingSchedulePeriod:
    """One period within a charging profile."""

    start_period: int  # seconds from profile start
    limit: float       # power in W or current in A
    number_phases: int = 3


@dataclass
class ChargingProfile:
    """An OCPP charging profile that controls charger output."""

    profile_id: int
    stack_level: int = 0
    purpose: ChargingProfilePurpose = ChargingProfilePurpose.TX_PROFILE
    kind: str = "Absolute"  # Absolute | Recurring | Relative
    rate_unit: ChargingRateUnit = ChargingRateUnit.WATTS
    schedule: list[ChargingSchedulePeriod] = field(default_factory=list)
    valid_from: float | None = None
    valid_to: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chargingProfileId": self.profile_id,
            "stackLevel": self.stack_level,
            "chargingProfilePurpose": self.purpose.value,
            "chargingProfileKind": self.kind,
            "chargingSchedule": {
                "chargingRateUnit": self.rate_unit.value,
                "chargingSchedulePeriod": [
                    {
                        "startPeriod": p.start_period,
                        "limit": p.limit,
                        "numberPhases": p.number_phases,
                    }
                    for p in self.schedule
                ],
            },
        }


@dataclass
class ChargePoint:
    """Representation of an OCPP ChargePoint (EV charger)."""

    charge_point_id: str
    status: ChargePointStatus = ChargePointStatus.AVAILABLE
    vendor: str = ""
    model: str = ""
    serial_number: str = ""
    firmware_version: str = ""
    num_connectors: int = 1
    max_power_kw: float = 22.0
    current_power_kw: float = 0.0
    current_soc: float | None = None
    v2g_capable: bool = False
    active_transaction_id: str | None = None
    active_profile: ChargingProfile | None = None
    last_heartbeat: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "charge_point_id": self.charge_point_id,
            "status": self.status.value,
            "vendor": self.vendor,
            "model": self.model,
            "max_power_kw": self.max_power_kw,
            "current_power_kw": self.current_power_kw,
            "current_soc": self.current_soc,
            "v2g_capable": self.v2g_capable,
            "active_transaction_id": self.active_transaction_id,
            "num_connectors": self.num_connectors,
        }


# Callback type for charge-point status changes
CPStatusCallback = Callable[[ChargePoint, ChargePointStatus], Awaitable[None]]


class OCPPAdapter(ProtocolAdapter):
    """OCPP 1.6/2.0.1 central system adapter.

    Configuration keys:
        ws_host, ws_port, ocpp_version ("1.6" | "2.0.1"),
        heartbeat_interval_s, auto_accept_boot
    """

    def __init__(self) -> None:
        super().__init__("ocpp", "1.6")
        self._charge_points: dict[str, ChargePoint] = {}
        self._status_callbacks: list[CPStatusCallback] = []
        self._server: Any = None
        self._message_queue: asyncio.Queue[ProtocolMessage] = asyncio.Queue(maxsize=500)

    # -- Lifecycle -----------------------------------------------------------

    async def connect(self) -> None:
        """Start the OCPP WebSocket central system."""
        self._status = ProtocolStatus.CONNECTING

        # In production, this starts a WebSocket server that ChargePoints
        # connect to.  For the MVP, we manage an in-memory registry of
        # charge points.
        self._status = ProtocolStatus.CONNECTED
        self._metrics.connected_since = time.time()

        version = self._config.get("ocpp_version", "1.6")
        self.version = version
        logger.info("OCPP central system started (v%s)", version)

    async def disconnect(self) -> None:
        if self._server is not None:
            self._server.close()
            self._server = None
        self._status = ProtocolStatus.DISCONNECTED
        self._metrics.connected_since = None
        logger.info("OCPP central system stopped")

    async def send(self, message: ProtocolMessage) -> None:
        if not self.is_connected:
            raise ConnectionError("OCPP not connected")

        cp_id = message.payload.get("charge_point_id")
        action = message.payload.get("action", "")

        if action == "RemoteStartTransaction":
            await self.remote_start(cp_id)
        elif action == "RemoteStopTransaction":
            await self.remote_stop(cp_id)
        elif action == "SetChargingProfile":
            profile_data = message.payload.get("profile", {})
            await self.set_charging_profile(cp_id, profile_data)
        else:
            logger.warning("Unknown OCPP action: %s", action)

        self._metrics.messages_sent += 1
        self._metrics.last_message_at = time.time()

    async def receive(self) -> ProtocolMessage | None:
        try:
            return self._message_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    # -- ChargePoint management ----------------------------------------------

    def register_charge_point(self, cp: ChargePoint) -> None:
        """Register a charge point in the central system."""
        self._charge_points[cp.charge_point_id] = cp
        logger.info("Registered charge point: %s", cp.charge_point_id)

    def unregister_charge_point(self, cp_id: str) -> None:
        self._charge_points.pop(cp_id, None)

    def get_charge_point(self, cp_id: str) -> ChargePoint | None:
        return self._charge_points.get(cp_id)

    def list_charge_points(self) -> list[ChargePoint]:
        return list(self._charge_points.values())

    def on_status_change(self, callback: CPStatusCallback) -> None:
        """Register a callback for charge-point status changes."""
        self._status_callbacks.append(callback)

    # -- OCPP operations -----------------------------------------------------

    async def remote_start(self, cp_id: str, connector_id: int = 1) -> bool:
        """Send RemoteStartTransaction to a charge point."""
        cp = self._charge_points.get(cp_id)
        if cp is None:
            logger.warning("Unknown charge point: %s", cp_id)
            return False

        if cp.status != ChargePointStatus.AVAILABLE:
            logger.warning("Charge point %s is %s, cannot start", cp_id, cp.status.value)
            return False

        cp.active_transaction_id = str(uuid.uuid4())
        await self._update_cp_status(cp, ChargePointStatus.CHARGING)
        logger.info("Remote start on %s (tx=%s)", cp_id, cp.active_transaction_id)
        return True

    async def remote_stop(self, cp_id: str) -> bool:
        """Send RemoteStopTransaction to a charge point."""
        cp = self._charge_points.get(cp_id)
        if cp is None:
            return False

        if cp.active_transaction_id is None:
            logger.warning("No active transaction on %s", cp_id)
            return False

        cp.active_transaction_id = None
        cp.current_power_kw = 0.0
        await self._update_cp_status(cp, ChargePointStatus.AVAILABLE)
        logger.info("Remote stop on %s", cp_id)
        return True

    async def set_charging_profile(
        self,
        cp_id: str,
        profile: ChargingProfile | dict[str, Any],
    ) -> bool:
        """Apply a charging profile to a charge point."""
        cp = self._charge_points.get(cp_id)
        if cp is None:
            return False

        if isinstance(profile, dict):
            # Build from dict
            periods = [
                ChargingSchedulePeriod(**p)
                for p in profile.get("schedule", [])
            ]
            profile = ChargingProfile(
                profile_id=profile.get("profile_id", 1),
                schedule=periods,
            )

        cp.active_profile = profile
        logger.info("Set charging profile on %s (id=%d)", cp_id, profile.profile_id)

        msg = ProtocolMessage(
            topic=f"ocpp/profile/{cp_id}",
            payload={"charge_point_id": cp_id, "profile": profile.to_dict()},
            source="ocpp",
        )
        await self._dispatch(msg)
        return True

    async def set_v2g_discharge_profile(
        self,
        cp_id: str,
        discharge_power_kw: float,
        duration_minutes: int = 60,
    ) -> bool:
        """Create a V2G discharge profile for a V2G-capable charger."""
        cp = self._charge_points.get(cp_id)
        if cp is None or not cp.v2g_capable:
            logger.warning("Charge point %s not V2G capable", cp_id)
            return False

        profile = ChargingProfile(
            profile_id=100,
            purpose=ChargingProfilePurpose.TX_PROFILE,
            schedule=[
                ChargingSchedulePeriod(
                    start_period=0,
                    limit=-discharge_power_kw * 1000,  # negative = discharge, in W
                    number_phases=3,
                ),
                ChargingSchedulePeriod(
                    start_period=duration_minutes * 60,
                    limit=0.0,
                    number_phases=3,
                ),
            ],
        )
        return await self.set_charging_profile(cp_id, profile)

    # -- Fleet-level queries -------------------------------------------------

    def total_charging_power(self) -> float:
        """Sum of all active charging power in kW."""
        return sum(cp.current_power_kw for cp in self._charge_points.values())

    def available_v2g_capacity(self) -> float:
        """Total V2G discharge capacity from available V2G chargers in kW."""
        return sum(
            cp.max_power_kw
            for cp in self._charge_points.values()
            if cp.v2g_capable and cp.status == ChargePointStatus.CHARGING
        )

    # -- Internal ------------------------------------------------------------

    async def _update_cp_status(self, cp: ChargePoint, new_status: ChargePointStatus) -> None:
        old = cp.status
        cp.status = new_status
        for cb in self._status_callbacks:
            try:
                await cb(cp, old)
            except Exception:
                logger.exception("Status callback error for %s", cp.charge_point_id)

        msg = ProtocolMessage(
            topic=f"ocpp/status/{cp.charge_point_id}",
            payload={"charge_point_id": cp.charge_point_id, "old": old.value, "new": new_status.value},
            source="ocpp",
        )
        await self._dispatch(msg)
        try:
            self._message_queue.put_nowait(msg)
        except asyncio.QueueFull:
            self._metrics.errors += 1
