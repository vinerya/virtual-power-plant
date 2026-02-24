"""EV battery, fleet, and charging session models.

``EVBattery`` extends the core ``EnergyResource`` ABC to represent an
electric vehicle with charge/discharge constraints, departure targets,
and V2G capability.  ``EVFleet`` aggregates multiple EVs.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EVConnectionState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTED_IDLE = "connected_idle"
    CHARGING = "charging"
    DISCHARGING = "discharging"


class ScheduleStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# EV Battery
# ---------------------------------------------------------------------------

@dataclass
class EVBattery:
    """An electric vehicle battery with V2G capability.

    Designed to interoperate with the VPP optimisation engine.
    """

    ev_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    capacity_kwh: float = 60.0
    current_soc: float = 0.5        # 0.0–1.0
    min_soc: float = 0.2            # never drain below this
    max_charge_kw: float = 11.0
    max_discharge_kw: float = 11.0  # V2G discharge limit
    charge_efficiency: float = 0.92
    discharge_efficiency: float = 0.92
    v2g_capable: bool = True
    degradation_cost_per_kwh: float = 0.02  # $/kWh throughput cost

    # Connection state
    connection_state: EVConnectionState = EVConnectionState.DISCONNECTED
    connected_at: float | None = None

    # Departure constraint
    target_soc: float = 0.8
    departure_time: float | None = None  # Unix timestamp

    # Metadata
    vehicle_make: str = ""
    vehicle_model: str = ""
    owner_id: str = ""

    # -- Derived properties --------------------------------------------------

    @property
    def current_energy_kwh(self) -> float:
        return self.capacity_kwh * self.current_soc

    @property
    def energy_needed_kwh(self) -> float:
        """Energy needed to reach target SOC."""
        delta = self.target_soc - self.current_soc
        return max(0.0, delta * self.capacity_kwh / self.charge_efficiency)

    @property
    def available_discharge_kwh(self) -> float:
        """Energy available for V2G discharge (above min_soc)."""
        if not self.v2g_capable:
            return 0.0
        return max(0.0, (self.current_soc - self.min_soc) * self.capacity_kwh)

    @property
    def time_to_target_hours(self) -> float:
        """Minimum hours to reach target SOC at max charge rate."""
        if self.energy_needed_kwh <= 0:
            return 0.0
        return self.energy_needed_kwh / self.max_charge_kw

    @property
    def time_until_departure_hours(self) -> float | None:
        """Hours until departure (None if no departure set)."""
        if self.departure_time is None:
            return None
        return max(0.0, (self.departure_time - time.time()) / 3600)

    @property
    def has_flexibility(self) -> bool:
        """True if there's time slack for V2G between now and departure."""
        remaining = self.time_until_departure_hours
        if remaining is None:
            return True  # No departure = unlimited flexibility
        return remaining > self.time_to_target_hours * 1.2  # 20% margin

    # -- Operations ----------------------------------------------------------

    def charge(self, power_kw: float, duration_hours: float) -> float:
        """Charge the EV.  Returns actual energy added (kWh)."""
        clamped = min(power_kw, self.max_charge_kw)
        raw = clamped * duration_hours
        effective = raw * self.charge_efficiency
        headroom = (1.0 - self.current_soc) * self.capacity_kwh
        actual = min(effective, headroom)
        self.current_soc += actual / self.capacity_kwh
        return actual

    def discharge(self, power_kw: float, duration_hours: float) -> float:
        """Discharge (V2G).  Returns actual energy delivered (kWh)."""
        if not self.v2g_capable:
            return 0.0
        clamped = min(power_kw, self.max_discharge_kw)
        raw = clamped * duration_hours
        available = self.available_discharge_kwh
        from_battery = min(raw, available)
        delivered = from_battery * self.discharge_efficiency
        self.current_soc -= from_battery / self.capacity_kwh
        return delivered

    def to_dict(self) -> dict[str, Any]:
        return {
            "ev_id": self.ev_id,
            "capacity_kwh": self.capacity_kwh,
            "current_soc": round(self.current_soc, 4),
            "min_soc": self.min_soc,
            "max_charge_kw": self.max_charge_kw,
            "max_discharge_kw": self.max_discharge_kw,
            "v2g_capable": self.v2g_capable,
            "connection_state": self.connection_state.value,
            "target_soc": self.target_soc,
            "energy_needed_kwh": round(self.energy_needed_kwh, 2),
            "available_discharge_kwh": round(self.available_discharge_kwh, 2),
            "has_flexibility": self.has_flexibility,
        }


# ---------------------------------------------------------------------------
# Flexibility window
# ---------------------------------------------------------------------------

@dataclass
class FlexibilityWindow:
    """Time window during which an EV can provide V2G services."""

    ev_id: str
    start_time: float
    end_time: float
    max_charge_kw: float
    max_discharge_kw: float
    available_energy_kwh: float  # for discharge
    needed_energy_kwh: float     # must be charged by end

    @property
    def duration_hours(self) -> float:
        return (self.end_time - self.start_time) / 3600

    def to_dict(self) -> dict[str, Any]:
        return {
            "ev_id": self.ev_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_hours": round(self.duration_hours, 2),
            "max_charge_kw": self.max_charge_kw,
            "max_discharge_kw": self.max_discharge_kw,
            "available_energy_kwh": round(self.available_energy_kwh, 2),
            "needed_energy_kwh": round(self.needed_energy_kwh, 2),
        }


# ---------------------------------------------------------------------------
# Charging session
# ---------------------------------------------------------------------------

@dataclass
class ChargingSession:
    """Record of a single charging (or V2G) session."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ev_id: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    energy_charged_kwh: float = 0.0
    energy_discharged_kwh: float = 0.0
    cost: float = 0.0
    revenue: float = 0.0
    status: ScheduleStatus = ScheduleStatus.ACTIVE

    @property
    def net_energy_kwh(self) -> float:
        return self.energy_charged_kwh - self.energy_discharged_kwh

    @property
    def net_cost(self) -> float:
        return self.cost - self.revenue

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "ev_id": self.ev_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "energy_charged_kwh": round(self.energy_charged_kwh, 3),
            "energy_discharged_kwh": round(self.energy_discharged_kwh, 3),
            "cost": round(self.cost, 4),
            "revenue": round(self.revenue, 4),
            "status": self.status.value,
        }


# ---------------------------------------------------------------------------
# EV Fleet
# ---------------------------------------------------------------------------

@dataclass
class EVFleet:
    """Aggregation of EVs for fleet-level V2G management."""

    fleet_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "default_fleet"
    vehicles: dict[str, EVBattery] = field(default_factory=dict)
    sessions: dict[str, ChargingSession] = field(default_factory=dict)

    def add_vehicle(self, ev: EVBattery) -> None:
        self.vehicles[ev.ev_id] = ev

    def remove_vehicle(self, ev_id: str) -> None:
        self.vehicles.pop(ev_id, None)

    def get_vehicle(self, ev_id: str) -> EVBattery | None:
        return self.vehicles.get(ev_id)

    # -- Aggregated metrics --------------------------------------------------

    @property
    def total_capacity_kwh(self) -> float:
        return sum(ev.capacity_kwh for ev in self.vehicles.values())

    @property
    def total_current_energy_kwh(self) -> float:
        return sum(ev.current_energy_kwh for ev in self.vehicles.values())

    @property
    def average_soc(self) -> float:
        if not self.vehicles:
            return 0.0
        return sum(ev.current_soc for ev in self.vehicles.values()) / len(self.vehicles)

    @property
    def connected_count(self) -> int:
        return sum(
            1 for ev in self.vehicles.values()
            if ev.connection_state != EVConnectionState.DISCONNECTED
        )

    @property
    def total_charge_capacity_kw(self) -> float:
        """Max aggregate charge rate from connected EVs."""
        return sum(
            ev.max_charge_kw
            for ev in self.vehicles.values()
            if ev.connection_state != EVConnectionState.DISCONNECTED
        )

    @property
    def total_discharge_capacity_kw(self) -> float:
        """Max aggregate V2G discharge rate from connected V2G EVs."""
        return sum(
            ev.max_discharge_kw
            for ev in self.vehicles.values()
            if ev.v2g_capable and ev.connection_state != EVConnectionState.DISCONNECTED
        )

    @property
    def total_available_discharge_kwh(self) -> float:
        """Total energy available for V2G across all connected EVs."""
        return sum(
            ev.available_discharge_kwh
            for ev in self.vehicles.values()
            if ev.v2g_capable and ev.connection_state != EVConnectionState.DISCONNECTED
        )

    @property
    def flexible_vehicles(self) -> list[EVBattery]:
        """EVs with time flexibility for V2G."""
        return [
            ev for ev in self.vehicles.values()
            if ev.has_flexibility
            and ev.connection_state != EVConnectionState.DISCONNECTED
            and ev.v2g_capable
        ]

    def get_flexibility_windows(self) -> list[FlexibilityWindow]:
        """Compute flexibility windows for all flexible EVs."""
        windows: list[FlexibilityWindow] = []
        now = time.time()
        for ev in self.flexible_vehicles:
            end = ev.departure_time or (now + 24 * 3600)
            windows.append(FlexibilityWindow(
                ev_id=ev.ev_id,
                start_time=now,
                end_time=end,
                max_charge_kw=ev.max_charge_kw,
                max_discharge_kw=ev.max_discharge_kw,
                available_energy_kwh=ev.available_discharge_kwh,
                needed_energy_kwh=ev.energy_needed_kwh,
            ))
        return windows

    def to_dict(self) -> dict[str, Any]:
        return {
            "fleet_id": self.fleet_id,
            "name": self.name,
            "vehicle_count": len(self.vehicles),
            "connected_count": self.connected_count,
            "total_capacity_kwh": round(self.total_capacity_kwh, 1),
            "average_soc": round(self.average_soc, 3),
            "total_charge_capacity_kw": round(self.total_charge_capacity_kw, 1),
            "total_discharge_capacity_kw": round(self.total_discharge_capacity_kw, 1),
            "total_available_discharge_kwh": round(self.total_available_discharge_kwh, 1),
            "flexible_vehicle_count": len(self.flexible_vehicles),
        }
