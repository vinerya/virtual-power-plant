"""V2G Aggregator — aggregate fleet flexibility for grid services.

The aggregator sits between the EV fleet and the grid operator, presenting
the fleet as a single virtual resource.  It handles flexibility assessment,
ancillary-service bidding, and dispatch signal distribution.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from vpp.v2g.models import EVBattery, EVFleet, EVConnectionState
from vpp.v2g.scheduler import V2GScheduler, V2GScheduleResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grid service types
# ---------------------------------------------------------------------------

class GridService(str, Enum):
    FREQUENCY_REGULATION = "frequency_regulation"
    SPINNING_RESERVE = "spinning_reserve"
    PEAK_SHAVING = "peak_shaving"
    ENERGY_ARBITRAGE = "energy_arbitrage"
    DEMAND_RESPONSE = "demand_response"
    VOLTAGE_SUPPORT = "voltage_support"


@dataclass
class FlexibilityBid:
    """A bid for grid ancillary services from the V2G fleet."""

    service: GridService
    capacity_kw: float
    duration_hours: float
    price_per_kw: float
    available_from: float = field(default_factory=time.time)
    available_until: float = 0.0
    fleet_id: str = ""
    ev_ids: list[str] = field(default_factory=list)

    @property
    def total_value(self) -> float:
        return self.capacity_kw * self.price_per_kw * self.duration_hours

    def to_dict(self) -> dict[str, Any]:
        return {
            "service": self.service.value,
            "capacity_kw": round(self.capacity_kw, 1),
            "duration_hours": self.duration_hours,
            "price_per_kw": self.price_per_kw,
            "total_value": round(self.total_value, 2),
            "available_from": self.available_from,
            "available_until": self.available_until,
            "ev_count": len(self.ev_ids),
        }


@dataclass
class DispatchSignal:
    """A dispatch signal to be distributed across fleet EVs."""

    target_power_kw: float      # positive = charge fleet, negative = discharge
    duration_seconds: int = 900  # 15 min default
    service: GridService = GridService.ENERGY_ARBITRAGE
    priority: int = 0            # higher = more urgent

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_power_kw": self.target_power_kw,
            "duration_seconds": self.duration_seconds,
            "service": self.service.value,
            "priority": self.priority,
        }


@dataclass
class DispatchResult:
    """Result of dispatching a signal across the fleet."""

    target_power_kw: float
    achieved_power_kw: float
    ev_allocations: dict[str, float] = field(default_factory=dict)  # ev_id -> kW
    shortfall_kw: float = 0.0

    @property
    def achievement_ratio(self) -> float:
        if self.target_power_kw == 0:
            return 1.0
        return abs(self.achieved_power_kw) / abs(self.target_power_kw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_power_kw": round(self.target_power_kw, 1),
            "achieved_power_kw": round(self.achieved_power_kw, 1),
            "achievement_ratio": round(self.achievement_ratio, 3),
            "shortfall_kw": round(self.shortfall_kw, 1),
            "ev_count": len(self.ev_allocations),
        }


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class V2GAggregator:
    """Aggregate EV fleet flexibility and dispatch V2G signals.

    Responsible for:
    1. Assessing total fleet flexibility at any point in time.
    2. Creating bids for ancillary-service markets.
    3. Distributing dispatch signals across individual EVs.
    4. Tracking fleet-level performance metrics.
    """

    def __init__(self, fleet: EVFleet, scheduler: V2GScheduler | None = None) -> None:
        self.fleet = fleet
        self.scheduler = scheduler or V2GScheduler()
        self._active_bids: list[FlexibilityBid] = []
        self._dispatch_history: list[DispatchResult] = []
        self._total_revenue: float = 0.0
        self._total_dispatches: int = 0

    # -- Flexibility assessment ----------------------------------------------

    def assess_flexibility(self) -> dict[str, Any]:
        """Assess current fleet flexibility for grid services."""
        connected = [
            ev for ev in self.fleet.vehicles.values()
            if ev.connection_state != EVConnectionState.DISCONNECTED
        ]
        v2g_capable = [ev for ev in connected if ev.v2g_capable and ev.has_flexibility]

        max_charge_kw = sum(ev.max_charge_kw for ev in connected)
        max_discharge_kw = sum(ev.max_discharge_kw for ev in v2g_capable)
        available_energy_kwh = sum(ev.available_discharge_kwh for ev in v2g_capable)

        # Estimate sustainable discharge duration
        if max_discharge_kw > 0:
            sustainable_hours = available_energy_kwh / max_discharge_kw
        else:
            sustainable_hours = 0.0

        return {
            "timestamp": time.time(),
            "connected_evs": len(connected),
            "v2g_capable_evs": len(v2g_capable),
            "max_charge_kw": round(max_charge_kw, 1),
            "max_discharge_kw": round(max_discharge_kw, 1),
            "available_energy_kwh": round(available_energy_kwh, 1),
            "sustainable_discharge_hours": round(sustainable_hours, 2),
            "average_soc": round(self.fleet.average_soc, 3),
        }

    # -- Bidding -------------------------------------------------------------

    def create_flexibility_bid(
        self,
        service: GridService,
        capacity_fraction: float = 0.8,
        duration_hours: float = 1.0,
        price_per_kw: float = 0.05,
    ) -> FlexibilityBid | None:
        """Create a bid offering fleet flexibility to the grid.

        Uses a conservative fraction (default 80%) of available capacity
        to account for uncertainty.
        """
        flexibility = self.assess_flexibility()
        max_kw = flexibility["max_discharge_kw"]

        bid_capacity = max_kw * capacity_fraction
        if bid_capacity < 1.0:
            logger.info("Insufficient flexibility for %s bid", service.value)
            return None

        # Verify we have enough energy for the duration
        available_kwh = flexibility["available_energy_kwh"]
        required_kwh = bid_capacity * duration_hours
        if available_kwh < required_kwh:
            # Reduce bid capacity to match available energy
            bid_capacity = available_kwh / duration_hours
            if bid_capacity < 1.0:
                return None

        now = time.time()
        # Select EVs for this bid
        ev_ids = [
            ev.ev_id for ev in self.fleet.flexible_vehicles
        ]

        bid = FlexibilityBid(
            service=service,
            capacity_kw=bid_capacity,
            duration_hours=duration_hours,
            price_per_kw=price_per_kw,
            available_from=now,
            available_until=now + duration_hours * 3600,
            fleet_id=self.fleet.fleet_id,
            ev_ids=ev_ids,
        )
        self._active_bids.append(bid)
        logger.info(
            "Created %s bid: %.1f kW for %.1f hours (value=%.2f)",
            service.value, bid_capacity, duration_hours, bid.total_value,
        )
        return bid

    # -- Dispatch ------------------------------------------------------------

    def dispatch(self, signal: DispatchSignal) -> DispatchResult:
        """Distribute a dispatch signal across fleet EVs.

        Uses a proportional allocation strategy: each EV gets a share
        proportional to its max power capacity.
        """
        self._total_dispatches += 1
        is_discharge = signal.target_power_kw < 0
        target_abs = abs(signal.target_power_kw)

        # Select eligible EVs
        if is_discharge:
            eligible = [
                ev for ev in self.fleet.vehicles.values()
                if ev.v2g_capable
                and ev.connection_state != EVConnectionState.DISCONNECTED
                and ev.available_discharge_kwh > 0
            ]
            total_capacity = sum(ev.max_discharge_kw for ev in eligible)
        else:
            eligible = [
                ev for ev in self.fleet.vehicles.values()
                if ev.connection_state != EVConnectionState.DISCONNECTED
                and ev.current_soc < 1.0
            ]
            total_capacity = sum(ev.max_charge_kw for ev in eligible)

        if total_capacity <= 0:
            return DispatchResult(
                target_power_kw=signal.target_power_kw,
                achieved_power_kw=0.0,
                shortfall_kw=target_abs,
            )

        # Proportional allocation
        allocations: dict[str, float] = {}
        achieved = 0.0

        for ev in eligible:
            if is_discharge:
                ev_max = ev.max_discharge_kw
            else:
                ev_max = ev.max_charge_kw

            share = (ev_max / total_capacity) * target_abs
            clamped = min(share, ev_max)

            # Energy constraint check
            duration_hours = signal.duration_seconds / 3600
            if is_discharge:
                max_energy = ev.available_discharge_kwh
                energy_needed = clamped * duration_hours / ev.discharge_efficiency
                if energy_needed > max_energy:
                    clamped = max_energy * ev.discharge_efficiency / duration_hours
            else:
                headroom = (1.0 - ev.current_soc) * ev.capacity_kwh
                energy = clamped * duration_hours * ev.charge_efficiency
                if energy > headroom:
                    clamped = headroom / (duration_hours * ev.charge_efficiency)

            if clamped > 0.01:
                allocations[ev.ev_id] = -clamped if is_discharge else clamped
                achieved += clamped

        shortfall = max(0.0, target_abs - achieved)

        result = DispatchResult(
            target_power_kw=signal.target_power_kw,
            achieved_power_kw=-achieved if is_discharge else achieved,
            ev_allocations=allocations,
            shortfall_kw=shortfall,
        )
        self._dispatch_history.append(result)

        logger.info(
            "Dispatched %.1f kW across %d EVs (achieved %.1f kW, shortfall %.1f kW)",
            signal.target_power_kw,
            len(allocations),
            result.achieved_power_kw,
            shortfall,
        )
        return result

    # -- Metrics -------------------------------------------------------------

    def get_metrics(self) -> dict[str, Any]:
        """Return aggregator performance metrics."""
        return {
            "total_dispatches": self._total_dispatches,
            "total_revenue": round(self._total_revenue, 2),
            "active_bids": len(self._active_bids),
            "avg_achievement_ratio": (
                round(
                    sum(r.achievement_ratio for r in self._dispatch_history)
                    / len(self._dispatch_history),
                    3,
                )
                if self._dispatch_history
                else 0.0
            ),
            "fleet": self.fleet.to_dict(),
            "flexibility": self.assess_flexibility(),
        }

    def get_active_bids(self) -> list[FlexibilityBid]:
        """Return currently active bids (not expired)."""
        now = time.time()
        self._active_bids = [b for b in self._active_bids if b.available_until > now]
        return list(self._active_bids)
