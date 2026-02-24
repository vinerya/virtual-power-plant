"""Microgrid controller — islanding, reconnection, and load management.

Manages transitions between grid-connected and islanded operation,
coordinates grid-forming inverters, and implements load shedding when
generation is insufficient.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from vpp.grid.inverter import GridFormingInverter, GridFollowingInverter, InverterModel

logger = logging.getLogger(__name__)


class MicrogridState(str, Enum):
    GRID_CONNECTED = "grid_connected"
    ISLANDING = "islanding"         # transition in progress
    ISLANDED = "islanded"
    RECONNECTING = "reconnecting"   # resync in progress
    FAULT = "fault"


@dataclass
class LoadPriority:
    """A load with a shedding priority."""

    load_id: str
    power_kw: float
    priority: int = 5   # 1 = critical (never shed), 10 = lowest priority
    is_shed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "load_id": self.load_id,
            "power_kw": round(self.power_kw, 2),
            "priority": self.priority,
            "is_shed": self.is_shed,
        }


@dataclass
class MicrogridMetrics:
    """Runtime metrics for the microgrid controller."""

    island_events: int = 0
    reconnection_events: int = 0
    load_shed_events: int = 0
    total_island_duration_s: float = 0.0
    last_island_at: float | None = None
    last_reconnect_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "island_events": self.island_events,
            "reconnection_events": self.reconnection_events,
            "load_shed_events": self.load_shed_events,
            "total_island_duration_s": round(self.total_island_duration_s, 1),
        }


class MicrogridController:
    """Microgrid controller managing island/reconnect transitions.

    Coordinates:
    - Island detection (frequency/voltage deviation)
    - Transition to island mode (switch grid-following → grid-forming)
    - Load shedding when generation < demand
    - Resynchronisation and reconnection to the main grid
    """

    def __init__(
        self,
        nominal_frequency_hz: float = 50.0,
        nominal_voltage_pu: float = 1.0,
        frequency_threshold_hz: float = 0.5,
        voltage_threshold_pu: float = 0.1,
        resync_frequency_tolerance_hz: float = 0.05,
        resync_voltage_tolerance_pu: float = 0.02,
        resync_phase_tolerance_deg: float = 5.0,
    ) -> None:
        self.nominal_frequency_hz = nominal_frequency_hz
        self.nominal_voltage_pu = nominal_voltage_pu
        self.frequency_threshold_hz = frequency_threshold_hz
        self.voltage_threshold_pu = voltage_threshold_pu
        self.resync_frequency_tolerance_hz = resync_frequency_tolerance_hz
        self.resync_voltage_tolerance_pu = resync_voltage_tolerance_pu
        self.resync_phase_tolerance_deg = resync_phase_tolerance_deg

        self._state = MicrogridState.GRID_CONNECTED
        self._inverters: dict[str, InverterModel] = {}
        self._loads: dict[str, LoadPriority] = {}
        self._metrics = MicrogridMetrics()
        self._island_start: float | None = None

    # -- Properties ----------------------------------------------------------

    @property
    def state(self) -> MicrogridState:
        return self._state

    @property
    def metrics(self) -> MicrogridMetrics:
        return self._metrics

    @property
    def is_islanded(self) -> bool:
        return self._state in (MicrogridState.ISLANDED, MicrogridState.ISLANDING)

    # -- Registration --------------------------------------------------------

    def add_inverter(self, inverter: InverterModel) -> None:
        self._inverters[inverter.inverter_id] = inverter

    def remove_inverter(self, inverter_id: str) -> None:
        self._inverters.pop(inverter_id, None)

    def add_load(self, load: LoadPriority) -> None:
        self._loads[load.load_id] = load

    def remove_load(self, load_id: str) -> None:
        self._loads.pop(load_id, None)

    # -- Island detection ----------------------------------------------------

    def detect_island(self, grid_frequency_hz: float, grid_voltage_pu: float) -> bool:
        """Check if grid loss is detected based on frequency/voltage deviation."""
        freq_deviation = abs(grid_frequency_hz - self.nominal_frequency_hz)
        voltage_deviation = abs(grid_voltage_pu - self.nominal_voltage_pu)

        return (
            freq_deviation > self.frequency_threshold_hz
            or voltage_deviation > self.voltage_threshold_pu
        )

    # -- Transition to island ------------------------------------------------

    def initiate_islanding(self) -> bool:
        """Transition to island mode.

        1. Switch grid-forming inverters to voltage-source mode.
        2. Check generation vs demand.
        3. Shed loads if necessary.
        """
        if self._state != MicrogridState.GRID_CONNECTED:
            logger.warning("Cannot island from state %s", self._state.value)
            return False

        self._state = MicrogridState.ISLANDING
        logger.info("Initiating island transition")

        # Enable grid-forming mode on all GFM inverters
        gfm_count = 0
        for inv in self._inverters.values():
            if isinstance(inv, GridFormingInverter):
                inv.state.is_online = True
                gfm_count += 1

        if gfm_count == 0:
            logger.error("No grid-forming inverters available — cannot island")
            self._state = MicrogridState.FAULT
            return False

        # Check power balance and shed if needed
        self._balance_power()

        self._state = MicrogridState.ISLANDED
        self._island_start = time.time()
        self._metrics.island_events += 1
        self._metrics.last_island_at = self._island_start
        logger.info("Island mode active (%d grid-forming inverters)", gfm_count)
        return True

    # -- Reconnection --------------------------------------------------------

    def check_resync_ready(
        self,
        grid_frequency_hz: float,
        grid_voltage_pu: float,
        phase_angle_deg: float = 0.0,
    ) -> bool:
        """Check if resynchronisation conditions are met."""
        microgrid_freq = self._get_microgrid_frequency()
        microgrid_voltage = self._get_microgrid_voltage()

        freq_ok = abs(microgrid_freq - grid_frequency_hz) < self.resync_frequency_tolerance_hz
        voltage_ok = abs(microgrid_voltage - grid_voltage_pu) < self.resync_voltage_tolerance_pu
        phase_ok = abs(phase_angle_deg) < self.resync_phase_tolerance_deg

        return freq_ok and voltage_ok and phase_ok

    def initiate_reconnection(
        self,
        grid_frequency_hz: float,
        grid_voltage_pu: float,
        phase_angle_deg: float = 0.0,
    ) -> bool:
        """Reconnect to the main grid.

        1. Verify sync conditions.
        2. Close interconnect breaker.
        3. Transfer grid-forming inverters back to grid-following behaviour.
        4. Restore shed loads.
        """
        if self._state != MicrogridState.ISLANDED:
            logger.warning("Cannot reconnect from state %s", self._state.value)
            return False

        if not self.check_resync_ready(grid_frequency_hz, grid_voltage_pu, phase_angle_deg):
            logger.warning("Resync conditions not met")
            return False

        self._state = MicrogridState.RECONNECTING
        logger.info("Initiating reconnection to grid")

        # Restore all shed loads
        for load in self._loads.values():
            load.is_shed = False

        self._state = MicrogridState.GRID_CONNECTED
        self._metrics.reconnection_events += 1
        self._metrics.last_reconnect_at = time.time()

        if self._island_start is not None:
            self._metrics.total_island_duration_s += time.time() - self._island_start
            self._island_start = None

        logger.info("Reconnected to grid")
        return True

    # -- Load shedding -------------------------------------------------------

    def _balance_power(self) -> None:
        """Check generation vs demand and shed loads if necessary."""
        generation = sum(
            inv.state.real_power_kw
            for inv in self._inverters.values()
            if inv.state.is_online and inv.state.real_power_kw > 0
        )
        rated_capacity = sum(
            inv.rated_power_kw
            for inv in self._inverters.values()
            if inv.state.is_online
        )

        # Use rated capacity as available generation estimate
        available = max(generation, rated_capacity * 0.8)

        demand = sum(
            load.power_kw
            for load in self._loads.values()
            if not load.is_shed
        )

        if demand <= available:
            return

        # Need to shed: shed lowest priority first
        sorted_loads = sorted(
            [l for l in self._loads.values() if not l.is_shed and l.priority > 1],
            key=lambda l: -l.priority,  # highest number = lowest priority
        )

        excess = demand - available
        for load in sorted_loads:
            if excess <= 0:
                break
            load.is_shed = True
            excess -= load.power_kw
            self._metrics.load_shed_events += 1
            logger.info("Shed load %s (%.1f kW, priority %d)", load.load_id, load.power_kw, load.priority)

    def shed_loads_below_priority(self, min_priority: int) -> int:
        """Manually shed all loads with priority >= min_priority."""
        count = 0
        for load in self._loads.values():
            if load.priority >= min_priority and not load.is_shed:
                load.is_shed = True
                count += 1
                self._metrics.load_shed_events += 1
        return count

    def restore_loads_above_priority(self, max_priority: int) -> int:
        """Restore loads with priority <= max_priority."""
        count = 0
        for load in self._loads.values():
            if load.priority <= max_priority and load.is_shed:
                load.is_shed = False
                count += 1
        return count

    # -- Queries -------------------------------------------------------------

    def _get_microgrid_frequency(self) -> float:
        """Average frequency from grid-forming inverters."""
        gfm = [
            inv for inv in self._inverters.values()
            if isinstance(inv, GridFormingInverter) and inv.state.is_online
        ]
        if not gfm:
            return self.nominal_frequency_hz
        return sum(inv.state.frequency_hz for inv in gfm) / len(gfm)

    def _get_microgrid_voltage(self) -> float:
        """Average voltage from grid-forming inverters."""
        gfm = [
            inv for inv in self._inverters.values()
            if isinstance(inv, GridFormingInverter) and inv.state.is_online
        ]
        if not gfm:
            return self.nominal_voltage_pu
        return sum(inv.state.voltage_pu for inv in gfm) / len(gfm)

    def get_total_generation(self) -> float:
        return sum(
            max(0, inv.state.real_power_kw)
            for inv in self._inverters.values()
            if inv.state.is_online
        )

    def get_total_demand(self) -> float:
        return sum(
            load.power_kw for load in self._loads.values() if not load.is_shed
        )

    def get_shed_load_kw(self) -> float:
        return sum(
            load.power_kw for load in self._loads.values() if load.is_shed
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "is_islanded": self.is_islanded,
            "inverter_count": len(self._inverters),
            "load_count": len(self._loads),
            "total_generation_kw": round(self.get_total_generation(), 1),
            "total_demand_kw": round(self.get_total_demand(), 1),
            "shed_load_kw": round(self.get_shed_load_kw(), 1),
            "microgrid_frequency_hz": round(self._get_microgrid_frequency(), 3),
            "microgrid_voltage_pu": round(self._get_microgrid_voltage(), 4),
            "metrics": self._metrics.to_dict(),
        }
