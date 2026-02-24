"""Inverter models — grid-forming and grid-following.

Grid-forming inverters establish voltage and frequency (droop control,
virtual synchronous machine).  Grid-following inverters track the grid
and inject/absorb power based on a setpoint.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Base inverter
# ---------------------------------------------------------------------------

@dataclass
class InverterState:
    """Instantaneous inverter operating point."""

    real_power_kw: float = 0.0
    reactive_power_kvar: float = 0.0
    voltage_pu: float = 1.0        # per-unit
    frequency_hz: float = 50.0
    current_a: float = 0.0
    power_factor: float = 1.0
    is_online: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "real_power_kw": round(self.real_power_kw, 3),
            "reactive_power_kvar": round(self.reactive_power_kvar, 3),
            "voltage_pu": round(self.voltage_pu, 4),
            "frequency_hz": round(self.frequency_hz, 3),
            "power_factor": round(self.power_factor, 3),
            "is_online": self.is_online,
        }


class InverterModel(ABC):
    """Abstract base for all inverter models."""

    def __init__(
        self,
        inverter_id: str,
        rated_power_kw: float,
        rated_voltage_v: float = 400.0,
        nominal_frequency_hz: float = 50.0,
    ) -> None:
        self.inverter_id = inverter_id
        self.rated_power_kw = rated_power_kw
        self.rated_voltage_v = rated_voltage_v
        self.nominal_frequency_hz = nominal_frequency_hz
        self.state = InverterState(frequency_hz=nominal_frequency_hz)

    @abstractmethod
    def update(self, dt_seconds: float, grid_voltage_pu: float, grid_frequency_hz: float) -> InverterState:
        """Update inverter state for one time step."""

    @abstractmethod
    def set_power_reference(self, p_kw: float, q_kvar: float = 0.0) -> None:
        """Set active/reactive power reference."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "inverter_id": self.inverter_id,
            "type": type(self).__name__,
            "rated_power_kw": self.rated_power_kw,
            "state": self.state.to_dict(),
        }


# ---------------------------------------------------------------------------
# Grid-following inverter
# ---------------------------------------------------------------------------

class GridFollowingInverter(InverterModel):
    """Grid-following (current-source) inverter.

    Tracks the grid voltage and frequency; injects power at the commanded
    setpoint.  Used for standard PV inverters and battery inverters in
    grid-connected mode.
    """

    def __init__(
        self,
        inverter_id: str,
        rated_power_kw: float,
        rated_voltage_v: float = 400.0,
        nominal_frequency_hz: float = 50.0,
        ramp_rate_kw_per_s: float = 100.0,
    ) -> None:
        super().__init__(inverter_id, rated_power_kw, rated_voltage_v, nominal_frequency_hz)
        self.ramp_rate_kw_per_s = ramp_rate_kw_per_s
        self._p_ref: float = 0.0
        self._q_ref: float = 0.0

    def set_power_reference(self, p_kw: float, q_kvar: float = 0.0) -> None:
        self._p_ref = max(-self.rated_power_kw, min(p_kw, self.rated_power_kw))
        self._q_ref = q_kvar

    def update(self, dt_seconds: float, grid_voltage_pu: float, grid_frequency_hz: float) -> InverterState:
        if not self.state.is_online:
            return self.state

        # Ramp toward reference
        max_delta = self.ramp_rate_kw_per_s * dt_seconds
        delta = self._p_ref - self.state.real_power_kw
        delta = max(-max_delta, min(delta, max_delta))
        self.state.real_power_kw += delta
        self.state.reactive_power_kvar = self._q_ref

        # Track grid
        self.state.voltage_pu = grid_voltage_pu
        self.state.frequency_hz = grid_frequency_hz

        # Power factor
        s = math.sqrt(self.state.real_power_kw**2 + self.state.reactive_power_kvar**2)
        self.state.power_factor = abs(self.state.real_power_kw / s) if s > 0.001 else 1.0

        return self.state


# ---------------------------------------------------------------------------
# Grid-forming inverter
# ---------------------------------------------------------------------------

@dataclass
class DroopSettings:
    """Droop control parameters."""

    p_droop: float = 0.05      # 5% frequency droop
    q_droop: float = 0.05      # 5% voltage droop
    deadband_hz: float = 0.02  # frequency deadband
    deadband_pu: float = 0.01  # voltage deadband


@dataclass
class VSMSettings:
    """Virtual synchronous machine parameters."""

    inertia_constant_s: float = 5.0   # H: seconds of stored energy at rated
    damping_coefficient: float = 20.0  # D: damping torque coefficient


class GridFormingInverter(InverterModel):
    """Grid-forming (voltage-source) inverter.

    Establishes voltage and frequency using droop control and optional
    virtual synchronous machine (VSM) emulation.  Used for battery
    inverters providing grid services and islanded microgrid support.
    """

    def __init__(
        self,
        inverter_id: str,
        rated_power_kw: float,
        rated_voltage_v: float = 400.0,
        nominal_frequency_hz: float = 50.0,
        droop: DroopSettings | None = None,
        vsm: VSMSettings | None = None,
    ) -> None:
        super().__init__(inverter_id, rated_power_kw, rated_voltage_v, nominal_frequency_hz)
        self.droop = droop or DroopSettings()
        self.vsm = vsm or VSMSettings()
        self._p_ref: float = 0.0
        self._q_ref: float = 0.0
        self._omega: float = 2 * math.pi * nominal_frequency_hz  # angular velocity
        self._theta: float = 0.0  # angle
        self._use_vsm: bool = False

    def enable_vsm(self, enable: bool = True) -> None:
        """Enable or disable virtual synchronous machine mode."""
        self._use_vsm = enable

    def set_power_reference(self, p_kw: float, q_kvar: float = 0.0) -> None:
        self._p_ref = max(-self.rated_power_kw, min(p_kw, self.rated_power_kw))
        self._q_ref = q_kvar

    def update(self, dt_seconds: float, grid_voltage_pu: float, grid_frequency_hz: float) -> InverterState:
        if not self.state.is_online:
            return self.state

        if self._use_vsm:
            return self._update_vsm(dt_seconds, grid_voltage_pu, grid_frequency_hz)
        return self._update_droop(dt_seconds, grid_voltage_pu, grid_frequency_hz)

    def _update_droop(
        self, dt_seconds: float, grid_voltage_pu: float, grid_frequency_hz: float,
    ) -> InverterState:
        """Droop control: adjust power based on frequency/voltage deviation."""
        f0 = self.nominal_frequency_hz
        droop = self.droop

        # Frequency droop → active power
        delta_f = grid_frequency_hz - f0
        if abs(delta_f) > droop.deadband_hz:
            # P increases when frequency drops (negative delta_f)
            p_droop_kw = -(delta_f / (droop.p_droop * f0)) * self.rated_power_kw
        else:
            p_droop_kw = 0.0

        # Voltage droop → reactive power
        delta_v = grid_voltage_pu - 1.0
        if abs(delta_v) > droop.deadband_pu:
            q_droop_kvar = -(delta_v / droop.q_droop) * self.rated_power_kw
        else:
            q_droop_kvar = 0.0

        target_p = self._p_ref + p_droop_kw
        target_q = self._q_ref + q_droop_kvar

        # Clamp to rated
        target_p = max(-self.rated_power_kw, min(target_p, self.rated_power_kw))

        self.state.real_power_kw = target_p
        self.state.reactive_power_kvar = target_q
        self.state.voltage_pu = 1.0 - droop.q_droop * (target_q / self.rated_power_kw)
        self.state.frequency_hz = f0 - droop.p_droop * f0 * (target_p / self.rated_power_kw)

        # Power factor
        s = math.sqrt(target_p**2 + target_q**2)
        self.state.power_factor = abs(target_p / s) if s > 0.001 else 1.0

        return self.state

    def _update_vsm(
        self, dt_seconds: float, grid_voltage_pu: float, grid_frequency_hz: float,
    ) -> InverterState:
        """Virtual synchronous machine: emulate inertia + damping."""
        f0 = self.nominal_frequency_hz
        omega0 = 2 * math.pi * f0
        H = self.vsm.inertia_constant_s
        D = self.vsm.damping_coefficient

        # Mechanical power reference (normalised)
        p_m = self._p_ref / self.rated_power_kw if self.rated_power_kw > 0 else 0

        # Electrical power (current output, normalised)
        p_e = self.state.real_power_kw / self.rated_power_kw if self.rated_power_kw > 0 else 0

        # Swing equation: 2H * d(omega)/dt = P_m - P_e - D*(omega - omega0)
        delta_omega = self._omega - omega0
        d_omega = (p_m - p_e - D * delta_omega / omega0) / (2 * H) * omega0
        self._omega += d_omega * dt_seconds

        # Update angle
        self._theta += self._omega * dt_seconds

        # Update state
        new_freq = self._omega / (2 * math.pi)
        self.state.frequency_hz = new_freq

        # Active power follows the swing dynamics
        self.state.real_power_kw = self._p_ref + D * (self._omega - omega0) / omega0 * self.rated_power_kw

        # Clamp
        self.state.real_power_kw = max(
            -self.rated_power_kw, min(self.state.real_power_kw, self.rated_power_kw),
        )

        # Reactive power via Q-droop on voltage
        droop = self.droop
        delta_v = grid_voltage_pu - 1.0
        self.state.reactive_power_kvar = self._q_ref - (delta_v / droop.q_droop) * self.rated_power_kw
        self.state.voltage_pu = grid_voltage_pu

        s = math.sqrt(self.state.real_power_kw**2 + self.state.reactive_power_kvar**2)
        self.state.power_factor = abs(self.state.real_power_kw / s) if s > 0.001 else 1.0

        return self.state

    # -- Virtual inertia response --------------------------------------------

    def inertia_response(self, frequency_hz: float) -> float:
        """Calculate the virtual inertia power response (kW) to a frequency event."""
        f0 = self.nominal_frequency_hz
        rocof = (frequency_hz - self.state.frequency_hz)  # simplified
        if abs(rocof) < 0.001:
            return 0.0

        # P_inertia = -2 * H * S_rated * (df/dt) / f0
        # Simplified for discrete time step
        return -2 * self.vsm.inertia_constant_s * self.rated_power_kw * rocof / f0

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base["droop"] = {
            "p_droop": self.droop.p_droop,
            "q_droop": self.droop.q_droop,
        }
        base["vsm_enabled"] = self._use_vsm
        if self._use_vsm:
            base["vsm"] = {
                "H": self.vsm.inertia_constant_s,
                "D": self.vsm.damping_coefficient,
            }
        return base
