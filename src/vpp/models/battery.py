"""
Advanced physics-based battery models for the Virtual Power Plant library.
Includes electrochemical modeling, aging effects, and thermal dynamics.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ..config import ResourceConfig


@dataclass
class BatteryState:
    """Current state of the battery."""

    soc: float  # State of charge (0-1)
    soh: float  # State of health (0-1)
    temperature: float  # Temperature in Celsius
    voltage: float  # Terminal voltage in V
    current: float  # Current in A (positive = charging)
    power: float  # Power in kW (positive = charging)
    cycle_count: float  # Equivalent full cycles
    calendar_age_days: float  # Calendar age in days
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class BatteryParameters:
    """Physical and electrical parameters of the battery."""

    # Basic specifications
    nominal_capacity: float  # Ah
    nominal_voltage: float  # V
    max_voltage: float  # V
    min_voltage: float  # V
    max_current: float  # A

    # Electrochemical parameters
    internal_resistance: float  # Ohm
    capacity_fade_rate: float = 0.0002  # per cycle
    resistance_growth_rate: float = 0.0001  # per cycle
    calendar_fade_rate: float = 0.00005  # per day

    # Thermal parameters
    thermal_mass: float = 1000.0  # J/K
    thermal_resistance: float = 0.1  # K/W
    ambient_temperature: float = 25.0  # C

    # Efficiency parameters
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95

    # Safety limits
    max_temperature: float = 60.0  # C
    min_temperature: float = -20.0  # C
    max_soc: float = 0.95
    min_soc: float = 0.05


class BatteryModel(ABC):
    """Abstract base class for battery models."""

    def __init__(self, parameters: BatteryParameters, config: ResourceConfig):
        self.parameters = parameters
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

        # Initialize state
        initial_soc = float(config.parameters.get("initial_soc", 0.5))
        self.state = BatteryState(
            soc=initial_soc,
            soh=1.0,
            temperature=parameters.ambient_temperature,
            voltage=parameters.nominal_voltage,
            current=0.0,
            power=0.0,
            cycle_count=0.0,
            calendar_age_days=0.0,
        )

        self._power_history: list[tuple[datetime, float]] = []
        self._soc_history: list[tuple[datetime, float]] = []

    @abstractmethod
    def update(self, power_setpoint: float, dt: float) -> BatteryState:
        """Update battery state given power setpoint and time step."""
        pass

    @abstractmethod
    def get_max_charge_power(self) -> float:
        """Get maximum charging power at current state."""
        pass

    @abstractmethod
    def get_max_discharge_power(self) -> float:
        """Get maximum discharging power at current state."""
        pass

    def _coulomb_count(self, power_setpoint: float, dt: float) -> tuple[float, float, float]:
        """Apply ``power_setpoint`` (kW, + = charging) for ``dt`` seconds.

        Returns ``(actual_power_kw, cell_current_a, new_soc)``. The power is
        limited by ``get_max_charge_power`` / ``get_max_discharge_power`` and
        so that SoC stays within ``[min_soc, max_soc]`` during the step; SoC
        then moves by exactly the charge that reached the cells.
        """
        p = self.parameters
        soc = self.state.soc
        actual_power = float(
            np.clip(power_setpoint, -self.get_max_discharge_power(), self.get_max_charge_power())
        )
        current = _cell_current(p, actual_power)

        capacity_ah = p.nominal_capacity * self.state.soh
        if dt > 0 and capacity_ah > 0:
            # Largest charge/discharge current that does not cross a SoC limit
            # this step (never forces a move if SoC starts outside the limits).
            max_in = max(0.0, (p.max_soc - soc) * capacity_ah * 3600.0 / dt)
            max_out = max(0.0, (soc - p.min_soc) * capacity_ah * 3600.0 / dt)
            limited = min(max(current, -max_out), max_in)
            if limited != current:
                current = limited
                actual_power = _terminal_power(p, current)
            new_soc = soc + current * dt / 3600.0 / capacity_ah
            # Guard against floating-point drift past the limits.
            new_soc = float(min(max(new_soc, min(p.min_soc, soc)), max(p.max_soc, soc)))
        else:
            new_soc = soc
        return actual_power, current, new_soc

    def get_available_energy(self) -> float:
        """Get available energy for discharge in kWh."""
        available_capacity = (
            self.state.soc - self.parameters.min_soc
        ) * self.parameters.nominal_capacity
        return available_capacity * self.parameters.nominal_voltage / 1000.0

    def get_storage_capacity(self) -> float:
        """Get remaining storage capacity for charging in kWh."""
        remaining_capacity = (
            self.parameters.max_soc - self.state.soc
        ) * self.parameters.nominal_capacity
        return remaining_capacity * self.parameters.nominal_voltage / 1000.0

    def is_safe_to_operate(self) -> bool:
        """Check if battery is within safe operating limits."""
        return (
            self.parameters.min_temperature
            <= self.state.temperature
            <= self.parameters.max_temperature
            and self.parameters.min_soc <= self.state.soc <= self.parameters.max_soc
            and self.state.soh > 0.7  # Minimum health threshold
        )


def _cell_current(parameters: BatteryParameters, power_kw: float) -> float:
    """Current through the cells (A, + = charging) for a terminal power (kW).

    Conversion losses: charging stores ``charge_efficiency`` of the power
    delivered; discharging draws ``power / discharge_efficiency`` from the
    cells. Current is referred to the nominal voltage, so SoC follows
    Coulomb counting of the delivered energy.
    """
    if power_kw >= 0:
        return power_kw * 1000.0 * parameters.charge_efficiency / parameters.nominal_voltage
    return power_kw * 1000.0 / (parameters.discharge_efficiency * parameters.nominal_voltage)


def _terminal_power(parameters: BatteryParameters, current: float) -> float:
    """Inverse of :func:`_cell_current`: terminal power (kW) for a cell current (A)."""
    if current >= 0:
        return current * parameters.nominal_voltage / (parameters.charge_efficiency * 1000.0)
    return current * parameters.nominal_voltage * parameters.discharge_efficiency / 1000.0


def _ecm_max_charge_power(
    parameters: BatteryParameters, state: BatteryState, ocv: float, resistance: float
) -> float:
    """Max charging power (kW) from current/voltage/SOC/temperature limits.

    The voltage limit is the current at which ``ocv + I * resistance``
    reaches ``max_voltage``.
    """
    # SOC limit
    if state.soc >= parameters.max_soc:
        return 0.0

    # Temperature limit
    if state.temperature >= parameters.max_temperature:
        return 0.0

    # Voltage limit
    voltage_headroom = parameters.max_voltage - ocv
    if voltage_headroom <= 0:
        return 0.0
    max_current = parameters.max_current
    if resistance > 0:
        max_current = min(max_current, voltage_headroom / resistance)

    return _terminal_power(parameters, max_current)


def _ecm_max_discharge_power(
    parameters: BatteryParameters, state: BatteryState, ocv: float, resistance: float
) -> float:
    """Max discharging power (kW) from current/voltage/SOC/temperature limits.

    The voltage limit is the current at which ``ocv - I * resistance``
    falls to ``min_voltage``.
    """
    # SOC limit
    if state.soc <= parameters.min_soc:
        return 0.0

    # Temperature limit
    if state.temperature <= parameters.min_temperature:
        return 0.0

    # Voltage limit
    voltage_margin = ocv - parameters.min_voltage
    if voltage_margin <= 0:
        return 0.0
    max_current = parameters.max_current
    if resistance > 0:
        max_current = min(max_current, voltage_margin / resistance)

    return -_terminal_power(parameters, -max_current)


class SimpleEquivalentCircuitModel(BatteryModel):
    """Simple equivalent circuit battery model with aging."""

    def __init__(self, parameters: BatteryParameters, config: ResourceConfig):
        super().__init__(parameters, config)
        self._last_soc = self.state.soc

    def update(self, power_setpoint: float, dt: float) -> BatteryState:
        """Update battery state using equivalent circuit model."""
        actual_power, current, new_soc = self._coulomb_count(power_setpoint, dt)

        # Terminal voltage: open-circuit (taken as nominal) plus the ohmic
        # drop, which raises the voltage while charging (current > 0).
        internal_resistance = self._internal_resistance()
        terminal_voltage = self.parameters.nominal_voltage + current * internal_resistance

        # Update thermal state
        power_loss = current**2 * internal_resistance / 1000  # kW
        temperature_rise = power_loss * self.parameters.thermal_resistance
        new_temperature = self.parameters.ambient_temperature + temperature_rise

        # Update aging
        self._update_aging(abs(current), dt)

        # Update cycle count
        soc_change = abs(new_soc - self._last_soc)
        self.state.cycle_count += soc_change / 2.0  # Half cycle per SOC swing
        self._last_soc = new_soc

        # Update calendar age
        self.state.calendar_age_days += dt / (24 * 3600)

        # Update state
        self.state.soc = new_soc
        self.state.temperature = new_temperature
        self.state.voltage = terminal_voltage
        self.state.current = current
        self.state.power = actual_power
        self.state.last_update = datetime.now()

        # Record history
        self._power_history.append((self.state.last_update, actual_power))
        self._soc_history.append((self.state.last_update, new_soc))

        # Limit history size
        max_history = 1000
        if len(self._power_history) > max_history:
            self._power_history = self._power_history[-max_history:]
            self._soc_history = self._soc_history[-max_history:]

        return self.state

    def _update_aging(self, current: float, dt: float) -> None:
        """Update state of health based on cycling and calendar aging."""
        # Cycle aging
        cycle_stress = (current / self.parameters.max_current) ** 2
        cycle_aging = self.parameters.capacity_fade_rate * cycle_stress * dt / 3600

        # Calendar aging (temperature dependent)
        temp_factor = np.exp((self.state.temperature - 25) / 10)  # Arrhenius-like
        calendar_aging = self.parameters.calendar_fade_rate * temp_factor * dt / (24 * 3600)

        # Update SOH
        total_aging = cycle_aging + calendar_aging
        self.state.soh = max(0.7, self.state.soh - total_aging)

    def _internal_resistance(self) -> float:
        return self.parameters.internal_resistance * (
            1 + self.parameters.resistance_growth_rate * self.state.cycle_count
        )

    def get_max_charge_power(self) -> float:
        """Get maximum charging power considering all constraints."""
        return _ecm_max_charge_power(
            self.parameters,
            self.state,
            self.parameters.nominal_voltage,
            self._internal_resistance(),
        )

    def get_max_discharge_power(self) -> float:
        """Get maximum discharging power considering all constraints."""
        return _ecm_max_discharge_power(
            self.parameters,
            self.state,
            self.parameters.nominal_voltage,
            self._internal_resistance(),
        )


class AdvancedElectrochemicalModel(BatteryModel):
    """Advanced electrochemical battery model with detailed physics."""

    def __init__(self, parameters: BatteryParameters, config: ResourceConfig):
        super().__init__(parameters, config)

        # Additional electrochemical parameters.  Coerced with float():
        # PyYAML follows YAML 1.1, where ``1e-14`` (no dot) is a *string*,
        # so values from a YAML config file can arrive as str.
        params = config.parameters
        self.diffusion_coefficient = float(params.get("diffusion_coefficient", 1e-14))  # m²/s
        self.particle_radius = float(params.get("particle_radius", 5e-6))  # m
        self.electrode_thickness = float(params.get("electrode_thickness", 100e-6))  # m
        self.porosity = float(params.get("porosity", 0.3))

        # Normalised lithium concentration in the active particles: the
        # particle average ("bulk") is the SoC; the surface leads it while
        # current flows.  Start at rest at the initial SoC.
        self.bulk_concentration = self.state.soc
        self.surface_concentration = self.state.soc

        self._last_soc = self.state.soc

    def update(self, power_setpoint: float, dt: float) -> BatteryState:
        """Update using advanced electrochemical model."""
        # SoC by Coulomb counting of the charge that reaches the cells.
        actual_power, current, new_soc = self._coulomb_count(power_setpoint, dt)

        # Update concentration dynamics
        self._update_concentration(new_soc, dt)

        # Calculate open circuit voltage from concentration
        ocv = self._calculate_ocv(self.bulk_concentration)

        # Calculate overpotentials.  Each carries the sign of the current
        # (asinh is odd; the surface leads the bulk in the direction of the
        # current), so they add to the OCV when charging and subtract when
        # discharging.
        activation_overpotential = self._calculate_activation_overpotential(current)
        concentration_overpotential = self._calculate_concentration_overpotential(current)
        ohmic_overpotential = current * self.parameters.internal_resistance

        terminal_voltage = (
            ocv + activation_overpotential + concentration_overpotential + ohmic_overpotential
        )

        # Update thermal state with more detailed heat generation
        reversible_heat = current * self._calculate_entropy_coefficient() * self.state.temperature
        irreversible_heat = current**2 * self.parameters.internal_resistance
        total_heat = (reversible_heat + irreversible_heat) / 1000  # kW

        temperature_rise = total_heat * self.parameters.thermal_resistance
        new_temperature = self.parameters.ambient_temperature + temperature_rise

        # Update aging
        self._update_aging(abs(current), dt)

        # Update cycle count
        soc_change = abs(new_soc - self._last_soc)
        self.state.cycle_count += soc_change / 2.0
        self._last_soc = new_soc

        # Update calendar age
        self.state.calendar_age_days += dt / (24 * 3600)

        # Update state
        self.state.soc = new_soc
        self.state.temperature = new_temperature
        self.state.voltage = terminal_voltage
        self.state.current = current
        self.state.power = actual_power
        self.state.last_update = datetime.now()

        # Record history
        self._power_history.append((self.state.last_update, actual_power))
        self._soc_history.append((self.state.last_update, new_soc))

        return self.state

    def _diffusion_time_constant(self) -> float:
        """Solid-phase diffusion time constant R^2/(15 D) in seconds."""
        return self.particle_radius**2 / (15 * self.diffusion_coefficient)

    def _update_concentration(self, new_soc: float, dt: float) -> None:
        """Update lithium concentrations after a step that moved SoC to ``new_soc``.

        The particle-average (bulk) concentration *is* the SoC: it changes by
        exactly the charge moved.  (The old flux formula here was
        dimensionally wrong, ran several times faster than Coulomb counting,
        and kept the bulk drifting towards the surface after the current
        stopped or reversed.)  The surface concentration uses the
        polynomial-profile approximation of Fickian diffusion in a sphere,
        ``c_surf - c_avg = tau * d(c_avg)/dt`` with ``tau = R^2/(15 D)``.
        """
        rate = (new_soc - self.bulk_concentration) / dt if dt > 0 else 0.0
        self.bulk_concentration = new_soc
        self.surface_concentration = float(
            np.clip(new_soc + self._diffusion_time_constant() * rate, 0.01, 0.99)
        )

    def _calculate_ocv(self, concentration: float) -> float:
        """Pack open-circuit voltage (V) at a normalised concentration (= SoC).

        Rises with SoC, +/-4 % of nominal voltage across the SoC range.  (The
        previous curve was a single-cell voltage of ~3-4 V, clipped to the
        pack's ``min_voltage`` and so pinned there, which blocked discharge.)
        """
        ocv = self.parameters.nominal_voltage * (1.0 + 0.08 * (concentration - 0.5))
        return float(np.clip(ocv, self.parameters.min_voltage, self.parameters.max_voltage))

    def _calculate_activation_overpotential(self, current: float) -> float:
        """Calculate activation overpotential using Butler-Volmer kinetics."""
        if abs(current) < 1e-6:
            return 0.0

        # Exchange current density (A/m²)
        i0 = 1.0

        # Tafel slope
        alpha = 0.5
        F = 96485  # C/mol
        R = 8.314  # J/mol/K
        T = self.state.temperature + 273.15  # K

        # Butler-Volmer equation (linearized for small overpotentials)
        eta = (R * T / (alpha * F)) * np.asinh(current / (2 * i0))

        return float(eta)

    def _calculate_concentration_overpotential(self, current: float) -> float:
        """Calculate concentration overpotential."""
        if abs(current) < 1e-6:
            return 0.0

        # Simplified concentration overpotential
        R = 8.314  # J/mol/K
        T = self.state.temperature + 273.15  # K
        F = 96485  # C/mol

        # Concentration ratio
        c_ratio = self.surface_concentration / self.bulk_concentration
        eta_conc = (R * T / F) * np.log(c_ratio)

        return float(eta_conc)

    def _calculate_entropy_coefficient(self) -> float:
        """Calculate entropy coefficient for reversible heat calculation."""
        # Simplified entropy coefficient (V/K)
        return -0.0001 * (1 - 2 * self.state.soc)

    def _update_aging(self, current: float, dt: float) -> None:
        """Update aging with more detailed mechanisms."""
        # SEI layer growth (calendar aging)
        temp_factor = np.exp((self.state.temperature - 25) / 10)
        sei_growth = self.parameters.calendar_fade_rate * temp_factor * dt / (24 * 3600)

        # Active material loss (cycle aging)
        current_stress = (current / self.parameters.max_current) ** 1.5
        am_loss = self.parameters.capacity_fade_rate * current_stress * dt / 3600

        # Lithium plating (high current charging)
        if current > 0 and current > 0.8 * self.parameters.max_current:
            plating_factor = ((current / self.parameters.max_current) - 0.8) / 0.2
            li_plating = 0.001 * plating_factor * dt / 3600
        else:
            li_plating = 0.0

        # Total aging
        total_aging = sei_growth + am_loss + li_plating
        self.state.soh = max(0.7, self.state.soh - total_aging)

        # Update internal resistance
        resistance_growth = self.parameters.resistance_growth_rate * total_aging
        self.parameters.internal_resistance *= 1 + resistance_growth

    def get_max_charge_power(self) -> float:
        """Get maximum charging power with electrochemical constraints."""
        # Basic current/voltage/SOC/temperature constraints.  (``BatteryModel``'s
        # version is abstract and returns None, so it cannot be used via super().)
        basic_limit = _ecm_max_charge_power(
            self.parameters,
            self.state,
            self._calculate_ocv(self.bulk_concentration),
            self.parameters.internal_resistance,
        )

        # Concentration constraint (prevent lithium plating): taper to zero
        # as the surface concentration approaches 0.95.
        taper = float(np.clip((0.95 - self.surface_concentration) / 0.1, 0.0, 1.0))
        return basic_limit * taper

    def get_max_discharge_power(self) -> float:
        """Get maximum discharging power with electrochemical constraints."""
        # Basic constraints (see get_max_charge_power).
        basic_limit = _ecm_max_discharge_power(
            self.parameters,
            self.state,
            self._calculate_ocv(self.bulk_concentration),
            self.parameters.internal_resistance,
        )

        # Concentration constraint (prevent over-discharge): taper to zero
        # as the surface concentration approaches 0.05.
        taper = float(np.clip((self.surface_concentration - 0.05) / 0.1, 0.0, 1.0))
        return basic_limit * taper


def create_battery_model(
    model_type: str, parameters: BatteryParameters, config: ResourceConfig
) -> BatteryModel:
    """Factory function to create battery models."""
    models = {"simple": SimpleEquivalentCircuitModel, "advanced": AdvancedElectrochemicalModel}

    if model_type not in models:
        raise ValueError(f"Unknown battery model type: {model_type}")

    return models[model_type](parameters, config)
