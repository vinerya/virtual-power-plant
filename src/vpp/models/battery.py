"""
Advanced physics-based battery models for the Virtual Power Plant library.
Includes electrochemical modeling, aging effects, and thermal dynamics.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod

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
        initial_soc = config.parameters.get("initial_soc", 0.5)
        self.state = BatteryState(
            soc=initial_soc,
            soh=1.0,
            temperature=parameters.ambient_temperature,
            voltage=parameters.nominal_voltage,
            current=0.0,
            power=0.0,
            cycle_count=0.0,
            calendar_age_days=0.0
        )
        
        self._power_history: List[Tuple[datetime, float]] = []
        self._soc_history: List[Tuple[datetime, float]] = []
    
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
    
    def get_available_energy(self) -> float:
        """Get available energy for discharge in kWh."""
        available_capacity = (self.state.soc - self.parameters.min_soc) * self.parameters.nominal_capacity
        return available_capacity * self.parameters.nominal_voltage / 1000.0
    
    def get_storage_capacity(self) -> float:
        """Get remaining storage capacity for charging in kWh."""
        remaining_capacity = (self.parameters.max_soc - self.state.soc) * self.parameters.nominal_capacity
        return remaining_capacity * self.parameters.nominal_voltage / 1000.0
    
    def is_safe_to_operate(self) -> bool:
        """Check if battery is within safe operating limits."""
        return (
            self.parameters.min_temperature <= self.state.temperature <= self.parameters.max_temperature and
            self.parameters.min_soc <= self.state.soc <= self.parameters.max_soc and
            self.state.soh > 0.7  # Minimum health threshold
        )


class SimpleEquivalentCircuitModel(BatteryModel):
    """Simple equivalent circuit battery model with aging."""
    
    def __init__(self, parameters: BatteryParameters, config: ResourceConfig):
        super().__init__(parameters, config)
        self._last_soc = self.state.soc
    
    def update(self, power_setpoint: float, dt: float) -> BatteryState:
        """Update battery state using equivalent circuit model."""
        # Limit power based on current constraints
        max_charge = self.get_max_charge_power()
        max_discharge = self.get_max_discharge_power()
        actual_power = np.clip(power_setpoint, -max_discharge, max_charge)
        
        # Calculate current from power
        if actual_power >= 0:  # Charging
            efficiency = self.parameters.charge_efficiency
            current = actual_power * 1000 / (self.parameters.nominal_voltage * efficiency)
        else:  # Discharging
            efficiency = self.parameters.discharge_efficiency
            current = actual_power * 1000 / (self.parameters.nominal_voltage / efficiency)
        
        # Update SOC
        capacity_ah = self.parameters.nominal_capacity * self.state.soh
        delta_soc = (current * dt / 3600) / capacity_ah
        new_soc = np.clip(self.state.soc + delta_soc, 
                         self.parameters.min_soc, 
                         self.parameters.max_soc)
        
        # Calculate terminal voltage
        internal_resistance = self.parameters.internal_resistance * (1 + self.parameters.resistance_growth_rate * self.state.cycle_count)
        voltage_drop = current * internal_resistance
        terminal_voltage = self.parameters.nominal_voltage - voltage_drop
        
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
        cycle_stress = (current / self.parameters.max_current)**2
        cycle_aging = self.parameters.capacity_fade_rate * cycle_stress * dt / 3600
        
        # Calendar aging (temperature dependent)
        temp_factor = np.exp((self.state.temperature - 25) / 10)  # Arrhenius-like
        calendar_aging = self.parameters.calendar_fade_rate * temp_factor * dt / (24 * 3600)
        
        # Update SOH
        total_aging = cycle_aging + calendar_aging
        self.state.soh = max(0.7, self.state.soh - total_aging)
    
    def get_max_charge_power(self) -> float:
        """Get maximum charging power considering all constraints."""
        # Current limit
        max_current = self.parameters.max_current
        
        # Voltage limit
        voltage_headroom = self.parameters.max_voltage - self.state.voltage
        if voltage_headroom <= 0:
            return 0.0
        
        # SOC limit
        if self.state.soc >= self.parameters.max_soc:
            return 0.0
        
        # Temperature limit
        if self.state.temperature >= self.parameters.max_temperature:
            return 0.0
        
        # Calculate power limit
        max_power = min(
            max_current * self.parameters.nominal_voltage / 1000,  # Current limit
            voltage_headroom * max_current / 1000  # Voltage limit
        )
        
        return max_power * self.parameters.charge_efficiency
    
    def get_max_discharge_power(self) -> float:
        """Get maximum discharging power considering all constraints."""
        # Current limit
        max_current = self.parameters.max_current
        
        # Voltage limit
        voltage_margin = self.state.voltage - self.parameters.min_voltage
        if voltage_margin <= 0:
            return 0.0
        
        # SOC limit
        if self.state.soc <= self.parameters.min_soc:
            return 0.0
        
        # Temperature limit
        if self.state.temperature <= self.parameters.min_temperature:
            return 0.0
        
        # Calculate power limit
        max_power = min(
            max_current * self.parameters.nominal_voltage / 1000,  # Current limit
            voltage_margin * max_current / 1000  # Voltage limit
        )
        
        return max_power / self.parameters.discharge_efficiency


class AdvancedElectrochemicalModel(BatteryModel):
    """Advanced electrochemical battery model with detailed physics."""
    
    def __init__(self, parameters: BatteryParameters, config: ResourceConfig):
        super().__init__(parameters, config)
        
        # Additional electrochemical parameters
        self.diffusion_coefficient = config.parameters.get("diffusion_coefficient", 1e-14)  # m²/s
        self.particle_radius = config.parameters.get("particle_radius", 5e-6)  # m
        self.electrode_thickness = config.parameters.get("electrode_thickness", 100e-6)  # m
        self.porosity = config.parameters.get("porosity", 0.3)
        
        # Concentration states
        self.surface_concentration = 0.5  # Normalized
        self.bulk_concentration = 0.5  # Normalized
        
        self._last_soc = self.state.soc
    
    def update(self, power_setpoint: float, dt: float) -> BatteryState:
        """Update using advanced electrochemical model."""
        # Limit power based on current constraints
        max_charge = self.get_max_charge_power()
        max_discharge = self.get_max_discharge_power()
        actual_power = np.clip(power_setpoint, -max_discharge, max_charge)
        
        # Calculate current
        if actual_power >= 0:  # Charging
            efficiency = self.parameters.charge_efficiency
            current = actual_power * 1000 / (self.parameters.nominal_voltage * efficiency)
        else:  # Discharging
            efficiency = self.parameters.discharge_efficiency
            current = actual_power * 1000 / (self.parameters.nominal_voltage / efficiency)
        
        # Update concentration dynamics
        self._update_concentration(current, dt)
        
        # Calculate open circuit voltage from concentration
        ocv = self._calculate_ocv(self.bulk_concentration)
        
        # Calculate overpotentials
        activation_overpotential = self._calculate_activation_overpotential(current)
        concentration_overpotential = self._calculate_concentration_overpotential(current)
        ohmic_overpotential = current * self.parameters.internal_resistance
        
        # Terminal voltage
        if current >= 0:  # Charging
            terminal_voltage = ocv + activation_overpotential + concentration_overpotential + ohmic_overpotential
        else:  # Discharging
            terminal_voltage = ocv - activation_overpotential - concentration_overpotential + ohmic_overpotential
        
        # Update SOC from bulk concentration
        new_soc = self.bulk_concentration
        new_soc = np.clip(new_soc, self.parameters.min_soc, self.parameters.max_soc)
        
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
    
    def _update_concentration(self, current: float, dt: float) -> None:
        """Update lithium concentration using diffusion dynamics."""
        # Flux at particle surface
        surface_flux = current / (96485 * self.parameters.nominal_capacity * 3600)  # mol/m²/s
        
        # Diffusion time constant
        tau = self.particle_radius**2 / (15 * self.diffusion_coefficient)
        
        # Update surface concentration
        concentration_change = surface_flux * dt / (self.particle_radius / 3)
        self.surface_concentration += concentration_change
        self.surface_concentration = np.clip(self.surface_concentration, 0.01, 0.99)
        
        # Update bulk concentration with diffusion lag
        concentration_error = self.surface_concentration - self.bulk_concentration
        self.bulk_concentration += concentration_error * dt / tau
        self.bulk_concentration = np.clip(self.bulk_concentration, 0.01, 0.99)
    
    def _calculate_ocv(self, concentration: float) -> float:
        """Calculate open circuit voltage from concentration."""
        # Simplified OCV curve for lithium-ion
        x = concentration
        ocv = (
            4.2 - 1.5 * x + 
            0.5 * np.sin(2 * np.pi * x) + 
            0.1 * np.sin(4 * np.pi * x)
        )
        return np.clip(ocv, self.parameters.min_voltage, self.parameters.max_voltage)
    
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
        
        return eta
    
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
        
        return eta_conc
    
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
        current_stress = (current / self.parameters.max_current)**1.5
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
        self.parameters.internal_resistance *= (1 + resistance_growth)
    
    def get_max_charge_power(self) -> float:
        """Get maximum charging power with electrochemical constraints."""
        # Basic constraints from parent class
        basic_limit = super().get_max_charge_power()
        
        # Concentration constraint (prevent lithium plating)
        if self.surface_concentration > 0.95:
            concentration_limit = 0.0
        else:
            concentration_limit = (0.95 - self.surface_concentration) * self.parameters.max_current * self.parameters.nominal_voltage / 1000
        
        return min(basic_limit, concentration_limit)
    
    def get_max_discharge_power(self) -> float:
        """Get maximum discharging power with electrochemical constraints."""
        # Basic constraints from parent class
        basic_limit = super().get_max_discharge_power()
        
        # Concentration constraint (prevent over-discharge)
        if self.surface_concentration < 0.05:
            concentration_limit = 0.0
        else:
            concentration_limit = (self.surface_concentration - 0.05) * self.parameters.max_current * self.parameters.nominal_voltage / 1000
        
        return min(basic_limit, concentration_limit)


def create_battery_model(model_type: str, parameters: BatteryParameters, config: ResourceConfig) -> BatteryModel:
    """Factory function to create battery models."""
    models = {
        "simple": SimpleEquivalentCircuitModel,
        "advanced": AdvancedElectrochemicalModel
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown battery model type: {model_type}")
    
    return models[model_type](parameters, config)
