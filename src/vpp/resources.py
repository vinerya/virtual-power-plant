"""Energy resource implementations for the Virtual Power Plant."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List

from .exceptions import ResourceError

class EnergyResource(ABC):
    """Abstract base class for all energy resources."""
    
    def __init__(
        self,
        rated_power: float,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        power_curve: Optional[Callable[[float], float]] = None
    ):
        """Initialize resource with rated power capacity and optional parameters."""
        if rated_power <= 0:
            raise ResourceError("Rated power must be positive")
        
        self.name = name or f"{self.__class__.__name__}_{id(self)}"
        self.rated_power = rated_power
        self._current_power = 0.0
        self._efficiency = 0.95  # Default efficiency
        self._online = True
        self._last_update = datetime.utcnow()
        self._metadata = metadata or {}
        self._power_curve = power_curve
        self._state_history: List[Dict[str, Any]] = []
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get current resource metrics."""
        return {
            "name": self.name,
            "rated_power": self.rated_power,
            "current_power": self._current_power,
            "efficiency": self._efficiency,
            "online": self._online,
            "last_update": self._last_update.isoformat(),
            **self._metadata
        }
    
    @property
    def is_online(self) -> bool:
        """Check if resource is online."""
        return self._online
    
    def set_power(self, power: float) -> None:
        """Set current power output."""
        if not self._online:
            raise ResourceError("Cannot set power: resource is offline")
        
        if power < 0 or power > self.rated_power:
            raise ResourceError(
                f"Power must be between 0 and {self.rated_power} kW"
            )
        
        # Apply custom power curve if available
        if self._power_curve:
            power = self._power_curve(power)
        
        self._current_power = power
        self._last_update = datetime.utcnow()
        
        # Record state for analysis
        self._record_state()
    
    def _record_state(self) -> None:
        """Record current state for analysis."""
        state = {
            "timestamp": self._last_update,
            "power": self._current_power,
            "efficiency": self._efficiency,
            **self._metadata
        }
        self._state_history.append(state)
    
    def get_state_history(self) -> List[Dict[str, Any]]:
        """Get resource state history for analysis."""
        return self._state_history
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata for research/analysis."""
        self._metadata[key] = value
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get resource metadata."""
        return self._metadata
    
    def set_power_curve(self, curve: Callable[[float], float]) -> None:
        """Set custom power curve function."""
        self._power_curve = curve

class Battery(EnergyResource):
    """Battery energy storage system."""
    
    def __init__(
        self,
        capacity: float,
        current_charge: float,
        max_power: float,
        nominal_voltage: float
    ):
        """Initialize battery with capacity and charge state."""
        super().__init__(rated_power=max_power)
        
        if capacity <= 0:
            raise ResourceError("Capacity must be positive")
        if current_charge < 0 or current_charge > capacity:
            raise ResourceError("Current charge must be between 0 and capacity")
        if nominal_voltage <= 0:
            raise ResourceError("Nominal voltage must be positive")
        
        self.capacity = capacity  # kWh
        self.current_charge = current_charge  # kWh
        self.nominal_voltage = nominal_voltage  # V
        self.charge_efficiency = 0.95
        self.discharge_efficiency = 0.95
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get battery-specific metrics."""
        metrics = super().get_metrics()
        metrics.update({
            "capacity": self.capacity,
            "current_charge": self.current_charge,
            "state_of_charge": self.current_charge / self.capacity * 100,
            "nominal_voltage": self.nominal_voltage,
            "charge_efficiency": self.charge_efficiency,
            "discharge_efficiency": self.discharge_efficiency
        })
        return metrics
    
    def charge(self, power: float, duration: float) -> None:
        """Charge battery with given power for duration."""
        if not self.is_online:
            raise ResourceError("Cannot charge: battery is offline")
        
        if power < 0 or power > self.rated_power:
            raise ResourceError(
                f"Charging power must be between 0 and {self.rated_power} kW"
            )
        
        energy = power * duration * self.charge_efficiency
        new_charge = self.current_charge + energy
        
        if new_charge > self.capacity:
            raise ResourceError("Cannot charge: would exceed capacity")
        
        self.current_charge = new_charge
        self._current_power = power
        self._last_update = datetime.utcnow()
    
    def discharge(self, power: float, duration: float) -> None:
        """Discharge battery with given power for duration."""
        if not self.is_online:
            raise ResourceError("Cannot discharge: battery is offline")
        
        if power < 0 or power > self.rated_power:
            raise ResourceError(
                f"Discharge power must be between 0 and {self.rated_power} kW"
            )
        
        energy = power * duration / self.discharge_efficiency
        new_charge = self.current_charge - energy
        
        if new_charge < 0:
            raise ResourceError("Cannot discharge: insufficient charge")
        
        self.current_charge = new_charge
        self._current_power = -power  # Negative for discharge
        self._last_update = datetime.utcnow()

class Solar(EnergyResource):
    """Solar power generation system."""
    
    def __init__(
        self,
        peak_power: float,
        panel_area: float,
        efficiency: float
    ):
        """Initialize solar system with panel specifications."""
        super().__init__(rated_power=peak_power)
        
        if panel_area <= 0:
            raise ResourceError("Panel area must be positive")
        if efficiency <= 0 or efficiency > 1:
            raise ResourceError("Efficiency must be between 0 and 1")
        
        self.panel_area = panel_area  # m²
        self.panel_efficiency = efficiency
        self._irradiance = 0.0  # W/m²
        self._temperature = 25.0  # °C
        self._temp_coefficient = -0.004  # Typical value for Si cells
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get solar-specific metrics."""
        metrics = super().get_metrics()
        metrics.update({
            "panel_area": self.panel_area,
            "panel_efficiency": self.panel_efficiency,
            "irradiance": self._irradiance,
            "temperature": self._temperature,
            "temp_coefficient": self._temp_coefficient
        })
        return metrics
    
    def update_conditions(
        self,
        irradiance: float,
        temperature: Optional[float] = None
    ) -> None:
        """Update environmental conditions and recalculate power."""
        if irradiance < 0:
            raise ResourceError("Irradiance cannot be negative")
        
        self._irradiance = irradiance
        if temperature is not None:
            self._temperature = temperature
        
        # Calculate power output
        base_power = (
            self._irradiance *
            self.panel_area *
            self.panel_efficiency / 1000  # Convert W to kW
        )
        
        # Apply temperature correction
        temp_factor = 1 + self._temp_coefficient * (self._temperature - 25)
        power = base_power * temp_factor
        
        # Limit to rated power
        self._current_power = min(power, self.rated_power)
        self._last_update = datetime.utcnow()

class WindTurbine(EnergyResource):
    """Wind power generation system."""
    
    def __init__(
        self,
        rated_power: float,
        rotor_diameter: float,
        hub_height: float,
        cut_in_speed: float,
        cut_out_speed: float,
        rated_speed: float
    ):
        """Initialize wind turbine with specifications."""
        super().__init__(rated_power=rated_power)
        
        if rotor_diameter <= 0:
            raise ResourceError("Rotor diameter must be positive")
        if hub_height <= 0:
            raise ResourceError("Hub height must be positive")
        if cut_in_speed < 0:
            raise ResourceError("Cut-in speed cannot be negative")
        if cut_out_speed <= cut_in_speed:
            raise ResourceError("Cut-out speed must be greater than cut-in speed")
        if rated_speed <= cut_in_speed or rated_speed >= cut_out_speed:
            raise ResourceError(
                "Rated speed must be between cut-in and cut-out speeds"
            )
        
        self.rotor_diameter = rotor_diameter
        self.hub_height = hub_height
        self.cut_in_speed = cut_in_speed
        self.cut_out_speed = cut_out_speed
        self.rated_speed = rated_speed
        
        self._wind_speed = 0.0
        self._air_density = 1.225  # kg/m³ at sea level, 15°C
        self._power_coefficient = 0.4  # Typical value
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get wind turbine-specific metrics."""
        metrics = super().get_metrics()
        metrics.update({
            "rotor_diameter": self.rotor_diameter,
            "hub_height": self.hub_height,
            "cut_in_speed": self.cut_in_speed,
            "cut_out_speed": self.cut_out_speed,
            "rated_speed": self.rated_speed,
            "wind_speed": self._wind_speed,
            "air_density": self._air_density,
            "power_coefficient": self._power_coefficient
        })
        return metrics
    
    def update_wind(
        self,
        wind_speed: float,
        air_density: Optional[float] = None
    ) -> None:
        """Update wind conditions and recalculate power output."""
        if wind_speed < 0:
            raise ResourceError("Wind speed cannot be negative")
        
        self._wind_speed = wind_speed
        if air_density is not None:
            if air_density <= 0:
                raise ResourceError("Air density must be positive")
            self._air_density = air_density
        
        # Calculate power based on wind speed
        if wind_speed < self.cut_in_speed or wind_speed > self.cut_out_speed:
            power = 0.0
        elif wind_speed >= self.rated_speed:
            power = self.rated_power
        else:
            # Power curve between cut-in and rated speed
            swept_area = 3.14159 * (self.rotor_diameter / 2) ** 2
            power = (
                0.5 *
                self._air_density *
                swept_area *
                wind_speed ** 3 *
                self._power_coefficient / 1000  # Convert W to kW
            )
            power = min(power, self.rated_power)
        
        self._current_power = power
        self._last_update = datetime.utcnow()
