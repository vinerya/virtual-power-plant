"""Data models for the Virtual Power Plant."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional

@dataclass
class ResourceState:
    """Current state of an energy resource."""
    power_output: float
    status: str  # ONLINE, OFFLINE
    timestamp: datetime
    metrics: Dict[str, Any]

@dataclass
class SystemState:
    """Current state of the VPP system."""
    total_power: float
    timestamp: datetime
    metrics: Dict[str, Any]

@dataclass
class OptimizationResult:
    """Result of an optimization run."""
    success: bool
    target_power: float
    actual_power: float
    timestamp: datetime

@dataclass
class WeatherData:
    """Weather conditions affecting resources."""
    temperature: float
    irradiance: Optional[float] = None  # W/m²
    wind_speed: Optional[float] = None  # m/s
    timestamp: datetime = datetime.utcnow()
