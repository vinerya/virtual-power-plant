"""Virtual Power Plant (VPP) library initialization."""

from .models import WeatherData
from .core import VirtualPowerPlant
from .config import VPPConfig
from .resources import Battery, Solar, WindTurbine
from .exceptions import VPPError

__version__ = "0.1.0"
__author__ = "VPP Team"
__license__ = "MIT"

__all__ = [
    "VirtualPowerPlant",
    "VPPConfig",
    "Battery",
    "Solar",
    "WindTurbine",
    "WeatherData",
    "VPPError"
]
