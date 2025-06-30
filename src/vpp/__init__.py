"""Virtual Power Plant (VPP) library initialization."""

from .core import VirtualPowerPlant
from .config import VPPConfig
from .exceptions import VPPError

# Import advanced modules
from . import optimization
from . import models

__version__ = "1.0.0"
__author__ = "VPP Development Team"
__license__ = "MIT"

__all__ = [
    "VirtualPowerPlant",
    "VPPConfig", 
    "VPPError",
    "optimization",
    "models"
]
