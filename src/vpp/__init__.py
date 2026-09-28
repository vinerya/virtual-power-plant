"""Virtual Power Plant (VPP) library initialization."""

from . import models, optimization
from ._version import __version__
from .config import VPPConfig
from .core import VirtualPowerPlant
from .exceptions import VPPError

__author__ = "VPP Development Team"
__license__ = "MIT"

__all__ = ["VPPConfig", "VPPError", "VirtualPowerPlant", "__version__", "models", "optimization"]
