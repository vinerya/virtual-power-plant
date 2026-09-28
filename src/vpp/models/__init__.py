"""
Advanced physics-based models for Virtual Power Plant resources.

This package provides sophisticated models for VPP components with:
- Physics-based battery models with electrochemical accuracy
- Aging and thermal dynamics simulation
- Safety monitoring and constraint enforcement
- Integration with the optimization framework

The models are designed to provide realistic behavior simulation
while maintaining computational efficiency for optimization.
"""

from .battery import (
    AdvancedElectrochemicalModel,
    BatteryModel,
    BatteryParameters,
    BatteryState,
    SimpleEquivalentCircuitModel,
    create_battery_model,
)

# Package metadata (the version lives in vpp.__version__ only)
__author__ = "VPP Development Team"

# Export all public classes and functions
__all__ = [
    # Battery models
    "BatteryState",
    "BatteryParameters",
    "BatteryModel",
    "SimpleEquivalentCircuitModel",
    "AdvancedElectrochemicalModel",
    "create_battery_model",
]
