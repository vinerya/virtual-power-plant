"""
Configuration package for the Virtual Power Plant library.
Provides comprehensive, hierarchical, and validatable configuration management.
"""

from .base import (
    BaseConfig,
    ConfigFormat,
    ConfigValidationResult,
    ConstraintConfig,
    HeuristicConfig,
    OptimizationConfig,
    OptimizationObjective,
    RuleConfig,
    RuleEngineConfig,
    ValidationLevel,
)
from .vpp_config import (
    MonitoringConfig,
    ResourceConfig,
    SecurityConfig,
    SimulationConfig,
    VPPConfig,
)

__all__ = [
    "BaseConfig",
    "ConfigFormat",
    "ConfigValidationResult",
    "ConstraintConfig",
    "HeuristicConfig",
    "MonitoringConfig",
    "OptimizationConfig",
    "OptimizationObjective",
    "ResourceConfig",
    "RuleConfig",
    "RuleEngineConfig",
    "SecurityConfig",
    "SimulationConfig",
    "VPPConfig",
    "ValidationLevel",
]

# Package metadata (the version lives in vpp.__version__ only)
__author__ = "VPP Development Team"
