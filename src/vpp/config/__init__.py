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
    # Base configuration classes
    "BaseConfig",
    "ConfigFormat",
    "ValidationLevel",
    "ConfigValidationResult",
    # Optimization configuration
    "OptimizationObjective",
    "ConstraintConfig",
    "OptimizationConfig",
    # Heuristic configuration
    "HeuristicConfig",
    # Rule engine configuration
    "RuleConfig",
    "RuleEngineConfig",
    # VPP configuration components
    "ResourceConfig",
    "MonitoringConfig",
    "SimulationConfig",
    "SecurityConfig",
    # Main configuration class
    "VPPConfig",
]

# Package metadata (the version lives in vpp.__version__ only)
__author__ = "VPP Development Team"
