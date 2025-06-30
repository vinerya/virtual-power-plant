"""
Configuration package for the Virtual Power Plant library.
Provides comprehensive, hierarchical, and validatable configuration management.
"""

from .base import (
    BaseConfig,
    ConfigFormat,
    ValidationLevel,
    ConfigValidationResult,
    OptimizationObjective,
    ConstraintConfig,
    OptimizationConfig,
    HeuristicConfig,
    RuleConfig,
    RuleEngineConfig
)

from .vpp_config import (
    ResourceConfig,
    MonitoringConfig,
    SimulationConfig,
    SecurityConfig,
    VPPConfig
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
    "VPPConfig"
]

# Version information
__version__ = "1.0.0"
__author__ = "VPP Development Team"
