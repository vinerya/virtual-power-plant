"""Configuration management for the Virtual Power Plant."""

from dataclasses import dataclass
from typing import Dict, Any, Optional

from .exceptions import ConfigurationError

@dataclass
class OptimizationConfig:
    """Optimization configuration."""
    strategy: str = "proportional"
    parameters: Dict[str, Any] = None

@dataclass
class VPPConfig:
    """Main VPP configuration."""
    
    def __init__(
        self,
        name: str,
        optimization: Optional[Dict[str, Any]] = None
    ):
        """Initialize configuration."""
        if not name:
            raise ConfigurationError("Name is required")
        self.name = name
        
        # Initialize optimization config
        self.optimization = OptimizationConfig(**optimization) if optimization else OptimizationConfig()
