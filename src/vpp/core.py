"""Core Virtual Power Plant implementation."""

from datetime import datetime
from typing import List, Optional, Dict, Any

from .config import VPPConfig
from .resources import EnergyResource
from .exceptions import ResourceError, OptimizationError

class VirtualPowerPlant:
    """Main VPP class that manages resources and optimization."""
    
    def __init__(self, config: VPPConfig):
        """Initialize VPP with configuration."""
        self.config = config
        self.resources: List[EnergyResource] = []
        self.target_power: Optional[float] = None
        self.last_optimization_time: Optional[datetime] = None
    
    def add_resource(self, resource: EnergyResource) -> None:
        """Add a new energy resource to the VPP."""
        self.resources.append(resource)
    
    def remove_resource(self, resource: EnergyResource) -> None:
        """Remove a resource from the VPP."""
        if resource in self.resources:
            self.resources.remove(resource)
    
    @property
    def total_capacity(self) -> float:
        """Get total power capacity of all resources."""
        return sum(r.rated_power for r in self.resources)
    
    def get_total_power(self) -> float:
        """Get current total power output."""
        return sum(r._current_power for r in self.resources)
    
    def optimize_dispatch(self, target_power: float) -> bool:
        """Optimize resource dispatch to meet target power."""
        try:
            if target_power < 0:
                raise OptimizationError("Target power cannot be negative")
            
            if target_power > self.total_capacity:
                raise OptimizationError("Target power exceeds total capacity")
            
            # Simple proportional distribution
            total_capacity = self.total_capacity
            if total_capacity == 0:
                return False
            
            ratio = target_power / total_capacity
            for resource in self.resources:
                resource._current_power = resource.rated_power * ratio
            
            self.target_power = target_power
            self.last_optimization_time = datetime.utcnow()
            
            return True
            
        except Exception:
            return False
