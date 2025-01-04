"""Advanced optimization strategies for the Virtual Power Plant."""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable, Tuple, Set
from dataclasses import dataclass
import numpy as np
from pulp import (
    LpProblem, LpMinimize, LpMaximize, LpVariable, lpSum, LpStatus,
    LpBinary, LpInteger, LpContinuous
)

from .resources import EnergyResource, Battery
from .exceptions import OptimizationError

@dataclass
class OptimizationConstraint:
    """Constraint definition for optimization."""
    min_power: float = 0.0
    max_power: Optional[float] = None
    ramp_up_rate: Optional[float] = None  # kW/min
    ramp_down_rate: Optional[float] = None  # kW/min
    min_uptime: Optional[int] = None  # minutes
    min_downtime: Optional[int] = None  # minutes
    maintenance_window: Optional[Tuple[datetime, datetime]] = None
    priority: int = 1  # 1 (highest) to 5 (lowest)
    cost_per_kwh: Optional[float] = None

@dataclass
class OptimizationResult:
    """Detailed result of an optimization run."""
    success: bool
    target_power: float
    actual_power: float
    resource_allocation: Dict[str, float]
    constraints_satisfied: Dict[str, bool]
    performance_metrics: Dict[str, float]
    computation_time: float  # seconds
    iteration_count: int
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class OptimizationProfile:
    """Profile for optimization preferences."""
    mode: str = "balanced"  # balanced, cost_minimizing, emission_minimizing
    cost_weight: float = 1.0
    emission_weight: float = 1.0
    reliability_weight: float = 1.0
    constraints: Dict[str, OptimizationConstraint] = None
    time_horizon: int = 60  # minutes
    resolution: int = 5  # minutes

class OptimizationStrategy(ABC):
    """Abstract base class for optimization strategies."""
    
    def __init__(self, **kwargs):
        """Initialize strategy with optional parameters."""
        self.parameters = kwargs
    
    @abstractmethod
    def optimize(
        self,
        resources: List[EnergyResource],
        target_power: float
    ) -> OptimizationResult:
        """Optimize resource dispatch to meet target power."""
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return self.parameters

class RuleBasedStrategy(OptimizationStrategy):
    """Rule-based optimization with customizable rules."""
    
    def __init__(self, rules: List[Callable[[EnergyResource], float]] = None, **kwargs):
        """Initialize with optional custom rules."""
        super().__init__(**kwargs)
        self.rules = rules or [
            lambda r: r.rated_power,  # Default rule: use rated power
            lambda r: getattr(r, '_efficiency', 0.95),  # Consider efficiency
        ]
    
    def optimize(
        self,
        resources: List[EnergyResource],
        target_power: float
    ) -> OptimizationResult:
        """Apply rules to optimize resource dispatch."""
        try:
            # Calculate resource scores based on rules
            scores = {}
            for resource in resources:
                score = sum(rule(resource) for rule in self.rules)
                scores[resource] = score
            
            # Normalize scores
            total_score = sum(scores.values())
            if total_score == 0:
                raise OptimizationError("No valid resource scores")
            
            # Allocate power based on scores
            actual_power = 0
            allocations = {}
            
            for resource in resources:
                power = (scores[resource] / total_score) * target_power
                power = min(power, resource.rated_power)
                resource.set_power(power)
                actual_power += power
                allocations[resource.__class__.__name__] = power
            
            return OptimizationResult(
                success=True,
                target_power=target_power,
                actual_power=actual_power,
                resource_allocation=allocations,
                metadata={"scores": {r.__class__.__name__: s for r, s in scores.items()}}
            )
            
        except Exception as e:
            return OptimizationResult(
                success=False,
                target_power=target_power,
                actual_power=0,
                resource_allocation={},
                metadata={"error": str(e)}
            )

class LinearProgrammingStrategy(OptimizationStrategy):
    """Linear Programming based optimization."""
    
    def optimize(
        self,
        resources: List[EnergyResource],
        target_power: float
    ) -> OptimizationResult:
        """Optimize using Linear Programming."""
        try:
            # Create optimization problem
            prob = LpProblem("VPP_Optimization", LpMinimize)
            
            # Decision variables
            power_vars = {
                r: LpVariable(f"power_{i}", 0, r.rated_power)
                for i, r in enumerate(resources)
            }
            
            # Objective: Minimize deviation from target
            prob += lpSum([power_vars[r] for r in resources]) - target_power
            
            # Constraints
            for r in resources:
                prob += power_vars[r] >= 0  # Non-negative power
                prob += power_vars[r] <= r.rated_power  # Capacity limit
            
            # Solve
            prob.solve()
            
            if LpStatus[prob.status] == 'Optimal':
                # Apply solution
                actual_power = 0
                allocations = {}
                
                for resource in resources:
                    power = power_vars[resource].value()
                    resource.set_power(power)
                    actual_power += power
                    allocations[resource.__class__.__name__] = power
                
                return OptimizationResult(
                    success=True,
                    target_power=target_power,
                    actual_power=actual_power,
                    resource_allocation=allocations,
                    metadata={"solver_status": LpStatus[prob.status]}
                )
            else:
                raise OptimizationError(f"Solver status: {LpStatus[prob.status]}")
            
        except Exception as e:
            return OptimizationResult(
                success=False,
                target_power=target_power,
                actual_power=0,
                resource_allocation={},
                metadata={"error": str(e)}
            )

class ReinforcementLearningStrategy(OptimizationStrategy):
    """Placeholder for Reinforcement Learning based optimization."""
    
    def __init__(self, model_path: Optional[str] = None, **kwargs):
        """Initialize RL strategy (placeholder)."""
        super().__init__(**kwargs)
        self.model_path = model_path
        print("Note: RL strategy is a placeholder for future implementation")
    
    def optimize(
        self,
        resources: List[EnergyResource],
        target_power: float
    ) -> OptimizationResult:
        """Placeholder optimization using RL (currently uses simple proportional)."""
        # For now, implement a simple fallback strategy
        total_capacity = sum(r.rated_power for r in resources)
        if total_capacity == 0:
            return OptimizationResult(
                success=False,
                target_power=target_power,
                actual_power=0,
                resource_allocation={},
                metadata={"note": "RL not implemented, no capacity available"}
            )
        
        ratio = target_power / total_capacity
        actual_power = 0
        allocations = {}
        
        for resource in resources:
            power = resource.rated_power * ratio
            resource.set_power(power)
            actual_power += power
            allocations[resource.__class__.__name__] = power
        
        return OptimizationResult(
            success=True,
            target_power=target_power,
            actual_power=actual_power,
            resource_allocation=allocations,
            metadata={
                "note": "RL not implemented, using proportional distribution",
                "ratio": ratio
            }
        )

def get_strategy(name: str, **kwargs) -> OptimizationStrategy:
    """Factory function to create optimization strategy."""
    strategies = {
        "rule_based": RuleBasedStrategy,
        "lp": LinearProgrammingStrategy,
        "rl": ReinforcementLearningStrategy
    }
    
    if name not in strategies:
        raise ValueError(f"Unknown optimization strategy: {name}")
    
    return strategies[name](**kwargs)
