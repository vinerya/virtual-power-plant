"""
Distributed optimization for coordinating multiple VPP sites.
Provides rule-based fallbacks and plugin architecture for expert models.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
import time

from .base import (
    OptimizationPlugin, RuleBasedOptimizer, OptimizationProblem, 
    OptimizationResult, OptimizationStatus
)


@dataclass
class SiteState:
    """State of an individual VPP site."""
    site_id: str
    location: str
    total_capacity: float  # kW
    available_capacity: float  # kW
    current_generation: float  # kW
    current_load: float  # kW
    battery_soc: float  # 0-1
    battery_capacity: float  # kWh
    marginal_cost: float  # $/kWh
    constraints: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoordinationSignal:
    """Coordination signal between VPP sites."""
    target_power: Dict[str, float]  # site_id -> power setpoint (kW)
    price_signals: Dict[str, float]  # site_id -> price signal ($/kWh)
    reserve_allocations: Dict[str, float]  # site_id -> reserve capacity (kW)
    coordination_method: str
    convergence_achieved: bool
    iterations: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimpleConsensusRules(RuleBasedOptimizer):
    """Simple rule-based coordination for distributed VPP sites."""
    
    def __init__(self):
        super().__init__("simple_consensus_rules")
        self.price_tolerance = 0.01  # $/kWh
        self.power_tolerance = 1.0   # kW
        self.max_iterations = 10
    
    def solve(self, problem: OptimizationProblem) -> OptimizationResult:
        """Solve using simple consensus-based coordination."""
        start_time = datetime.now()
        
        try:
            # Extract site states
            sites = self._extract_site_states(problem)
            
            # Apply coordination rules
            coordination_signal = self._coordinate_sites(sites, problem)
            
            solve_time = (datetime.now() - start_time).total_seconds()
            
            # Create solution
            solution = {
                "target_power": coordination_signal.target_power,
                "price_signals": coordination_signal.price_signals,
                "reserve_allocations": coordination_signal.reserve_allocations,
                "coordination_method": coordination_signal.coordination_method,
                "convergence_achieved": coordination_signal.convergence_achieved,
                "iterations": coordination_signal.iterations,
                "method": "simple_consensus_rules"
            }
            
            return OptimizationResult(
                status=OptimizationStatus.SUCCESS,
                objective_value=self._calculate_total_cost(sites, coordination_signal),
                solution=solution,
                solve_time=solve_time,
                metadata={
                    "method": "simple_consensus_rules",
                    "num_sites": len(sites),
                    "convergence_achieved": coordination_signal.convergence_achieved
                }
            )
            
        except Exception as e:
            solve_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Simple consensus coordination failed: {e}")
            
            return OptimizationResult(
                status=OptimizationStatus.FAILED,
                objective_value=float('inf'),
                solution={},
                solve_time=solve_time,
                metadata={"error": str(e)}
            )
    
    def _extract_site_states(self, problem: OptimizationProblem) -> List[SiteState]:
        """Extract site states from optimization problem."""
        sites_data = problem.parameters.get("sites", [])
        sites = []
        
        for site_data in sites_data:
            site = SiteState(
                site_id=site_data.get("site_id", "unknown"),
                location=site_data.get("location", "unknown"),
                total_capacity=site_data.get("total_capacity", 0.0),
                available_capacity=site_data.get("available_capacity", 0.0),
                current_generation=site_data.get("current_generation", 0.0),
                current_load=site_data.get("current_load", 0.0),
                battery_soc=site_data.get("battery_soc", 0.5),
                battery_capacity=site_data.get("battery_capacity", 0.0),
                marginal_cost=site_data.get("marginal_cost", 0.1),
                constraints=site_data.get("constraints", {}),
                metadata=site_data.get("metadata", {})
            )
            sites.append(site)
        
        return sites
    
    def _coordinate_sites(self, sites: List[SiteState], 
                         problem: OptimizationProblem) -> CoordinationSignal:
        """Coordinate sites using simple consensus rules."""
        
        # Target total power from problem
        target_total_power = problem.parameters.get("target_total_power", 0.0)
        target_reserve = problem.parameters.get("target_reserve", 0.0)
        
        # Initialize coordination signal
        coordination = CoordinationSignal(
            target_power={},
            price_signals={},
            reserve_allocations={},
            coordination_method="merit_order_with_consensus",
            convergence_achieved=False,
            iterations=0
        )
        
        # Simple merit order dispatch
        if target_total_power != 0:
            coordination = self._merit_order_dispatch(sites, target_total_power, coordination)
        
        # Allocate reserves based on capacity
        if target_reserve > 0:
            coordination = self._allocate_reserves(sites, target_reserve, coordination)
        
        # Apply load balancing if needed
        coordination = self._balance_loads(sites, coordination)
        
        coordination.convergence_achieved = True  # Simple rules always "converge"
        coordination.iterations = 1
        
        return coordination
    
    def _merit_order_dispatch(self, sites: List[SiteState], 
                            target_power: float,
                            coordination: CoordinationSignal) -> CoordinationSignal:
        """Dispatch sites based on merit order (marginal cost)."""
        
        # Sort sites by marginal cost
        if target_power > 0:  # Need generation
            sorted_sites = sorted(sites, key=lambda s: s.marginal_cost)
        else:  # Need load (negative generation)
            sorted_sites = sorted(sites, key=lambda s: -s.marginal_cost)
        
        remaining_power = abs(target_power)
        
        for site in sorted_sites:
            if remaining_power <= 0:
                coordination.target_power[site.site_id] = 0.0
                coordination.price_signals[site.site_id] = site.marginal_cost
                continue
            
            # Calculate available capacity for this direction
            if target_power > 0:  # Generation needed
                available = site.available_capacity
            else:  # Load needed
                available = site.current_load  # Can reduce load
            
            # Allocate power
            allocated_power = min(remaining_power, available)
            
            if target_power > 0:
                coordination.target_power[site.site_id] = allocated_power
            else:
                coordination.target_power[site.site_id] = -allocated_power
            
            coordination.price_signals[site.site_id] = site.marginal_cost
            remaining_power -= allocated_power
        
        return coordination
    
    def _allocate_reserves(self, sites: List[SiteState], 
                          target_reserve: float,
                          coordination: CoordinationSignal) -> CoordinationSignal:
        """Allocate reserve capacity among sites."""
        
        # Calculate total available reserve capacity
        total_reserve_capacity = sum(
            max(0, site.total_capacity - site.current_generation) 
            for site in sites
        )
        
        if total_reserve_capacity <= 0:
            # No reserve capacity available
            for site in sites:
                coordination.reserve_allocations[site.site_id] = 0.0
            return coordination
        
        # Allocate reserves proportionally to available capacity
        for site in sites:
            site_reserve_capacity = max(0, site.total_capacity - site.current_generation)
            reserve_fraction = site_reserve_capacity / total_reserve_capacity
            allocated_reserve = min(target_reserve * reserve_fraction, site_reserve_capacity)
            coordination.reserve_allocations[site.site_id] = allocated_reserve
        
        return coordination
    
    def _balance_loads(self, sites: List[SiteState], 
                      coordination: CoordinationSignal) -> CoordinationSignal:
        """Apply load balancing rules."""
        
        # Calculate total allocated power
        total_allocated = sum(coordination.target_power.values())
        
        # If total is close to zero, apply load balancing
        if abs(total_allocated) < 10.0:  # Small imbalance
            # Distribute small imbalances based on available capacity
            total_capacity = sum(site.available_capacity for site in sites)
            
            if total_capacity > 0:
                for site in sites:
                    capacity_fraction = site.available_capacity / total_capacity
                    adjustment = -total_allocated * capacity_fraction
                    
                    current_allocation = coordination.target_power.get(site.site_id, 0.0)
                    coordination.target_power[site.site_id] = current_allocation + adjustment
        
        return coordination
    
    def _calculate_total_cost(self, sites: List[SiteState], 
                            coordination: CoordinationSignal) -> float:
        """Calculate total cost of coordination."""
        total_cost = 0.0
        
        for site in sites:
            power = coordination.target_power.get(site.site_id, 0.0)
            cost = power * site.marginal_cost
            total_cost += cost
        
        return total_cost


class ADMMDistributedPlugin(OptimizationPlugin):
    """ADMM-based distributed optimization plugin."""
    
    def __init__(self, rho: float = 1.0, max_iterations: int = 100):
        super().__init__("admm_distributed", "1.0")
        self.rho = rho  # ADMM penalty parameter
        self.max_iterations = max_iterations
        self.tolerance = 1e-3
        self._solver_available = False
    
    def _initialize_impl(self, config: Dict[str, Any]) -> None:
        """Initialize ADMM solver."""
        self.rho = config.get("rho", 1.0)
        self.max_iterations = config.get("max_iterations", 100)
        self.tolerance = config.get("tolerance", 1e-3)
        
        try:
            import cvxpy as cp
            self._solver_available = True
            self.logger.info("ADMM distributed plugin initialized with CVXPY")
        except ImportError:
            self.logger.warning("CVXPY not available, ADMM plugin disabled")
            self._solver_available = False
    
    def is_available(self) -> bool:
        """Check if ADMM solver is available."""
        return self._is_initialized and self._solver_available
    
    def validate_problem(self, problem: OptimizationProblem) -> bool:
        """Validate that problem can be solved with ADMM."""
        sites = problem.parameters.get("sites", [])
        return len(sites) > 1 and self._solver_available
    
    def solve(self, problem: OptimizationProblem, timeout_ms: Optional[int] = None) -> OptimizationResult:
        """Solve using ADMM distributed optimization."""
        if not self.is_available():
            raise RuntimeError("ADMM plugin not available")
        
        start_time = datetime.now()
        
        try:
            import cvxpy as cp
            
            # Extract problem data
            sites_data = problem.parameters.get("sites", [])
            target_total_power = problem.parameters.get("target_total_power", 0.0)
            num_sites = len(sites_data)
            
            if num_sites < 2:
                raise ValueError("ADMM requires at least 2 sites")
            
            # ADMM variables
            # Local variables for each site
            local_powers = [cp.Variable(name=f"power_{i}") for i in range(num_sites)]
            
            # Global consensus variable
            global_power = cp.Variable(name="global_power")
            
            # Dual variables (Lagrange multipliers)
            dual_vars = [cp.Variable(name=f"dual_{i}") for i in range(num_sites)]
            
            # ADMM iterations
            iteration = 0
            converged = False
            
            # Initialize variables
            for i in range(num_sites):
                local_powers[i].value = 0.0
                dual_vars[i].value = 0.0
            global_power.value = 0.0
            
            while iteration < self.max_iterations and not converged:
                # Step 1: Update local variables (parallel)
                for i, site_data in enumerate(sites_data):
                    self._update_local_variable(
                        local_powers[i], global_power, dual_vars[i], 
                        site_data, self.rho
                    )
                
                # Step 2: Update global variable
                avg_local = sum(p.value for p in local_powers) / num_sites
                global_power.value = avg_local
                
                # Step 3: Update dual variables
                for i in range(num_sites):
                    dual_vars[i].value += self.rho * (local_powers[i].value - global_power.value)
                
                # Check convergence
                primal_residual = np.sqrt(sum((p.value - global_power.value)**2 for p in local_powers))
                dual_residual = self.rho * np.sqrt(sum((p.value - avg_local)**2 for p in local_powers))
                
                if primal_residual < self.tolerance and dual_residual < self.tolerance:
                    converged = True
                
                iteration += 1
                
                # Timeout check
                if timeout_ms and (datetime.now() - start_time).total_seconds() * 1000 > timeout_ms:
                    break
            
            solve_time = (datetime.now() - start_time).total_seconds()
            
            # Create solution
            target_power = {
                sites_data[i]["site_id"]: float(local_powers[i].value) 
                for i in range(num_sites)
            }
            
            price_signals = {
                sites_data[i]["site_id"]: float(dual_vars[i].value) 
                for i in range(num_sites)
            }
            
            solution = {
                "target_power": target_power,
                "price_signals": price_signals,
                "reserve_allocations": {site["site_id"]: 0.0 for site in sites_data},
                "coordination_method": "admm",
                "convergence_achieved": converged,
                "iterations": iteration,
                "method": "admm_optimization"
            }
            
            total_cost = sum(
                local_powers[i].value * sites_data[i].get("marginal_cost", 0.1)
                for i in range(num_sites)
            )
            
            return OptimizationResult(
                status=OptimizationStatus.SUCCESS if converged else OptimizationStatus.TIMEOUT,
                objective_value=total_cost,
                solution=solution,
                solve_time=solve_time,
                metadata={
                    "iterations": iteration,
                    "converged": converged,
                    "rho": self.rho,
                    "num_sites": num_sites
                }
            )
            
        except Exception as e:
            solve_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"ADMM optimization failed: {e}")
            
            return OptimizationResult(
                status=OptimizationStatus.FAILED,
                objective_value=float('inf'),
                solution={},
                solve_time=solve_time,
                metadata={"error": str(e)}
            )
    
    def _update_local_variable(self, local_var, global_var, dual_var, 
                              site_data: Dict[str, Any], rho: float) -> None:
        """Update local variable in ADMM iteration."""
        import cvxpy as cp
        
        # Local cost function
        marginal_cost = site_data.get("marginal_cost", 0.1)
        max_capacity = site_data.get("available_capacity", 100.0)
        
        # Local problem: minimize cost + ADMM penalty
        cost = marginal_cost * local_var
        augmented_lagrangian = (cost + 
                               dual_var.value * (local_var - global_var.value) +
                               (rho / 2) * cp.square(local_var - global_var.value))
        
        # Constraints
        constraints = [
            local_var >= -max_capacity,
            local_var <= max_capacity
        ]
        
        # Solve local problem
        local_problem = cp.Problem(cp.Minimize(augmented_lagrangian), constraints)
        local_problem.solve(verbose=False)


class DistributedOptimizationManager:
    """Manager for distributed VPP coordination."""
    
    def __init__(self):
        self.logger = logging.getLogger("distributed_optimization")
        self._coordination_history = []
        self._max_history = 500
    
    def create_distributed_problem(self, sites_data: List[Dict[str, Any]],
                                 coordination_targets: Dict[str, float] = None) -> OptimizationProblem:
        """Create a distributed optimization problem."""
        
        coordination_targets = coordination_targets or {}
        
        problem = OptimizationProblem(
            variables={
                "site_powers": {"type": "continuous", "bounds": (-1000, 1000)},
                "reserve_allocations": {"type": "continuous", "bounds": (0, 1000)}
            },
            objectives=[{
                "name": "minimize_total_cost",
                "type": "minimize",
                "weight": 1.0
            }],
            constraints=[
                {"name": "power_balance", "type": "equality"},
                {"name": "capacity_limits", "type": "inequality"},
                {"name": "reserve_requirements", "type": "inequality"}
            ],
            parameters={
                "sites": sites_data,
                "target_total_power": coordination_targets.get("total_power", 0.0),
                "target_reserve": coordination_targets.get("reserve", 0.0),
                **coordination_targets
            },
            time_horizon=1,  # Single time step for coordination
            time_step=1.0,
            metadata={
                "type": "distributed",
                "num_sites": len(sites_data),
                "coordination_targets": coordination_targets
            }
        )
        
        return problem
    
    def record_coordination(self, coordination_result: CoordinationSignal) -> None:
        """Record coordination results for analysis."""
        entry = {
            "timestamp": datetime.now(),
            "method": coordination_result.coordination_method,
            "converged": coordination_result.convergence_achieved,
            "iterations": coordination_result.iterations,
            "num_sites": len(coordination_result.target_power),
            "total_power": sum(coordination_result.target_power.values()),
            "total_reserve": sum(coordination_result.reserve_allocations.values())
        }
        
        self._coordination_history.append(entry)
        
        if len(self._coordination_history) > self._max_history:
            self._coordination_history = self._coordination_history[-self._max_history:]
    
    def get_coordination_stats(self) -> Dict[str, Any]:
        """Get distributed coordination statistics."""
        if not self._coordination_history:
            return {}
        
        recent = self._coordination_history[-50:]  # Last 50 coordinations
        
        convergence_rate = sum(1 for h in recent if h["converged"]) / len(recent)
        avg_iterations = np.mean([h["iterations"] for h in recent])
        
        return {
            "convergence_rate": convergence_rate,
            "avg_iterations": avg_iterations,
            "total_coordinations": len(self._coordination_history),
            "methods_used": list(set(h["method"] for h in recent)),
            "avg_sites_coordinated": np.mean([h["num_sites"] for h in recent])
        }
    
    def validate_site_data(self, sites_data: List[Dict[str, Any]]) -> List[str]:
        """Validate site data for distributed optimization."""
        errors = []
        
        required_fields = ["site_id", "total_capacity", "available_capacity", "marginal_cost"]
        
        for i, site in enumerate(sites_data):
            for field in required_fields:
                if field not in site:
                    errors.append(f"Site {i}: Missing required field '{field}'")
                elif not isinstance(site[field], (int, float)):
                    errors.append(f"Site {i}: Field '{field}' must be numeric")
            
            # Validate capacity constraints
            if "total_capacity" in site and "available_capacity" in site:
                if site["available_capacity"] > site["total_capacity"]:
                    errors.append(f"Site {i}: Available capacity exceeds total capacity")
        
        # Check for duplicate site IDs
        site_ids = [site.get("site_id") for site in sites_data]
        if len(site_ids) != len(set(site_ids)):
            errors.append("Duplicate site IDs found")
        
        return errors
