"""
Stochastic optimization with uncertainty handling.
Provides rule-based fallbacks and plugin architecture for expert models.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
import random

from .base import (
    OptimizationPlugin, RuleBasedOptimizer, OptimizationProblem, 
    OptimizationResult, OptimizationStatus
)


@dataclass
class Scenario:
    """A single scenario for stochastic optimization."""
    id: str
    probability: float
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioSet:
    """Collection of scenarios with utilities."""
    scenarios: List[Scenario]
    base_case: Optional[Scenario] = None
    
    def __post_init__(self):
        # Normalize probabilities
        total_prob = sum(s.probability for s in self.scenarios)
        if total_prob > 0:
            for scenario in self.scenarios:
                scenario.probability /= total_prob
    
    def get_scenario_by_id(self, scenario_id: str) -> Optional[Scenario]:
        """Get scenario by ID."""
        for scenario in self.scenarios:
            if scenario.id == scenario_id:
                return scenario
        return None
    
    def get_expected_value(self, key: str) -> float:
        """Calculate expected value for a given key across scenarios."""
        return sum(s.probability * s.data.get(key, 0.0) for s in self.scenarios)
    
    def get_percentile(self, key: str, percentile: float) -> float:
        """Get percentile value for a given key."""
        values = [s.data.get(key, 0.0) for s in self.scenarios]
        return np.percentile(values, percentile)


class ScenarioGenerator:
    """Generate scenarios for stochastic optimization."""
    
    def __init__(self, random_seed: Optional[int] = None):
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
    
    def generate_price_scenarios(self, base_prices: List[float], 
                                num_scenarios: int = 10,
                                volatility: float = 0.2) -> ScenarioSet:
        """Generate electricity price scenarios."""
        scenarios = []
        
        for i in range(num_scenarios):
            # Generate correlated price variations
            price_multipliers = np.random.lognormal(
                mean=0, sigma=volatility, size=len(base_prices)
            )
            
            scenario_prices = [base * mult for base, mult in zip(base_prices, price_multipliers)]
            
            scenario = Scenario(
                id=f"price_scenario_{i}",
                probability=1.0 / num_scenarios,
                data={"prices": scenario_prices},
                metadata={"volatility": volatility, "type": "price"}
            )
            scenarios.append(scenario)
        
        return ScenarioSet(scenarios)
    
    def generate_renewable_scenarios(self, base_forecast: List[float],
                                   num_scenarios: int = 10,
                                   forecast_error: float = 0.15) -> ScenarioSet:
        """Generate renewable generation scenarios."""
        scenarios = []
        
        for i in range(num_scenarios):
            # Generate forecast errors with persistence
            errors = np.random.normal(0, forecast_error, len(base_forecast))
            
            # Add persistence to errors
            for j in range(1, len(errors)):
                errors[j] = 0.7 * errors[j-1] + 0.3 * errors[j]
            
            scenario_generation = [
                max(0, base * (1 + error)) 
                for base, error in zip(base_forecast, errors)
            ]
            
            scenario = Scenario(
                id=f"renewable_scenario_{i}",
                probability=1.0 / num_scenarios,
                data={"generation": scenario_generation},
                metadata={"forecast_error": forecast_error, "type": "renewable"}
            )
            scenarios.append(scenario)
        
        return ScenarioSet(scenarios)
    
    def generate_load_scenarios(self, base_load: List[float],
                              num_scenarios: int = 10,
                              uncertainty: float = 0.1) -> ScenarioSet:
        """Generate load demand scenarios."""
        scenarios = []
        
        for i in range(num_scenarios):
            # Generate load variations with daily patterns
            load_variations = np.random.normal(1.0, uncertainty, len(base_load))
            
            scenario_load = [
                max(0, base * variation) 
                for base, variation in zip(base_load, load_variations)
            ]
            
            scenario = Scenario(
                id=f"load_scenario_{i}",
                probability=1.0 / num_scenarios,
                data={"load": scenario_load},
                metadata={"uncertainty": uncertainty, "type": "load"}
            )
            scenarios.append(scenario)
        
        return ScenarioSet(scenarios)


class SimpleStochasticRules(RuleBasedOptimizer):
    """Rule-based stochastic optimization fallback."""
    
    def __init__(self):
        super().__init__("simple_stochastic_rules")
        self.risk_aversion = 0.1  # Conservative approach
    
    def solve(self, problem: OptimizationProblem) -> OptimizationResult:
        """Solve using conservative rule-based approach."""
        start_time = datetime.now()
        
        try:
            # Extract scenarios from problem
            scenarios = problem.parameters.get("scenarios", [])
            if not scenarios:
                return self._create_deterministic_solution(problem)
            
            # Use conservative approach: worst-case + safety margin
            solution = self._conservative_dispatch(problem, scenarios)
            
            solve_time = (datetime.now() - start_time).total_seconds()
            
            return OptimizationResult(
                status=OptimizationStatus.SUCCESS,
                objective_value=solution["objective"],
                solution=solution,
                solve_time=solve_time,
                metadata={
                    "method": "conservative_rules",
                    "num_scenarios": len(scenarios),
                    "risk_approach": "worst_case_plus_margin"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Rule-based stochastic optimization failed: {e}")
            solve_time = (datetime.now() - start_time).total_seconds()
            
            return OptimizationResult(
                status=OptimizationStatus.FAILED,
                objective_value=float('inf'),
                solution={},
                solve_time=solve_time,
                metadata={"error": str(e)}
            )
    
    def _create_deterministic_solution(self, problem: OptimizationProblem) -> OptimizationResult:
        """Create solution when no scenarios are provided."""
        # Simple deterministic dispatch
        time_horizon = problem.time_horizon
        time_step = problem.time_step
        num_periods = int(time_horizon / time_step)
        
        # Basic dispatch: charge during low prices, discharge during high prices
        solution = {
            "battery_power": [0.0] * num_periods,
            "battery_soc": [0.5] * num_periods,
            "objective": 0.0,
            "method": "deterministic_fallback"
        }
        
        return OptimizationResult(
            status=OptimizationStatus.SUCCESS,
            objective_value=0.0,
            solution=solution,
            solve_time=0.001,
            metadata={"method": "deterministic_fallback"}
        )
    
    def _conservative_dispatch(self, problem: OptimizationProblem, 
                             scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Conservative dispatch considering worst-case scenarios."""
        time_horizon = problem.time_horizon
        time_step = problem.time_step
        num_periods = int(time_horizon / time_step)
        
        # Extract scenario data
        price_scenarios = []
        renewable_scenarios = []
        load_scenarios = []
        
        for scenario in scenarios:
            if "prices" in scenario:
                price_scenarios.append(scenario["prices"])
            if "generation" in scenario:
                renewable_scenarios.append(scenario["generation"])
            if "load" in scenario:
                load_scenarios.append(scenario["load"])
        
        # Calculate conservative estimates
        if price_scenarios:
            # Use 90th percentile for buying, 10th percentile for selling
            conservative_buy_prices = [
                np.percentile([s[i] for s in price_scenarios], 90)
                for i in range(min(num_periods, len(price_scenarios[0])))
            ]
            conservative_sell_prices = [
                np.percentile([s[i] for s in price_scenarios], 10)
                for i in range(min(num_periods, len(price_scenarios[0])))
            ]
        else:
            conservative_buy_prices = [0.1] * num_periods
            conservative_sell_prices = [0.05] * num_periods
        
        if renewable_scenarios:
            # Use 10th percentile for renewable generation (conservative)
            conservative_renewable = [
                np.percentile([s[i] for s in renewable_scenarios], 10)
                for i in range(min(num_periods, len(renewable_scenarios[0])))
            ]
        else:
            conservative_renewable = [0.0] * num_periods
        
        if load_scenarios:
            # Use 90th percentile for load (conservative)
            conservative_load = [
                np.percentile([s[i] for s in load_scenarios], 90)
                for i in range(min(num_periods, len(load_scenarios[0])))
            ]
        else:
            conservative_load = [100.0] * num_periods
        
        # Simple rule-based dispatch
        battery_power = []
        battery_soc = [0.5]  # Start at 50% SOC
        total_cost = 0.0
        
        # Battery parameters (conservative defaults)
        battery_capacity = problem.parameters.get("battery_capacity", 1000.0)  # kWh
        max_power = problem.parameters.get("max_power", 250.0)  # kW
        efficiency = problem.parameters.get("efficiency", 0.9)
        
        for t in range(num_periods):
            current_soc = battery_soc[-1]
            
            # Net load (load - renewable)
            net_load = conservative_load[t] - conservative_renewable[t]
            
            # Dispatch rules
            if conservative_sell_prices[t] > conservative_buy_prices[t] * 1.2:
                # High price spread - discharge if possible
                if current_soc > 0.2:  # Keep 20% reserve
                    discharge_power = min(max_power, net_load, 
                                        (current_soc - 0.2) * battery_capacity / time_step)
                    power = -discharge_power
                else:
                    power = 0.0
            elif conservative_buy_prices[t] < conservative_sell_prices[t] * 0.8:
                # Low prices - charge if possible
                if current_soc < 0.9:  # Don't overcharge
                    charge_power = min(max_power, 
                                     (0.9 - current_soc) * battery_capacity / time_step)
                    power = charge_power
                else:
                    power = 0.0
            else:
                # Neutral prices - maintain SOC
                power = 0.0
            
            # Update SOC
            if power > 0:  # Charging
                energy_change = power * time_step * efficiency
                total_cost += power * conservative_buy_prices[t] * time_step
            else:  # Discharging
                energy_change = power * time_step / efficiency
                total_cost += power * conservative_sell_prices[t] * time_step
            
            new_soc = current_soc + energy_change / battery_capacity
            new_soc = max(0.1, min(0.95, new_soc))  # Enforce SOC limits
            
            battery_power.append(power)
            battery_soc.append(new_soc)
        
        return {
            "battery_power": battery_power,
            "battery_soc": battery_soc[1:],  # Remove initial SOC
            "conservative_buy_prices": conservative_buy_prices,
            "conservative_sell_prices": conservative_sell_prices,
            "conservative_renewable": conservative_renewable,
            "conservative_load": conservative_load,
            "objective": total_cost,
            "method": "conservative_rules"
        }


class CVaRStochasticPlugin(OptimizationPlugin):
    """Example expert plugin for CVaR-based stochastic optimization."""
    
    def __init__(self, risk_level: float = 0.05):
        super().__init__("cvar_stochastic", "1.0")
        self.risk_level = risk_level
        self._solver_available = False
    
    def _initialize_impl(self, config: Dict[str, Any]) -> None:
        """Initialize CVaR solver."""
        self.risk_level = config.get("risk_level", 0.05)
        
        # Check if required packages are available
        try:
            import cvxpy as cp
            self._solver_available = True
            self.logger.info("CVaR stochastic plugin initialized with CVXPY")
        except ImportError:
            self.logger.warning("CVXPY not available, CVaR plugin disabled")
            self._solver_available = False
    
    def is_available(self) -> bool:
        """Check if CVaR solver is available."""
        return self._is_initialized and self._solver_available
    
    def validate_problem(self, problem: OptimizationProblem) -> bool:
        """Validate that problem can be solved with CVaR."""
        scenarios = problem.parameters.get("scenarios", [])
        return len(scenarios) > 0 and self._solver_available
    
    def solve(self, problem: OptimizationProblem, timeout_ms: Optional[int] = None) -> OptimizationResult:
        """Solve using CVaR optimization."""
        if not self.is_available():
            raise RuntimeError("CVaR plugin not available")
        
        start_time = datetime.now()
        
        try:
            import cvxpy as cp
            
            # Extract problem data
            scenarios = problem.parameters.get("scenarios", [])
            time_horizon = problem.time_horizon
            time_step = problem.time_step
            num_periods = int(time_horizon / time_step)
            num_scenarios = len(scenarios)
            
            # Decision variables
            battery_power = cp.Variable((num_periods,), name="battery_power")
            battery_soc = cp.Variable((num_periods + 1,), name="battery_soc")
            
            # CVaR variables
            scenario_costs = cp.Variable((num_scenarios,), name="scenario_costs")
            var = cp.Variable(name="var")  # Value at Risk
            cvar_losses = cp.Variable((num_scenarios,), name="cvar_losses")
            
            # Parameters
            battery_capacity = problem.parameters.get("battery_capacity", 1000.0)
            max_power = problem.parameters.get("max_power", 250.0)
            efficiency = problem.parameters.get("efficiency", 0.9)
            
            # Constraints
            constraints = []
            
            # Initial SOC
            constraints.append(battery_soc[0] == 0.5)
            
            # SOC dynamics and limits
            for t in range(num_periods):
                # SOC update
                energy_change = cp.multiply(battery_power[t] * time_step, 
                                          cp.maximum(efficiency, 1/efficiency))
                constraints.append(
                    battery_soc[t+1] == battery_soc[t] + energy_change / battery_capacity
                )
                
                # SOC limits
                constraints.append(battery_soc[t+1] >= 0.1)
                constraints.append(battery_soc[t+1] <= 0.9)
                
                # Power limits
                constraints.append(battery_power[t] >= -max_power)
                constraints.append(battery_power[t] <= max_power)
            
            # Scenario costs
            for s, scenario in enumerate(scenarios):
                prices = scenario.get("prices", [0.1] * num_periods)
                cost = cp.sum([battery_power[t] * prices[t] * time_step 
                              for t in range(min(num_periods, len(prices)))])
                constraints.append(scenario_costs[s] == cost)
            
            # CVaR constraints
            for s in range(num_scenarios):
                constraints.append(cvar_losses[s] >= 0)
                constraints.append(cvar_losses[s] >= scenario_costs[s] - var)
            
            # Objective: minimize expected cost + CVaR penalty
            expected_cost = cp.sum(scenario_costs) / num_scenarios
            cvar = var + cp.sum(cvar_losses) / (num_scenarios * self.risk_level)
            
            objective = cp.Minimize(expected_cost + 0.1 * cvar)
            
            # Solve
            problem_cvx = cp.Problem(objective, constraints)
            
            # Set timeout
            solver_options = {}
            if timeout_ms:
                solver_options["max_iters"] = min(1000, timeout_ms // 10)
            
            problem_cvx.solve(**solver_options)
            
            solve_time = (datetime.now() - start_time).total_seconds()
            
            if problem_cvx.status == cp.OPTIMAL:
                solution = {
                    "battery_power": battery_power.value.tolist(),
                    "battery_soc": battery_soc.value[1:].tolist(),
                    "scenario_costs": scenario_costs.value.tolist(),
                    "var": var.value,
                    "cvar": cvar.value,
                    "expected_cost": expected_cost.value,
                    "method": "cvar_optimization"
                }
                
                return OptimizationResult(
                    status=OptimizationStatus.SUCCESS,
                    objective_value=problem_cvx.value,
                    solution=solution,
                    solve_time=solve_time,
                    metadata={
                        "risk_level": self.risk_level,
                        "num_scenarios": num_scenarios,
                        "solver_status": problem_cvx.status
                    }
                )
            else:
                return OptimizationResult(
                    status=OptimizationStatus.FAILED,
                    objective_value=float('inf'),
                    solution={},
                    solve_time=solve_time,
                    metadata={"solver_status": problem_cvx.status}
                )
                
        except Exception as e:
            solve_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"CVaR optimization failed: {e}")
            
            return OptimizationResult(
                status=OptimizationStatus.FAILED,
                objective_value=float('inf'),
                solution={},
                solve_time=solve_time,
                metadata={"error": str(e)}
            )


class StochasticOptimizationManager:
    """Manager for stochastic optimization with scenario generation."""
    
    def __init__(self):
        self.scenario_generator = ScenarioGenerator()
        self.logger = logging.getLogger("stochastic_optimization")
    
    def create_stochastic_problem(self, 
                                base_data: Dict[str, Any],
                                num_scenarios: int = 10,
                                uncertainty_config: Dict[str, float] = None) -> OptimizationProblem:
        """Create a stochastic optimization problem with scenarios."""
        
        uncertainty_config = uncertainty_config or {
            "price_volatility": 0.2,
            "renewable_error": 0.15,
            "load_uncertainty": 0.1
        }
        
        # Generate scenarios
        scenarios = []
        
        # Price scenarios
        if "base_prices" in base_data:
            price_scenarios = self.scenario_generator.generate_price_scenarios(
                base_data["base_prices"], 
                num_scenarios,
                uncertainty_config.get("price_volatility", 0.2)
            )
            scenarios.extend([s.data for s in price_scenarios.scenarios])
        
        # Renewable scenarios
        if "renewable_forecast" in base_data:
            renewable_scenarios = self.scenario_generator.generate_renewable_scenarios(
                base_data["renewable_forecast"],
                num_scenarios,
                uncertainty_config.get("renewable_error", 0.15)
            )
            # Merge with existing scenarios
            for i, scenario in enumerate(scenarios):
                if i < len(renewable_scenarios.scenarios):
                    scenario.update(renewable_scenarios.scenarios[i].data)
        
        # Load scenarios
        if "load_forecast" in base_data:
            load_scenarios = self.scenario_generator.generate_load_scenarios(
                base_data["load_forecast"],
                num_scenarios,
                uncertainty_config.get("load_uncertainty", 0.1)
            )
            # Merge with existing scenarios
            for i, scenario in enumerate(scenarios):
                if i < len(load_scenarios.scenarios):
                    scenario.update(load_scenarios.scenarios[i].data)
        
        # Create optimization problem
        problem = OptimizationProblem(
            variables={
                "battery_power": {"type": "continuous", "bounds": (-250, 250)},
                "battery_soc": {"type": "continuous", "bounds": (0.1, 0.9)}
            },
            objectives=[{
                "name": "minimize_cost",
                "type": "minimize",
                "weight": 1.0
            }],
            constraints=[
                {"name": "soc_dynamics", "type": "equality"},
                {"name": "power_limits", "type": "inequality"},
                {"name": "soc_limits", "type": "inequality"}
            ],
            parameters={
                "scenarios": scenarios,
                "battery_capacity": base_data.get("battery_capacity", 1000.0),
                "max_power": base_data.get("max_power", 250.0),
                "efficiency": base_data.get("efficiency", 0.9),
                **base_data
            },
            time_horizon=base_data.get("time_horizon", 24),
            time_step=base_data.get("time_step", 1.0),
            metadata={
                "type": "stochastic",
                "num_scenarios": num_scenarios,
                "uncertainty_config": uncertainty_config
            }
        )
        
        return problem
