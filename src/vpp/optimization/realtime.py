"""
Real-time optimization for sub-second grid services and frequency response.
Provides fast rule-based fallbacks and plugin architecture for expert models.
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
class RealTimeState:
    """Current real-time state of the VPP system."""
    timestamp: datetime
    grid_frequency: float  # Hz
    grid_voltage: float    # V
    total_load: float      # kW
    renewable_generation: float  # kW
    battery_soc: float     # 0-1
    battery_power: float   # kW (positive = charging)
    electricity_price: float  # $/kWh
    ancillary_service_signals: Dict[str, float] = field(default_factory=dict)
    grid_constraints: Dict[str, float] = field(default_factory=dict)


@dataclass
class RealTimeControlSignal:
    """Control signal for real-time optimization."""
    battery_power_setpoint: float  # kW
    renewable_curtailment: float   # kW
    load_shedding: float          # kW
    frequency_response_active: bool
    voltage_support_active: bool
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class FastDispatchRules(RuleBasedOptimizer):
    """Ultra-fast rule-based dispatch for real-time control."""
    
    def __init__(self):
        super().__init__("fast_dispatch_rules")
        
        # Control parameters
        self.frequency_deadband = 0.05  # Hz
        self.voltage_deadband = 0.02    # pu
        self.max_response_time = 0.1    # seconds
        
        # Priority levels
        self.SAFETY_PRIORITY = 1
        self.GRID_SERVICES_PRIORITY = 2
        self.ECONOMIC_PRIORITY = 3
        
        # Performance tracking
        self._last_solve_time = 0.0
        self._solve_count = 0
    
    def solve(self, problem: OptimizationProblem) -> OptimizationResult:
        """Solve using ultra-fast rule-based dispatch."""
        start_time = time.time()
        
        try:
            # Extract current state
            current_state = self._extract_state(problem)
            
            # Apply dispatch rules in priority order
            control_signal = self._apply_dispatch_rules(current_state, problem)
            
            solve_time = time.time() - start_time
            self._last_solve_time = solve_time
            self._solve_count += 1
            
            # Create solution
            solution = {
                "battery_power_setpoint": control_signal.battery_power_setpoint,
                "renewable_curtailment": control_signal.renewable_curtailment,
                "load_shedding": control_signal.load_shedding,
                "frequency_response_active": control_signal.frequency_response_active,
                "voltage_support_active": control_signal.voltage_support_active,
                "confidence": control_signal.confidence,
                "method": "fast_dispatch_rules",
                "solve_time_ms": solve_time * 1000
            }
            
            return OptimizationResult(
                status=OptimizationStatus.SUCCESS,
                objective_value=0.0,  # Rule-based doesn't optimize objective
                solution=solution,
                solve_time=solve_time,
                metadata={
                    "method": "fast_dispatch_rules",
                    "priority_actions": control_signal.metadata.get("priority_actions", []),
                    "grid_frequency": current_state.grid_frequency,
                    "battery_soc": current_state.battery_soc
                }
            )
            
        except Exception as e:
            solve_time = time.time() - start_time
            self.logger.error(f"Fast dispatch rules failed: {e}")
            
            return OptimizationResult(
                status=OptimizationStatus.FAILED,
                objective_value=float('inf'),
                solution={},
                solve_time=solve_time,
                metadata={"error": str(e)}
            )
    
    def _extract_state(self, problem: OptimizationProblem) -> RealTimeState:
        """Extract current state from optimization problem."""
        params = problem.parameters
        
        return RealTimeState(
            timestamp=datetime.now(),
            grid_frequency=params.get("grid_frequency", 60.0),
            grid_voltage=params.get("grid_voltage", 1.0),
            total_load=params.get("total_load", 0.0),
            renewable_generation=params.get("renewable_generation", 0.0),
            battery_soc=params.get("battery_soc", 0.5),
            battery_power=params.get("battery_power", 0.0),
            electricity_price=params.get("electricity_price", 0.1),
            ancillary_service_signals=params.get("ancillary_service_signals", {}),
            grid_constraints=params.get("grid_constraints", {})
        )
    
    def _apply_dispatch_rules(self, state: RealTimeState, 
                            problem: OptimizationProblem) -> RealTimeControlSignal:
        """Apply dispatch rules in priority order."""
        
        # Initialize control signal
        control = RealTimeControlSignal(
            battery_power_setpoint=0.0,
            renewable_curtailment=0.0,
            load_shedding=0.0,
            frequency_response_active=False,
            voltage_support_active=False,
            metadata={"priority_actions": []}
        )
        
        # Battery parameters
        battery_capacity = problem.parameters.get("battery_capacity", 1000.0)
        max_battery_power = problem.parameters.get("max_battery_power", 250.0)
        
        # Priority 1: Safety and emergency response
        safety_action = self._apply_safety_rules(state, control, problem)
        if safety_action:
            control.metadata["priority_actions"].append(f"SAFETY: {safety_action}")
            return control  # Safety overrides everything
        
        # Priority 2: Grid services (frequency and voltage support)
        grid_service_action = self._apply_grid_service_rules(state, control, problem)
        if grid_service_action:
            control.metadata["priority_actions"].append(f"GRID_SERVICE: {grid_service_action}")
        
        # Priority 3: Economic dispatch (if no higher priority actions)
        if not control.frequency_response_active and not control.voltage_support_active:
            economic_action = self._apply_economic_rules(state, control, problem)
            if economic_action:
                control.metadata["priority_actions"].append(f"ECONOMIC: {economic_action}")
        
        # Ensure constraints are respected
        control = self._enforce_constraints(state, control, problem)
        
        return control
    
    def _apply_safety_rules(self, state: RealTimeState, 
                          control: RealTimeControlSignal,
                          problem: OptimizationProblem) -> Optional[str]:
        """Apply safety rules - highest priority."""
        
        # Emergency frequency response
        if abs(state.grid_frequency - 60.0) > 0.5:  # Major frequency deviation
            if state.grid_frequency < 59.5:  # Under-frequency
                # Reduce load, increase generation
                control.load_shedding = min(state.total_load * 0.2, 100.0)  # Shed 20% or 100kW max
                if state.battery_soc > 0.2:
                    control.battery_power_setpoint = -min(250.0, state.battery_soc * 1000.0 / 4)
                return f"emergency_under_frequency_{state.grid_frequency:.2f}Hz"
            
            elif state.grid_frequency > 60.5:  # Over-frequency
                # Reduce generation, increase load
                control.renewable_curtailment = state.renewable_generation * 0.3
                if state.battery_soc < 0.9:
                    control.battery_power_setpoint = min(250.0, (0.9 - state.battery_soc) * 1000.0 / 2)
                return f"emergency_over_frequency_{state.grid_frequency:.2f}Hz"
        
        # Battery safety limits
        if state.battery_soc < 0.05:  # Critical low SOC
            control.battery_power_setpoint = min(100.0, state.renewable_generation * 0.5)
            return f"critical_low_soc_{state.battery_soc:.3f}"
        
        if state.battery_soc > 0.98:  # Critical high SOC
            control.battery_power_setpoint = -min(100.0, state.total_load * 0.3)
            return f"critical_high_soc_{state.battery_soc:.3f}"
        
        # Voltage emergency
        if abs(state.grid_voltage - 1.0) > 0.1:  # Major voltage deviation
            if state.grid_voltage < 0.9:  # Low voltage
                control.voltage_support_active = True
                control.battery_power_setpoint = -min(150.0, state.battery_soc * 1000.0 / 6)
                return f"emergency_low_voltage_{state.grid_voltage:.3f}pu"
            elif state.grid_voltage > 1.1:  # High voltage
                control.voltage_support_active = True
                control.renewable_curtailment = state.renewable_generation * 0.4
                return f"emergency_high_voltage_{state.grid_voltage:.3f}pu"
        
        return None
    
    def _apply_grid_service_rules(self, state: RealTimeState, 
                                control: RealTimeControlSignal,
                                problem: OptimizationProblem) -> Optional[str]:
        """Apply grid service rules - second priority."""
        
        # Primary frequency response
        freq_deviation = state.grid_frequency - 60.0
        if abs(freq_deviation) > self.frequency_deadband:
            control.frequency_response_active = True
            
            # Calculate proportional response
            max_freq_response = problem.parameters.get("max_freq_response", 100.0)  # kW
            response_gain = problem.parameters.get("freq_response_gain", 20.0)  # kW/Hz
            
            freq_response_power = -freq_deviation * response_gain  # Negative feedback
            freq_response_power = np.clip(freq_response_power, -max_freq_response, max_freq_response)
            
            # Check if battery can provide the response
            if freq_response_power > 0:  # Need to charge (frequency high)
                if state.battery_soc < 0.9:
                    control.battery_power_setpoint += freq_response_power
                    return f"freq_response_charge_{freq_response_power:.1f}kW"
            else:  # Need to discharge (frequency low)
                if state.battery_soc > 0.1:
                    control.battery_power_setpoint += freq_response_power
                    return f"freq_response_discharge_{abs(freq_response_power):.1f}kW"
        
        # Voltage support
        voltage_deviation = state.grid_voltage - 1.0
        if abs(voltage_deviation) > self.voltage_deadband:
            control.voltage_support_active = True
            
            max_voltage_support = problem.parameters.get("max_voltage_support", 50.0)  # kW
            
            if voltage_deviation < -self.voltage_deadband:  # Low voltage - inject reactive power
                # Simplified: reduce active power to support voltage
                voltage_support_power = min(max_voltage_support, abs(voltage_deviation) * 100)
                control.battery_power_setpoint -= voltage_support_power
                return f"voltage_support_low_{voltage_support_power:.1f}kW"
            
            elif voltage_deviation > self.voltage_deadband:  # High voltage - absorb reactive power
                voltage_support_power = min(max_voltage_support, voltage_deviation * 100)
                control.battery_power_setpoint += voltage_support_power
                return f"voltage_support_high_{voltage_support_power:.1f}kW"
        
        # Ancillary service signals
        for service, signal in state.ancillary_service_signals.items():
            if abs(signal) > 0.01:  # Significant signal
                service_response = signal * problem.parameters.get(f"{service}_capacity", 50.0)
                control.battery_power_setpoint += service_response
                return f"ancillary_service_{service}_{service_response:.1f}kW"
        
        return None
    
    def _apply_economic_rules(self, state: RealTimeState, 
                            control: RealTimeControlSignal,
                            problem: OptimizationProblem) -> Optional[str]:
        """Apply economic dispatch rules - lowest priority."""
        
        # Simple arbitrage based on price and renewable generation
        price_threshold_high = problem.parameters.get("price_threshold_high", 0.15)  # $/kWh
        price_threshold_low = problem.parameters.get("price_threshold_low", 0.05)   # $/kWh
        
        # Net renewable generation (after load)
        net_renewable = state.renewable_generation - state.total_load
        
        if state.electricity_price > price_threshold_high:
            # High prices - discharge battery if possible
            if state.battery_soc > 0.3:  # Keep some reserve
                discharge_power = min(100.0, (state.battery_soc - 0.3) * 1000.0 / 2)
                control.battery_power_setpoint -= discharge_power
                return f"economic_discharge_{discharge_power:.1f}kW_price_{state.electricity_price:.3f}"
        
        elif state.electricity_price < price_threshold_low and net_renewable > 0:
            # Low prices and excess renewable - charge battery
            if state.battery_soc < 0.8:
                charge_power = min(net_renewable, 100.0, (0.8 - state.battery_soc) * 1000.0 / 3)
                control.battery_power_setpoint += charge_power
                return f"economic_charge_{charge_power:.1f}kW_price_{state.electricity_price:.3f}"
        
        # Renewable smoothing
        renewable_variability = problem.parameters.get("renewable_variability", 0.0)
        if renewable_variability > 50.0:  # High variability
            smoothing_power = min(50.0, renewable_variability * 0.3)
            if state.battery_soc > 0.2 and state.battery_soc < 0.8:
                control.battery_power_setpoint += smoothing_power * (0.5 - state.battery_soc)
                return f"renewable_smoothing_{smoothing_power:.1f}kW"
        
        return None
    
    def _enforce_constraints(self, state: RealTimeState, 
                           control: RealTimeControlSignal,
                           problem: OptimizationProblem) -> RealTimeControlSignal:
        """Enforce all system constraints."""
        
        # Battery power limits
        max_battery_power = problem.parameters.get("max_battery_power", 250.0)
        control.battery_power_setpoint = np.clip(
            control.battery_power_setpoint, 
            -max_battery_power, 
            max_battery_power
        )
        
        # Battery SOC limits for power
        if control.battery_power_setpoint > 0:  # Charging
            if state.battery_soc > 0.95:
                control.battery_power_setpoint = 0.0
            elif state.battery_soc > 0.8:
                # Reduce charging power as SOC approaches limit
                reduction_factor = (0.95 - state.battery_soc) / 0.15
                control.battery_power_setpoint *= reduction_factor
        
        else:  # Discharging
            if state.battery_soc < 0.05:
                control.battery_power_setpoint = 0.0
            elif state.battery_soc < 0.2:
                # Reduce discharging power as SOC approaches limit
                reduction_factor = (state.battery_soc - 0.05) / 0.15
                control.battery_power_setpoint *= reduction_factor
        
        # Grid constraints
        max_grid_injection = state.grid_constraints.get("max_injection", float('inf'))
        max_grid_consumption = state.grid_constraints.get("max_consumption", float('inf'))
        
        # Calculate net grid power
        net_grid_power = (state.total_load - state.renewable_generation + 
                         control.renewable_curtailment - control.load_shedding +
                         control.battery_power_setpoint)
        
        if net_grid_power > max_grid_consumption:
            # Reduce consumption
            excess = net_grid_power - max_grid_consumption
            if control.battery_power_setpoint > 0:
                control.battery_power_setpoint = max(0, control.battery_power_setpoint - excess)
        
        elif net_grid_power < -max_grid_injection:
            # Reduce injection
            excess = abs(net_grid_power) - max_grid_injection
            if control.battery_power_setpoint < 0:
                control.battery_power_setpoint = min(0, control.battery_power_setpoint + excess)
        
        # Update confidence based on constraint violations
        if (abs(control.battery_power_setpoint) < max_battery_power * 0.95 and
            state.battery_soc > 0.1 and state.battery_soc < 0.9):
            control.confidence = 1.0
        else:
            control.confidence = 0.8  # Reduced confidence near limits
        
        return control
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the fast dispatch rules."""
        return {
            "avg_solve_time_ms": self._last_solve_time * 1000,
            "total_solves": self._solve_count,
            "max_response_time_ms": self.max_response_time * 1000,
            "frequency_deadband_hz": self.frequency_deadband,
            "voltage_deadband_pu": self.voltage_deadband
        }


class ModelPredictiveControlPlugin(OptimizationPlugin):
    """Example expert plugin for Model Predictive Control."""
    
    def __init__(self, prediction_horizon: int = 10):
        super().__init__("mpc_realtime", "1.0")
        self.prediction_horizon = prediction_horizon
        self._solver_available = False
    
    def _initialize_impl(self, config: Dict[str, Any]) -> None:
        """Initialize MPC solver."""
        self.prediction_horizon = config.get("prediction_horizon", 10)
        
        try:
            import cvxpy as cp
            self._solver_available = True
            self.logger.info("MPC real-time plugin initialized with CVXPY")
        except ImportError:
            self.logger.warning("CVXPY not available, MPC plugin disabled")
            self._solver_available = False
    
    def is_available(self) -> bool:
        """Check if MPC solver is available."""
        return self._is_initialized and self._solver_available
    
    def validate_problem(self, problem: OptimizationProblem) -> bool:
        """Validate that problem can be solved with MPC."""
        return (self._solver_available and 
                "load_forecast" in problem.parameters and
                "price_forecast" in problem.parameters)
    
    def solve(self, problem: OptimizationProblem, timeout_ms: Optional[int] = None) -> OptimizationResult:
        """Solve using Model Predictive Control."""
        if not self.is_available():
            raise RuntimeError("MPC plugin not available")
        
        start_time = datetime.now()
        
        try:
            import cvxpy as cp
            
            # Extract forecasts
            load_forecast = problem.parameters.get("load_forecast", [])
            price_forecast = problem.parameters.get("price_forecast", [])
            renewable_forecast = problem.parameters.get("renewable_forecast", [])
            
            horizon = min(self.prediction_horizon, len(load_forecast), len(price_forecast))
            if horizon < 2:
                raise ValueError("Insufficient forecast data for MPC")
            
            # Decision variables
            battery_power = cp.Variable(horizon, name="battery_power")
            battery_soc = cp.Variable(horizon + 1, name="battery_soc")
            
            # Parameters
            current_soc = problem.parameters.get("battery_soc", 0.5)
            battery_capacity = problem.parameters.get("battery_capacity", 1000.0)
            max_power = problem.parameters.get("max_battery_power", 250.0)
            efficiency = problem.parameters.get("efficiency", 0.9)
            time_step = problem.parameters.get("time_step", 0.25)  # 15 minutes
            
            # Constraints
            constraints = []
            
            # Initial SOC
            constraints.append(battery_soc[0] == current_soc)
            
            # SOC dynamics
            for t in range(horizon):
                energy_change = battery_power[t] * time_step * efficiency / battery_capacity
                constraints.append(battery_soc[t+1] == battery_soc[t] + energy_change)
                
                # Limits
                constraints.append(battery_soc[t+1] >= 0.1)
                constraints.append(battery_soc[t+1] <= 0.9)
                constraints.append(battery_power[t] >= -max_power)
                constraints.append(battery_power[t] <= max_power)
            
            # Objective: minimize cost over prediction horizon
            cost = 0
            for t in range(horizon):
                # Grid power = load - renewable + battery_power
                grid_power = (load_forecast[t] - 
                            renewable_forecast[t] if t < len(renewable_forecast) else 0 +
                            battery_power[t])
                cost += grid_power * price_forecast[t] * time_step
            
            # Add terminal cost to encourage good final SOC
            terminal_soc_penalty = cp.square(battery_soc[horizon] - 0.5) * 100
            
            objective = cp.Minimize(cost + terminal_soc_penalty)
            
            # Solve
            problem_cvx = cp.Problem(objective, constraints)
            
            # Set solver options for speed
            solver_options = {"verbose": False}
            if timeout_ms:
                solver_options["max_iters"] = min(100, timeout_ms // 50)
            
            problem_cvx.solve(**solver_options)
            
            solve_time = (datetime.now() - start_time).total_seconds()
            
            if problem_cvx.status == cp.OPTIMAL:
                # Return only the first control action (MPC principle)
                solution = {
                    "battery_power_setpoint": float(battery_power.value[0]),
                    "renewable_curtailment": 0.0,
                    "load_shedding": 0.0,
                    "frequency_response_active": False,
                    "voltage_support_active": False,
                    "confidence": 0.95,
                    "method": "mpc_optimization",
                    "predicted_trajectory": {
                        "battery_power": battery_power.value.tolist(),
                        "battery_soc": battery_soc.value.tolist(),
                        "total_cost": problem_cvx.value
                    }
                }
                
                return OptimizationResult(
                    status=OptimizationStatus.SUCCESS,
                    objective_value=problem_cvx.value,
                    solution=solution,
                    solve_time=solve_time,
                    metadata={
                        "prediction_horizon": horizon,
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
            self.logger.error(f"MPC optimization failed: {e}")
            
            return OptimizationResult(
                status=OptimizationStatus.FAILED,
                objective_value=float('inf'),
                solution={},
                solve_time=solve_time,
                metadata={"error": str(e)}
            )


class RealTimeOptimizationManager:
    """Manager for real-time optimization with fast response capabilities."""
    
    def __init__(self):
        self.logger = logging.getLogger("realtime_optimization")
        self._performance_history = []
        self._max_history = 1000
    
    def create_realtime_problem(self, current_state: Dict[str, Any],
                              forecasts: Dict[str, List[float]] = None) -> OptimizationProblem:
        """Create a real-time optimization problem."""
        
        forecasts = forecasts or {}
        
        problem = OptimizationProblem(
            variables={
                "battery_power_setpoint": {"type": "continuous", "bounds": (-250, 250)},
                "renewable_curtailment": {"type": "continuous", "bounds": (0, 1000)},
                "load_shedding": {"type": "continuous", "bounds": (0, 1000)}
            },
            objectives=[{
                "name": "real_time_control",
                "type": "feasibility",
                "weight": 1.0
            }],
            constraints=[
                {"name": "power_balance", "type": "equality"},
                {"name": "battery_limits", "type": "inequality"},
                {"name": "grid_constraints", "type": "inequality"}
            ],
            parameters={
                **current_state,
                **forecasts,
                "time_step": 0.25,  # 15 minutes for MPC
            },
            time_horizon=4,  # 1 hour ahead for MPC
            time_step=0.25,
            metadata={
                "type": "realtime",
                "requires_fast_response": True,
                "max_solve_time_ms": 100
            }
        )
        
        return problem
    
    def record_performance(self, solve_time: float, method: str, success: bool) -> None:
        """Record performance metrics."""
        entry = {
            "timestamp": datetime.now(),
            "solve_time_ms": solve_time * 1000,
            "method": method,
            "success": success
        }
        
        self._performance_history.append(entry)
        
        if len(self._performance_history) > self._max_history:
            self._performance_history = self._performance_history[-self._max_history:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get real-time optimization performance statistics."""
        if not self._performance_history:
            return {}
        
        recent = self._performance_history[-100:]  # Last 100 solves
        
        solve_times = [h["solve_time_ms"] for h in recent]
        success_rate = sum(1 for h in recent if h["success"]) / len(recent)
        
        return {
            "avg_solve_time_ms": np.mean(solve_times),
            "max_solve_time_ms": np.max(solve_times),
            "min_solve_time_ms": np.min(solve_times),
            "p95_solve_time_ms": np.percentile(solve_times, 95),
            "success_rate": success_rate,
            "total_solves": len(self._performance_history),
            "methods_used": list(set(h["method"] for h in recent))
        }
