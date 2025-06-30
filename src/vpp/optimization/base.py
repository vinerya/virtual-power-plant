"""
Base classes and interfaces for the enhanced optimization framework.
Provides plugin architecture for expert models with rule-based fallbacks.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime, timedelta
import logging
import time
import numpy as np
from enum import Enum

from ..config import BaseConfig, ConfigValidationResult


class OptimizationStatus(Enum):
    """Status of optimization solution."""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    FALLBACK_USED = "fallback_used"


@dataclass
class OptimizationResult:
    """Result of an optimization run."""
    status: OptimizationStatus
    objective_value: float
    solution: Dict[str, Any]
    solve_time: float
    iterations: int = 0
    gap: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    fallback_used: bool = False
    solver_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationProblem:
    """Generic optimization problem definition."""
    variables: Dict[str, Any]
    objectives: List[Dict[str, Any]]
    constraints: List[Dict[str, Any]]
    parameters: Dict[str, Any] = field(default_factory=dict)
    time_horizon: int = 24  # hours
    time_step: float = 1.0  # hours
    metadata: Dict[str, Any] = field(default_factory=dict)


class OptimizationPlugin(ABC):
    """Base class for expert optimization models."""
    
    def __init__(self, name: str, version: str = "1.0"):
        self.name = name
        self.version = version
        self.logger = logging.getLogger(f"optimization.{name}")
        self._is_initialized = False
        self._last_solve_time = 0.0
        self._solve_count = 0
        self._success_count = 0
    
    @abstractmethod
    def solve(self, problem: OptimizationProblem, timeout_ms: Optional[int] = None) -> OptimizationResult:
        """Solve the optimization problem."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is ready to use."""
        pass
    
    @abstractmethod
    def validate_problem(self, problem: OptimizationProblem) -> bool:
        """Validate that the problem can be solved by this plugin."""
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return model information for logging/debugging."""
        return {
            "name": self.name,
            "version": self.version,
            "is_available": self.is_available(),
            "solve_count": self._solve_count,
            "success_rate": self._success_count / max(1, self._solve_count),
            "avg_solve_time": self._last_solve_time
        }
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin with configuration."""
        try:
            self._initialize_impl(config)
            self._is_initialized = True
            self.logger.info(f"Plugin {self.name} initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize plugin {self.name}: {e}")
            return False
    
    def _initialize_impl(self, config: Dict[str, Any]) -> None:
        """Implementation-specific initialization."""
        pass
    
    def _record_solve_attempt(self, success: bool, solve_time: float) -> None:
        """Record solve statistics."""
        self._solve_count += 1
        if success:
            self._success_count += 1
        self._last_solve_time = solve_time


class RuleBasedOptimizer(ABC):
    """Base class for rule-based optimization fallbacks."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"rules.{name}")
    
    @abstractmethod
    def solve(self, problem: OptimizationProblem) -> OptimizationResult:
        """Solve using rule-based approach."""
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return rule engine metadata."""
        return {
            "name": self.name,
            "type": "rule_based",
            "always_available": True
        }


class OptimizationEngine:
    """Main optimization engine with plugin support and fallbacks."""
    
    def __init__(self, name: str = "optimization_engine"):
        self.name = name
        self.logger = logging.getLogger(f"optimization.{name}")
        
        # Plugin management
        self._plugins: Dict[str, OptimizationPlugin] = {}
        self._fallback_rules: Dict[str, RuleBasedOptimizer] = {}
        self._active_plugin: Optional[str] = None
        
        # Performance monitoring
        self._solve_history: List[Dict[str, Any]] = []
        self._max_history = 1000
        
        # Configuration
        self._default_timeout_ms = 5000
        self._enable_fallback = True
        self._validate_solutions = True
    
    def register_plugin(self, plugin: OptimizationPlugin, config: Dict[str, Any] = None) -> bool:
        """Register an expert optimization plugin."""
        try:
            if config and not plugin.initialize(config):
                return False
            
            self._plugins[plugin.name] = plugin
            self.logger.info(f"Registered optimization plugin: {plugin.name}")
            
            # Set as active if it's the first available plugin
            if self._active_plugin is None and plugin.is_available():
                self._active_plugin = plugin.name
                self.logger.info(f"Set active plugin: {plugin.name}")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to register plugin {plugin.name}: {e}")
            return False
    
    def register_fallback(self, fallback: RuleBasedOptimizer, problem_type: str) -> None:
        """Register a rule-based fallback for a problem type."""
        self._fallback_rules[problem_type] = fallback
        self.logger.info(f"Registered fallback rules for {problem_type}: {fallback.name}")
    
    def set_active_plugin(self, plugin_name: str) -> bool:
        """Set the active optimization plugin."""
        if plugin_name not in self._plugins:
            self.logger.error(f"Plugin {plugin_name} not found")
            return False
        
        if not self._plugins[plugin_name].is_available():
            self.logger.error(f"Plugin {plugin_name} is not available")
            return False
        
        self._active_plugin = plugin_name
        self.logger.info(f"Set active plugin: {plugin_name}")
        return True
    
    def solve(self, problem: OptimizationProblem, 
              timeout_ms: Optional[int] = None,
              force_fallback: bool = False) -> OptimizationResult:
        """Solve optimization problem with plugin and fallback support."""
        start_time = time.time()
        timeout_ms = timeout_ms or self._default_timeout_ms
        
        # Try active plugin first (unless forced to use fallback)
        if not force_fallback and self._active_plugin:
            plugin = self._plugins[self._active_plugin]
            
            if plugin.is_available() and plugin.validate_problem(problem):
                try:
                    self.logger.debug(f"Solving with plugin: {plugin.name}")
                    
                    # Check timeout before calling plugin
                    elapsed_ms = (time.time() - start_time) * 1000
                    if elapsed_ms >= timeout_ms:
                        self.logger.warning(f"Timeout before plugin call: {elapsed_ms:.1f}ms >= {timeout_ms}ms")
                    else:
                        remaining_timeout = max(1, int(timeout_ms - elapsed_ms))
                        result = plugin.solve(problem, remaining_timeout)
                    
                    # Record solve attempt
                    solve_time = time.time() - start_time
                    plugin._record_solve_attempt(
                        result.status == OptimizationStatus.SUCCESS, 
                        solve_time
                    )
                    
                    # Validate solution if enabled
                    if self._validate_solutions and result.status == OptimizationStatus.SUCCESS:
                        if not self._validate_solution(problem, result):
                            self.logger.warning("Plugin solution failed validation, using fallback")
                        else:
                            self._record_solve_history(problem, result, plugin.name)
                            return result
                    elif result.status == OptimizationStatus.SUCCESS:
                        self._record_solve_history(problem, result, plugin.name)
                        return result
                    
                except Exception as e:
                    self.logger.warning(f"Plugin {plugin.name} failed: {e}")
        
        # Fall back to rule-based optimization
        if self._enable_fallback:
            problem_type = problem.metadata.get("type", "generic")
            if problem_type in self._fallback_rules:
                self.logger.info(f"Using fallback rules for {problem_type}")
                try:
                    result = self._fallback_rules[problem_type].solve(problem)
                    result.fallback_used = True
                    result.status = OptimizationStatus.FALLBACK_USED
                    
                    solve_time = time.time() - start_time
                    result.solve_time = solve_time
                    
                    self._record_solve_history(problem, result, "fallback")
                    return result
                    
                except Exception as e:
                    self.logger.error(f"Fallback rules failed: {e}")
        
        # Complete failure
        solve_time = time.time() - start_time
        result = OptimizationResult(
            status=OptimizationStatus.FAILED,
            objective_value=float('inf'),
            solution={},
            solve_time=solve_time,
            metadata={"error": "All optimization methods failed"}
        )
        
        self._record_solve_history(problem, result, "failed")
        return result
    
    def _validate_solution(self, problem: OptimizationProblem, result: OptimizationResult) -> bool:
        """Validate optimization solution."""
        try:
            # Basic sanity checks
            if not result.solution:
                return False
            
            if not np.isfinite(result.objective_value):
                return False
            
            # Problem-specific validation can be added here
            return True
            
        except Exception as e:
            self.logger.error(f"Solution validation failed: {e}")
            return False
    
    def _record_solve_history(self, problem: OptimizationProblem, 
                             result: OptimizationResult, solver: str) -> None:
        """Record solve history for analysis."""
        history_entry = {
            "timestamp": datetime.now(),
            "solver": solver,
            "status": result.status.value,
            "objective_value": result.objective_value,
            "solve_time": result.solve_time,
            "problem_type": problem.metadata.get("type", "unknown"),
            "fallback_used": result.fallback_used
        }
        
        self._solve_history.append(history_entry)
        
        # Limit history size
        if len(self._solve_history) > self._max_history:
            self._solve_history = self._solve_history[-self._max_history:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get optimization performance statistics."""
        if not self._solve_history:
            return {}
        
        recent_history = self._solve_history[-100:]  # Last 100 solves
        
        total_solves = len(recent_history)
        successful_solves = sum(1 for h in recent_history 
                               if h["status"] in ["success", "fallback_used"])
        fallback_usage = sum(1 for h in recent_history if h["fallback_used"])
        
        avg_solve_time = np.mean([h["solve_time"] for h in recent_history])
        
        return {
            "total_solves": total_solves,
            "success_rate": successful_solves / total_solves if total_solves > 0 else 0,
            "fallback_usage_rate": fallback_usage / total_solves if total_solves > 0 else 0,
            "avg_solve_time": avg_solve_time,
            "active_plugin": self._active_plugin,
            "available_plugins": [name for name, plugin in self._plugins.items() 
                                if plugin.is_available()],
            "registered_fallbacks": list(self._fallback_rules.keys())
        }
    
    def get_plugin_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered plugins."""
        return {name: plugin.get_metadata() 
                for name, plugin in self._plugins.items()}
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the optimization engine."""
        self._default_timeout_ms = config.get("default_timeout_ms", 5000)
        self._enable_fallback = config.get("enable_fallback", True)
        self._validate_solutions = config.get("validate_solutions", True)
        self._max_history = config.get("max_history", 1000)
        
        self.logger.info(f"Optimization engine configured: {config}")


class OptimizationFactory:
    """Factory for creating optimization engines with standard configurations."""
    
    @staticmethod
    def create_engine(engine_type: str = "standard") -> OptimizationEngine:
        """Create a pre-configured optimization engine."""
        engine = OptimizationEngine(f"{engine_type}_engine")
        
        if engine_type == "standard":
            # Register standard fallbacks
            from .stochastic import SimpleStochasticRules
            from .realtime import FastDispatchRules
            from .distributed import SimpleConsensusRules
            
            engine.register_fallback(SimpleStochasticRules(), "stochastic")
            engine.register_fallback(FastDispatchRules(), "realtime")
            engine.register_fallback(SimpleConsensusRules(), "distributed")
        
        return engine
    
    @staticmethod
    def create_problem(problem_type: str, **kwargs) -> OptimizationProblem:
        """Create a standard optimization problem."""
        return OptimizationProblem(
            variables=kwargs.get("variables", {}),
            objectives=kwargs.get("objectives", []),
            constraints=kwargs.get("constraints", []),
            parameters=kwargs.get("parameters", {}),
            time_horizon=kwargs.get("time_horizon", 24),
            time_step=kwargs.get("time_step", 1.0),
            metadata={"type": problem_type, **kwargs.get("metadata", {})}
        )
