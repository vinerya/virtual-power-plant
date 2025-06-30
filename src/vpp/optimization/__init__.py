"""
Enhanced optimization framework for Virtual Power Plant management.

This package provides a comprehensive optimization framework with:
- Plugin architecture for expert models
- Rule-based fallbacks for robustness
- Stochastic optimization with uncertainty handling
- Real-time optimization for grid services
- Distributed optimization for multi-site coordination

The framework is designed to be production-ready with solid fallbacks
while providing clean interfaces for experts to integrate advanced models.
"""

from .base import (
    OptimizationStatus,
    OptimizationResult,
    OptimizationProblem,
    OptimizationPlugin,
    RuleBasedOptimizer,
    OptimizationEngine,
    OptimizationFactory
)

from .stochastic import (
    Scenario,
    ScenarioSet,
    ScenarioGenerator,
    SimpleStochasticRules,
    CVaRStochasticPlugin,
    StochasticOptimizationManager
)

from .realtime import (
    RealTimeState,
    RealTimeControlSignal,
    FastDispatchRules,
    ModelPredictiveControlPlugin,
    RealTimeOptimizationManager
)

from .distributed import (
    SiteState,
    CoordinationSignal,
    SimpleConsensusRules,
    ADMMDistributedPlugin,
    DistributedOptimizationManager
)

# Version information
__version__ = "1.0.0"
__author__ = "VPP Development Team"

# Export all public classes and functions
__all__ = [
    # Base framework
    "OptimizationStatus",
    "OptimizationResult", 
    "OptimizationProblem",
    "OptimizationPlugin",
    "RuleBasedOptimizer",
    "OptimizationEngine",
    "OptimizationFactory",
    
    # Stochastic optimization
    "Scenario",
    "ScenarioSet", 
    "ScenarioGenerator",
    "SimpleStochasticRules",
    "CVaRStochasticPlugin",
    "StochasticOptimizationManager",
    
    # Real-time optimization
    "RealTimeState",
    "RealTimeControlSignal",
    "FastDispatchRules",
    "ModelPredictiveControlPlugin", 
    "RealTimeOptimizationManager",
    
    # Distributed optimization
    "SiteState",
    "CoordinationSignal",
    "SimpleConsensusRules",
    "ADMMDistributedPlugin",
    "DistributedOptimizationManager",
]


def create_optimization_engine(engine_type: str = "standard") -> OptimizationEngine:
    """
    Create a pre-configured optimization engine.
    
    Args:
        engine_type: Type of engine to create ("standard", "research", "production")
        
    Returns:
        Configured OptimizationEngine instance
        
    Example:
        >>> engine = create_optimization_engine("standard")
        >>> engine.register_plugin(CVaRStochasticPlugin())
        >>> problem = OptimizationFactory.create_problem("stochastic")
        >>> result = engine.solve(problem)
    """
    return OptimizationFactory.create_engine(engine_type)


def create_stochastic_problem(base_data: dict, num_scenarios: int = 10, uncertainty_config: dict = None) -> OptimizationProblem:
    """
    Create a stochastic optimization problem with scenario generation.
    
    Args:
        base_data: Base forecast data (prices, load, renewable generation)
        num_scenarios: Number of scenarios to generate
        uncertainty_config: Configuration for uncertainty parameters
        
    Returns:
        OptimizationProblem configured for stochastic optimization
        
    Example:
        >>> base_data = {
        ...     "base_prices": [0.1, 0.15, 0.12, 0.08],
        ...     "renewable_forecast": [100, 150, 200, 120],
        ...     "battery_capacity": 1000.0
        ... }
        >>> problem = create_stochastic_problem(base_data, num_scenarios=20)
    """
    manager = StochasticOptimizationManager()
    return manager.create_stochastic_problem(base_data, num_scenarios, uncertainty_config)


def create_realtime_problem(current_state: dict, forecasts: dict = None) -> OptimizationProblem:
    """
    Create a real-time optimization problem for fast grid services.
    
    Args:
        current_state: Current system state (frequency, voltage, SOC, etc.)
        forecasts: Short-term forecasts for MPC (optional)
        
    Returns:
        OptimizationProblem configured for real-time optimization
        
    Example:
        >>> current_state = {
        ...     "grid_frequency": 59.95,
        ...     "battery_soc": 0.6,
        ...     "total_load": 500.0,
        ...     "renewable_generation": 300.0
        ... }
        >>> problem = create_realtime_problem(current_state)
    """
    manager = RealTimeOptimizationManager()
    return manager.create_realtime_problem(current_state, forecasts)


def create_distributed_problem(sites_data: list, coordination_targets: dict = None) -> OptimizationProblem:
    """
    Create a distributed optimization problem for multi-site coordination.
    
    Args:
        sites_data: List of site information dictionaries
        coordination_targets: Target power and reserve allocations
        
    Returns:
        OptimizationProblem configured for distributed optimization
        
    Example:
        >>> sites_data = [
        ...     {"site_id": "site1", "total_capacity": 500, "marginal_cost": 0.08},
        ...     {"site_id": "site2", "total_capacity": 300, "marginal_cost": 0.12}
        ... ]
        >>> targets = {"total_power": 600, "reserve": 100}
        >>> problem = create_distributed_problem(sites_data, targets)
    """
    manager = DistributedOptimizationManager()
    return manager.create_distributed_problem(sites_data, coordination_targets)


# Convenience functions for common use cases
def solve_with_fallback(problem: OptimizationProblem, 
                       plugin: OptimizationPlugin = None,
                       timeout_ms: int = 5000) -> OptimizationResult:
    """
    Solve optimization problem with automatic fallback to rules.
    
    Args:
        problem: Optimization problem to solve
        plugin: Expert plugin to try first (optional)
        timeout_ms: Maximum solve time in milliseconds
        
    Returns:
        OptimizationResult with solution or fallback result
        
    Example:
        >>> problem = create_stochastic_problem(base_data)
        >>> plugin = CVaRStochasticPlugin(risk_level=0.05)
        >>> result = solve_with_fallback(problem, plugin, timeout_ms=1000)
    """
    engine = create_optimization_engine("standard")
    
    if plugin:
        engine.register_plugin(plugin)
    
    return engine.solve(problem, timeout_ms=timeout_ms)


def validate_optimization_config(config: dict) -> list:
    """
    Validate optimization configuration parameters.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        List of validation error messages (empty if valid)
        
    Example:
        >>> config = {"strategy": "stochastic", "num_scenarios": 10}
        >>> errors = validate_optimization_config(config)
        >>> if not errors:
        ...     print("Configuration is valid")
    """
    errors = []
    
    # Validate strategy
    valid_strategies = ["stochastic", "realtime", "distributed", "deterministic"]
    strategy = config.get("strategy")
    if strategy and strategy not in valid_strategies:
        errors.append(f"Invalid strategy '{strategy}'. Must be one of {valid_strategies}")
    
    # Validate numeric parameters
    numeric_params = {
        "num_scenarios": (1, 1000),
        "timeout_ms": (1, 300000),
        "time_horizon": (0.1, 168),  # 6 minutes to 1 week
        "time_step": (0.01, 24)      # 36 seconds to 1 day
    }
    
    for param, (min_val, max_val) in numeric_params.items():
        value = config.get(param)
        if value is not None:
            if not isinstance(value, (int, float)):
                errors.append(f"Parameter '{param}' must be numeric")
            elif not (min_val <= value <= max_val):
                errors.append(f"Parameter '{param}' must be between {min_val} and {max_val}")
    
    # Validate plugin configurations
    plugins_config = config.get("plugins", {})
    for plugin_name, plugin_config in plugins_config.items():
        if not isinstance(plugin_config, dict):
            errors.append(f"Plugin '{plugin_name}' configuration must be a dictionary")
    
    return errors


# Module-level configuration
_default_config = {
    "default_timeout_ms": 5000,
    "enable_fallback": True,
    "validate_solutions": True,
    "log_performance": True
}


def configure_optimization(config: dict) -> None:
    """
    Configure global optimization settings.
    
    Args:
        config: Configuration dictionary
        
    Example:
        >>> configure_optimization({
        ...     "default_timeout_ms": 10000,
        ...     "enable_fallback": True,
        ...     "validate_solutions": True
        ... })
    """
    global _default_config
    _default_config.update(config)


def get_optimization_config() -> dict:
    """
    Get current global optimization configuration.
    
    Returns:
        Current configuration dictionary
    """
    return _default_config.copy()


# Performance monitoring utilities
def benchmark_optimization_methods(problem: OptimizationProblem, 
                                 methods: list = None,
                                 num_runs: int = 10) -> dict:
    """
    Benchmark different optimization methods on a problem.
    
    Args:
        problem: Problem to benchmark
        methods: List of method names to test (default: all available)
        num_runs: Number of runs per method for averaging
        
    Returns:
        Dictionary with performance statistics for each method
        
    Example:
        >>> problem = create_stochastic_problem(base_data)
        >>> stats = benchmark_optimization_methods(problem, num_runs=5)
        >>> print(f"Average solve time: {stats['cvar']['avg_solve_time']:.3f}s")
    """
    import time
    
    if methods is None:
        methods = ["rules", "cvar", "mpc"]
    
    results = {}
    
    for method in methods:
        method_results = {
            "solve_times": [],
            "objective_values": [],
            "success_count": 0
        }
        
        for run in range(num_runs):
            engine = create_optimization_engine("standard")
            
            # Configure method-specific plugin
            if method == "cvar":
                plugin = CVaRStochasticPlugin()
                engine.register_plugin(plugin)
            elif method == "mpc":
                plugin = ModelPredictiveControlPlugin()
                engine.register_plugin(plugin)
            # "rules" uses fallback only
            
            start_time = time.time()
            result = engine.solve(problem, force_fallback=(method == "rules"))
            solve_time = time.time() - start_time
            
            method_results["solve_times"].append(solve_time)
            method_results["objective_values"].append(result.objective_value)
            
            if result.status in [OptimizationStatus.SUCCESS, OptimizationStatus.FALLBACK_USED]:
                method_results["success_count"] += 1
        
        # Calculate statistics
        results[method] = {
            "avg_solve_time": sum(method_results["solve_times"]) / num_runs,
            "min_solve_time": min(method_results["solve_times"]),
            "max_solve_time": max(method_results["solve_times"]),
            "avg_objective": sum(method_results["objective_values"]) / num_runs,
            "success_rate": method_results["success_count"] / num_runs
        }
    
    return results


# Documentation and examples
def print_optimization_examples():
    """Print usage examples for the optimization framework."""
    
    examples = """
    VPP Optimization Framework - Usage Examples
    ==========================================
    
    1. Basic Stochastic Optimization:
    
        from vpp.optimization import create_stochastic_problem, CVaRStochasticPlugin
        
        base_data = {
            "base_prices": [0.1, 0.15, 0.12, 0.08] * 6,  # 24 hours
            "renewable_forecast": [100, 150, 200, 120] * 6,
            "battery_capacity": 1000.0,
            "max_power": 250.0
        }
        
        problem = create_stochastic_problem(base_data, num_scenarios=20)
        plugin = CVaRStochasticPlugin(risk_level=0.05)
        result = solve_with_fallback(problem, plugin)
        
        print(f"Optimal cost: ${result.objective_value:.2f}")
        print(f"Battery schedule: {result.solution['battery_power']}")
    
    2. Real-time Grid Services:
    
        from vpp.optimization import create_realtime_problem, FastDispatchRules
        
        current_state = {
            "grid_frequency": 59.95,  # Under-frequency event
            "battery_soc": 0.6,
            "total_load": 500.0,
            "renewable_generation": 300.0,
            "electricity_price": 0.12
        }
        
        problem = create_realtime_problem(current_state)
        result = solve_with_fallback(problem, timeout_ms=100)  # Fast response
        
        print(f"Battery setpoint: {result.solution['battery_power_setpoint']:.1f} kW")
        print(f"Frequency response: {result.solution['frequency_response_active']}")
    
    3. Multi-site Coordination:
    
        from vpp.optimization import create_distributed_problem, ADMMDistributedPlugin
        
        sites_data = [
            {
                "site_id": "wind_farm_1",
                "total_capacity": 500,
                "available_capacity": 400,
                "marginal_cost": 0.08,
                "location": "Texas"
            },
            {
                "site_id": "solar_storage_1", 
                "total_capacity": 300,
                "available_capacity": 250,
                "marginal_cost": 0.12,
                "location": "California"
            }
        ]
        
        targets = {"total_power": 600, "reserve": 100}
        problem = create_distributed_problem(sites_data, targets)
        
        plugin = ADMMDistributedPlugin(rho=1.0, max_iterations=50)
        result = solve_with_fallback(problem, plugin)
        
        print("Site allocations:")
        for site_id, power in result.solution['target_power'].items():
            print(f"  {site_id}: {power:.1f} kW")
    
    4. Performance Benchmarking:
    
        from vpp.optimization import benchmark_optimization_methods
        
        problem = create_stochastic_problem(base_data)
        stats = benchmark_optimization_methods(problem, num_runs=10)
        
        for method, metrics in stats.items():
            print(f"{method}: {metrics['avg_solve_time']:.3f}s avg, "
                  f"{metrics['success_rate']:.1%} success rate")
    
    5. Custom Expert Plugin:
    
        from vpp.optimization import OptimizationPlugin, OptimizationResult, OptimizationStatus
        
        class MyCustomPlugin(OptimizationPlugin):
            def __init__(self):
                super().__init__("my_custom_optimizer", "1.0")
            
            def is_available(self):
                return True  # Always available
            
            def validate_problem(self, problem):
                return True  # Can solve any problem
            
            def solve(self, problem, timeout_ms=None):
                # Your custom optimization logic here
                solution = {"battery_power": [0] * 24}
                
                return OptimizationResult(
                    status=OptimizationStatus.SUCCESS,
                    objective_value=100.0,
                    solution=solution,
                    solve_time=0.1
                )
        
        # Use your custom plugin
        engine = create_optimization_engine()
        engine.register_plugin(MyCustomPlugin())
        result = engine.solve(problem)
    """
    
    print(examples)


if __name__ == "__main__":
    print_optimization_examples()
