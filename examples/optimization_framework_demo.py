"""
Comprehensive demonstration of the enhanced optimization framework.

This example showcases:
- Stochastic optimization with uncertainty handling
- Real-time optimization for grid services
- Distributed optimization for multi-site coordination
- Plugin architecture with expert models and rule-based fallbacks
- Performance benchmarking and comparison
"""

import sys
import os
from pathlib import Path
import logging
import numpy as np
from datetime import datetime, timedelta
import time

# Add the src directory to the path so we can import vpp modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vpp.optimization import (
    # Core framework
    create_optimization_engine, OptimizationFactory,
    
    # Problem creation
    create_stochastic_problem, create_realtime_problem, create_distributed_problem,
    
    # Plugins and rules
    CVaRStochasticPlugin, SimpleStochasticRules,
    ModelPredictiveControlPlugin, FastDispatchRules,
    ADMMDistributedPlugin, SimpleConsensusRules,
    
    # Utilities
    solve_with_fallback, benchmark_optimization_methods,
    validate_optimization_config
)


def setup_logging():
    """Setup logging for the demonstration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def demonstrate_stochastic_optimization():
    """Demonstrate stochastic optimization with uncertainty handling."""
    print("=" * 70)
    print("STOCHASTIC OPTIMIZATION DEMONSTRATION")
    print("=" * 70)
    
    # Create base forecast data
    base_data = {
        "base_prices": [0.08, 0.09, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22,  # Morning
                       0.25, 0.28, 0.30, 0.28, 0.25, 0.22, 0.20, 0.18,  # Afternoon
                       0.22, 0.25, 0.28, 0.30, 0.25, 0.20, 0.15, 0.10], # Evening/Night
        "renewable_forecast": [0, 0, 0, 0, 50, 150, 300, 450,  # Solar ramp-up
                              600, 750, 800, 850, 800, 750, 600, 450,  # Peak solar
                              300, 150, 50, 0, 0, 0, 0, 0],  # Solar ramp-down
        "load_forecast": [400, 380, 360, 350, 370, 420, 480, 550,  # Morning load
                         600, 650, 680, 700, 720, 740, 760, 780,  # Day load
                         800, 820, 780, 720, 650, 580, 500, 450], # Evening load
        "battery_capacity": 2000.0,  # kWh
        "max_power": 500.0,          # kW
        "efficiency": 0.92
    }
    
    print("Creating stochastic optimization problem...")
    print(f"  Base case: 24-hour horizon with price volatility")
    print(f"  Battery: {base_data['battery_capacity']} kWh, {base_data['max_power']} kW")
    print(f"  Renewable: {max(base_data['renewable_forecast'])} kW peak solar")
    
    # Create stochastic problem with scenarios
    problem = create_stochastic_problem(
        base_data, 
        num_scenarios=20,
        uncertainty_config={
            "price_volatility": 0.25,      # 25% price volatility
            "renewable_error": 0.20,       # 20% forecast error
            "load_uncertainty": 0.10       # 10% load uncertainty
        }
    )
    
    print(f"  Generated {len(problem.parameters['scenarios'])} scenarios")
    
    # Test 1: Rule-based fallback (always available)
    print("\n1. Testing rule-based stochastic optimization (fallback)...")
    start_time = time.time()
    result_rules = solve_with_fallback(problem, timeout_ms=1000)
    rules_time = time.time() - start_time
    
    print(f"   Status: {result_rules.status.value}")
    print(f"   Solve time: {rules_time:.3f} seconds")
    print(f"   Objective value: ${result_rules.objective_value:.2f}")
    print(f"   Method: {result_rules.solution.get('method', 'unknown')}")
    
    # Test 2: CVaR optimization (if CVXPY available)
    print("\n2. Testing CVaR stochastic optimization (expert plugin)...")
    try:
        cvar_plugin = CVaRStochasticPlugin(risk_level=0.05)
        start_time = time.time()
        result_cvar = solve_with_fallback(problem, cvar_plugin, timeout_ms=5000)
        cvar_time = time.time() - start_time
        
        print(f"   Status: {result_cvar.status.value}")
        print(f"   Solve time: {cvar_time:.3f} seconds")
        print(f"   Objective value: ${result_cvar.objective_value:.2f}")
        print(f"   Method: {result_cvar.solution.get('method', 'unknown')}")
        
        if result_cvar.status.value == "success":
            print(f"   CVaR (5% risk): ${result_cvar.solution.get('cvar', 0):.2f}")
            print(f"   Expected cost: ${result_cvar.solution.get('expected_cost', 0):.2f}")
        
        # Compare results
        if result_rules.status.value in ["success", "fallback_used"] and result_cvar.status.value == "success":
            improvement = ((result_rules.objective_value - result_cvar.objective_value) / 
                          result_rules.objective_value * 100)
            print(f"   CVaR improvement over rules: {improvement:.1f}%")
    
    except Exception as e:
        print(f"   CVaR plugin failed: {e}")
        print("   (This is expected if CVXPY is not installed)")
    
    # Show battery schedule from best result
    best_result = result_cvar if 'result_cvar' in locals() and result_cvar.status.value == "success" else result_rules
    if "battery_power" in best_result.solution:
        battery_schedule = best_result.solution["battery_power"]
        print(f"\n3. Optimal battery schedule (first 8 hours):")
        for i in range(min(8, len(battery_schedule))):
            power = battery_schedule[i]
            action = "Charging" if power > 0 else "Discharging" if power < 0 else "Idle"
            print(f"   Hour {i:2d}: {power:6.1f} kW ({action})")


def demonstrate_realtime_optimization():
    """Demonstrate real-time optimization for grid services."""
    print("\n" + "=" * 70)
    print("REAL-TIME OPTIMIZATION DEMONSTRATION")
    print("=" * 70)
    
    # Simulate different grid conditions
    scenarios = [
        {
            "name": "Normal Operation",
            "state": {
                "grid_frequency": 60.00,
                "grid_voltage": 1.00,
                "battery_soc": 0.6,
                "total_load": 500.0,
                "renewable_generation": 300.0,
                "electricity_price": 0.12
            }
        },
        {
            "name": "Under-frequency Event",
            "state": {
                "grid_frequency": 59.85,  # Low frequency - need more generation
                "grid_voltage": 0.98,
                "battery_soc": 0.7,
                "total_load": 600.0,
                "renewable_generation": 200.0,
                "electricity_price": 0.25
            }
        },
        {
            "name": "Over-frequency Event", 
            "state": {
                "grid_frequency": 60.15,  # High frequency - need less generation
                "grid_voltage": 1.02,
                "battery_soc": 0.4,
                "total_load": 400.0,
                "renewable_generation": 500.0,
                "electricity_price": 0.08
            }
        },
        {
            "name": "High Price Period",
            "state": {
                "grid_frequency": 59.98,
                "grid_voltage": 1.01,
                "battery_soc": 0.8,
                "total_load": 700.0,
                "renewable_generation": 100.0,
                "electricity_price": 0.35  # Very high price
            }
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}:")
        state = scenario['state']
        
        print(f"   Grid frequency: {state['grid_frequency']:.2f} Hz")
        print(f"   Battery SOC: {state['battery_soc']:.1%}")
        print(f"   Load: {state['total_load']:.0f} kW")
        print(f"   Renewable: {state['renewable_generation']:.0f} kW")
        print(f"   Price: ${state['electricity_price']:.3f}/kWh")
        
        # Create real-time problem
        problem = create_realtime_problem(state)
        
        # Test fast dispatch rules (always available, sub-second response)
        start_time = time.time()
        result = solve_with_fallback(problem, timeout_ms=100)
        solve_time = time.time() - start_time
        
        print(f"   → Solve time: {solve_time*1000:.1f} ms")
        print(f"   → Status: {result.status.value}")
        
        if result.status.value in ["success", "fallback_used"]:
            solution = result.solution
            print(f"   → Battery setpoint: {solution.get('battery_power_setpoint', 0):.1f} kW")
            print(f"   → Frequency response: {solution.get('frequency_response_active', False)}")
            print(f"   → Voltage support: {solution.get('voltage_support_active', False)}")
            print(f"   → Load shedding: {solution.get('load_shedding', 0):.1f} kW")
            print(f"   → Renewable curtailment: {solution.get('renewable_curtailment', 0):.1f} kW")
            print(f"   → Confidence: {solution.get('confidence', 0):.1%}")
            
            # Show priority actions taken
            priority_actions = result.metadata.get("priority_actions", [])
            if priority_actions:
                print(f"   → Actions: {', '.join(priority_actions)}")


def demonstrate_distributed_optimization():
    """Demonstrate distributed optimization for multi-site coordination."""
    print("\n" + "=" * 70)
    print("DISTRIBUTED OPTIMIZATION DEMONSTRATION")
    print("=" * 70)
    
    # Create multi-site VPP data
    sites_data = [
        {
            "site_id": "wind_farm_texas",
            "location": "Texas",
            "total_capacity": 800.0,
            "available_capacity": 650.0,
            "current_generation": 150.0,
            "current_load": 50.0,
            "battery_soc": 0.4,
            "battery_capacity": 500.0,
            "marginal_cost": 0.06,  # Low cost wind
            "constraints": {"ramp_rate": 100.0}
        },
        {
            "site_id": "solar_storage_california",
            "location": "California", 
            "total_capacity": 600.0,
            "available_capacity": 400.0,
            "current_generation": 200.0,
            "current_load": 100.0,
            "battery_soc": 0.7,
            "battery_capacity": 1000.0,
            "marginal_cost": 0.08,  # Medium cost solar+storage
            "constraints": {"ramp_rate": 150.0}
        },
        {
            "site_id": "hydro_washington",
            "location": "Washington",
            "total_capacity": 400.0,
            "available_capacity": 350.0,
            "current_generation": 50.0,
            "current_load": 30.0,
            "battery_soc": 0.0,  # No battery
            "battery_capacity": 0.0,
            "marginal_cost": 0.04,  # Very low cost hydro
            "constraints": {"ramp_rate": 200.0}
        },
        {
            "site_id": "gas_peaker_nevada",
            "location": "Nevada",
            "total_capacity": 300.0,
            "available_capacity": 300.0,
            "current_generation": 0.0,
            "current_load": 20.0,
            "battery_soc": 0.0,  # No battery
            "battery_capacity": 0.0,
            "marginal_cost": 0.15,  # High cost gas peaker
            "constraints": {"ramp_rate": 50.0}
        }
    ]
    
    print("Multi-site VPP portfolio:")
    total_capacity = sum(site["total_capacity"] for site in sites_data)
    total_available = sum(site["available_capacity"] for site in sites_data)
    
    for site in sites_data:
        print(f"  {site['site_id']}:")
        print(f"    Location: {site['location']}")
        print(f"    Capacity: {site['available_capacity']:.0f}/{site['total_capacity']:.0f} kW available")
        print(f"    Cost: ${site['marginal_cost']:.3f}/kWh")
        if site['battery_capacity'] > 0:
            print(f"    Battery: {site['battery_capacity']:.0f} kWh at {site['battery_soc']:.1%} SOC")
    
    print(f"\nTotal portfolio: {total_available:.0f}/{total_capacity:.0f} kW available")
    
    # Test different coordination scenarios
    coordination_scenarios = [
        {
            "name": "Peak Demand Response",
            "targets": {"total_power": 800.0, "reserve": 200.0},
            "description": "High demand period requiring 800 kW generation"
        },
        {
            "name": "Load Following",
            "targets": {"total_power": 400.0, "reserve": 150.0},
            "description": "Moderate load following with reserve requirements"
        },
        {
            "name": "Excess Renewable",
            "targets": {"total_power": -200.0, "reserve": 100.0},
            "description": "Absorb excess renewable generation"
        }
    ]
    
    for i, scenario in enumerate(coordination_scenarios, 1):
        print(f"\n{i}. {scenario['name']} - {scenario['description']}")
        targets = scenario['targets']
        print(f"   Target power: {targets['total_power']:+.0f} kW")
        print(f"   Target reserve: {targets['reserve']:.0f} kW")
        
        # Create distributed problem
        problem = create_distributed_problem(sites_data, targets)
        
        # Test 1: Simple consensus rules (always available)
        print(f"\n   a) Simple consensus coordination:")
        start_time = time.time()
        result_rules = solve_with_fallback(problem, timeout_ms=1000)
        rules_time = time.time() - start_time
        
        print(f"      Status: {result_rules.status.value}")
        print(f"      Solve time: {rules_time:.3f} seconds")
        print(f"      Total cost: ${result_rules.objective_value:.2f}")
        
        if result_rules.status.value in ["success", "fallback_used"]:
            target_power = result_rules.solution["target_power"]
            reserve_allocations = result_rules.solution["reserve_allocations"]
            
            print(f"      Site allocations:")
            total_allocated_power = 0
            total_allocated_reserve = 0
            
            for site in sites_data:
                site_id = site["site_id"]
                power = target_power.get(site_id, 0)
                reserve = reserve_allocations.get(site_id, 0)
                total_allocated_power += power
                total_allocated_reserve += reserve
                
                print(f"        {site_id}: {power:+6.1f} kW, {reserve:4.1f} kW reserve")
            
            print(f"      Total allocated: {total_allocated_power:+.1f} kW power, {total_allocated_reserve:.1f} kW reserve")
            
            # Check if targets were met
            power_error = abs(total_allocated_power - targets["total_power"])
            reserve_error = abs(total_allocated_reserve - targets["reserve"])
            print(f"      Target accuracy: ±{power_error:.1f} kW power, ±{reserve_error:.1f} kW reserve")
        
        # Test 2: ADMM distributed optimization (if CVXPY available)
        print(f"\n   b) ADMM distributed optimization:")
        try:
            admm_plugin = ADMMDistributedPlugin(rho=1.0, max_iterations=50)
            start_time = time.time()
            result_admm = solve_with_fallback(problem, admm_plugin, timeout_ms=10000)
            admm_time = time.time() - start_time
            
            print(f"      Status: {result_admm.status.value}")
            print(f"      Solve time: {admm_time:.3f} seconds")
            print(f"      Iterations: {result_admm.metadata.get('iterations', 'N/A')}")
            print(f"      Converged: {result_admm.metadata.get('converged', False)}")
            
            if result_admm.status.value == "success":
                print(f"      Total cost: ${result_admm.objective_value:.2f}")
                
                # Compare with rules-based approach
                if result_rules.objective_value != float('inf'):
                    improvement = ((result_rules.objective_value - result_admm.objective_value) / 
                                  result_rules.objective_value * 100)
                    print(f"      Cost improvement over rules: {improvement:.1f}%")
        
        except Exception as e:
            print(f"      ADMM plugin failed: {e}")
            print("      (This is expected if CVXPY is not installed)")


def demonstrate_performance_benchmarking():
    """Demonstrate performance benchmarking capabilities."""
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARKING DEMONSTRATION")
    print("=" * 70)
    
    # Create a representative problem for benchmarking
    base_data = {
        "base_prices": [0.1, 0.15, 0.12, 0.08] * 6,  # 24 hours
        "renewable_forecast": [100, 150, 200, 120] * 6,
        "battery_capacity": 1000.0,
        "max_power": 250.0
    }
    
    problem = create_stochastic_problem(base_data, num_scenarios=10)
    
    print("Benchmarking optimization methods...")
    print(f"Problem: {problem.metadata['type']} with {len(problem.parameters['scenarios'])} scenarios")
    print(f"Time horizon: {problem.time_horizon} hours")
    
    # Benchmark different methods
    methods = ["rules"]  # Always available
    
    # Add advanced methods if available
    try:
        import cvxpy as cp
        methods.extend(["cvar", "mpc"])
        print("CVXPY available - testing advanced methods")
    except ImportError:
        print("CVXPY not available - testing rules only")
    
    print(f"\nRunning benchmark with {len(methods)} methods, 5 runs each...")
    
    try:
        stats = benchmark_optimization_methods(problem, methods=methods, num_runs=5)
        
        print(f"\nBenchmark Results:")
        print(f"{'Method':<15} {'Avg Time':<10} {'Success Rate':<12} {'Avg Cost':<10}")
        print("-" * 50)
        
        for method, metrics in stats.items():
            print(f"{method:<15} {metrics['avg_solve_time']:.3f}s    "
                  f"{metrics['success_rate']:.1%}        "
                  f"${metrics['avg_objective']:.2f}")
        
        # Find best performing method
        successful_methods = {m: s for m, s in stats.items() if s['success_rate'] > 0}
        if successful_methods:
            best_method = min(successful_methods.keys(), 
                            key=lambda m: successful_methods[m]['avg_objective'])
            print(f"\nBest performing method: {best_method}")
            print(f"Average cost: ${successful_methods[best_method]['avg_objective']:.2f}")
            print(f"Average solve time: {successful_methods[best_method]['avg_solve_time']:.3f}s")
    
    except Exception as e:
        print(f"Benchmarking failed: {e}")


def demonstrate_configuration_validation():
    """Demonstrate configuration validation capabilities."""
    print("\n" + "=" * 70)
    print("CONFIGURATION VALIDATION DEMONSTRATION")
    print("=" * 70)
    
    # Test various configurations
    test_configs = [
        {
            "name": "Valid Configuration",
            "config": {
                "strategy": "stochastic",
                "num_scenarios": 20,
                "timeout_ms": 5000,
                "time_horizon": 24,
                "time_step": 1.0
            }
        },
        {
            "name": "Invalid Strategy",
            "config": {
                "strategy": "invalid_strategy",
                "num_scenarios": 10
            }
        },
        {
            "name": "Out of Range Parameters",
            "config": {
                "strategy": "realtime",
                "num_scenarios": 2000,  # Too many
                "timeout_ms": 500000,   # Too long
                "time_step": 50         # Too large
            }
        },
        {
            "name": "Non-numeric Parameters",
            "config": {
                "strategy": "distributed",
                "num_scenarios": "ten",  # Should be numeric
                "timeout_ms": "fast"     # Should be numeric
            }
        }
    ]
    
    for test in test_configs:
        print(f"\nTesting: {test['name']}")
        print(f"Config: {test['config']}")
        
        errors = validate_optimization_config(test['config'])
        
        if not errors:
            print("✓ Configuration is valid")
        else:
            print("✗ Configuration has errors:")
            for error in errors:
                print(f"  - {error}")


def demonstrate_plugin_architecture():
    """Demonstrate the plugin architecture for expert models."""
    print("\n" + "=" * 70)
    print("PLUGIN ARCHITECTURE DEMONSTRATION")
    print("=" * 70)
    
    # Create a simple custom plugin
    from vpp.optimization import OptimizationPlugin, OptimizationResult, OptimizationStatus
    
    class DemoCustomPlugin(OptimizationPlugin):
        """Demo custom optimization plugin."""
        
        def __init__(self):
            super().__init__("demo_custom_optimizer", "1.0")
            self.solve_count = 0
        
        def is_available(self):
            return True  # Always available for demo
        
        def validate_problem(self, problem):
            return problem.metadata.get("type") == "stochastic"
        
        def solve(self, problem, timeout_ms=None):
            self.solve_count += 1
            
            # Simple heuristic: charge during low prices, discharge during high prices
            scenarios = problem.parameters.get("scenarios", [])
            if not scenarios:
                return OptimizationResult(
                    status=OptimizationStatus.FAILED,
                    objective_value=float('inf'),
                    solution={},
                    solve_time=0.001
                )
            
            # Use first scenario for simplicity
            prices = scenarios[0].get("prices", [0.1] * 24)
            
            # Simple threshold-based strategy
            avg_price = sum(prices) / len(prices)
            battery_schedule = []
            
            for price in prices:
                if price < avg_price * 0.8:  # Low price - charge
                    power = 100.0
                elif price > avg_price * 1.2:  # High price - discharge
                    power = -100.0
                else:  # Medium price - idle
                    power = 0.0
                battery_schedule.append(power)
            
            total_cost = sum(p * power for p, power in zip(prices, battery_schedule))
            
            return OptimizationResult(
                status=OptimizationStatus.SUCCESS,
                objective_value=total_cost,
                solution={
                    "battery_power": battery_schedule,
                    "method": "demo_custom_heuristic",
                    "avg_price_threshold": avg_price
                },
                solve_time=0.001,
                metadata={"solve_count": self.solve_count}
            )
    
    print("Creating custom optimization plugin...")
    
    # Create engine and register custom plugin
    engine = create_optimization_engine("standard")
    custom_plugin = DemoCustomPlugin()
    
    success = engine.register_plugin(custom_plugin)
    print(f"Plugin registration: {'✓ Success' if success else '✗ Failed'}")
    
    # Test the custom plugin
    base_data = {
        "base_prices": [0.08, 0.10, 0.15, 0.20, 0.25, 0.20, 0.15, 0.10] * 3,
        "battery_capacity": 1000.0,
        "max_power": 200.0
    }
    
    problem = create_stochastic_problem(base_data, num_scenarios=5)
    
    print(f"\nTesting custom plugin on stochastic problem...")
    result = engine.solve(problem)
    
    print(f"Status: {result.status.value}")
    print(f"Solve time: {result.solve_time:.3f} seconds")
    print(f"Objective value: ${result.objective_value:.2f}")
    print(f"Method: {result.solution.get('method', 'unknown')}")
    print(f"Plugin solve count: {result.metadata.get('solve_count', 'N/A')}")
    
    # Show plugin info
    plugin_info = engine.get_plugin_info()
    print(f"\nRegistered plugins:")
    for name, info in plugin_info.items():
        print(f"  {name}: {info}")
    
    # Show performance stats
    perf_stats = engine.get_performance_stats()
    print(f"\nEngine performance stats:")
    for key, value in perf_stats.items():
        print(f"  {key}: {value}")


def main():
    """Main demonstration function."""
    setup_logging()
    
    print("ENHANCED VPP OPTIMIZATION FRAMEWORK DEMONSTRATION")
    print("This demo showcases the advanced optimization capabilities")
    print("with plugin architecture, rule-based fallbacks, and expert models.\n")
    
    try:
        # Core optimization demonstrations
        demonstrate_stochastic_optimization()
        demonstrate_realtime_optimization()
        demonstrate_distributed_optimization()
        
        # Framework capabilities
        demonstrate_performance_benchmarking()
        demonstrate_configuration_validation()
        demonstrate_plugin_architecture()
        
        print("\n" + "=" * 70)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nKey features demonstrated:")
        print("✓ Stochastic optimization with scenario generation")
        print("✓ Real-time optimization for grid services")
        print("✓ Distributed optimization for multi-site coordination")
        print("✓ Plugin architecture for expert models")
        print("✓ Rule-based fallbacks for robustness")
        print("✓ Performance benchmarking and validation")
        print("✓ Custom plugin development")
        
        print("\nThe optimization framework provides:")
        print("- Production-ready rule-based fallbacks")
        print("- Clean interfaces for expert model integration")
        print("- Comprehensive performance monitoring")
        print("- Flexible configuration and validation")
        print("- Jupyter-friendly development workflow")
        
    except Exception as e:
        print(f"\nDemonstration failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
