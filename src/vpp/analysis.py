"""Analysis tools for research and benchmarking."""

from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from .resources import EnergyResource
from .optimization import OptimizationResult
from .simulation import SimulationMetrics, SimulationState
from .exceptions import AnalysisError

@dataclass
class ResourcePerformance:
    """Performance metrics for a resource."""
    capacity_factor: float
    availability: float
    efficiency: float
    ramp_rates: Dict[str, float]
    response_times: Dict[str, float]
    constraint_violations: Dict[str, int]
    cost_metrics: Dict[str, float]
    environmental_metrics: Dict[str, float]

@dataclass
class SystemPerformance:
    """System-wide performance metrics."""
    total_energy: float
    average_power: float
    peak_power: float
    load_factor: float
    optimization_metrics: Dict[str, float]
    reliability_metrics: Dict[str, float]
    economic_metrics: Dict[str, float]
    environmental_metrics: Dict[str, float]

class PerformanceAnalyzer:
    """Analyzes VPP system and resource performance."""
    
    def __init__(self, time_window: Optional[timedelta] = None):
        """Initialize analyzer with optional time window."""
        self.time_window = time_window
    
    def analyze_resource(
        self,
        states: List[SimulationState],
        resource_name: str
    ) -> ResourcePerformance:
        """Analyze performance of a specific resource."""
        try:
            # Extract resource states
            resource_states = [
                state.resource_states[resource_name]
                for state in states
                if resource_name in state.resource_states
            ]
            
            if not resource_states:
                raise AnalysisError(f"No data found for resource: {resource_name}")
            
            # Calculate metrics
            power_values = [state.get('current_power', 0) for state in resource_states]
            rated_power = max(state.get('rated_power', 0) for state in resource_states)
            
            # Basic metrics
            capacity_factor = np.mean(power_values) / rated_power if rated_power else 0
            availability = sum(1 for s in resource_states if s.get('online', False)) / len(resource_states)
            efficiency = np.mean([s.get('efficiency', 0) for s in resource_states])
            
            # Ramp rates
            power_changes = np.diff(power_values)
            ramp_rates = {
                "max_up": np.max(power_changes) if len(power_changes) > 0 else 0,
                "max_down": abs(np.min(power_changes)) if len(power_changes) > 0 else 0,
                "average": np.mean(np.abs(power_changes)) if len(power_changes) > 0 else 0
            }
            
            return ResourcePerformance(
                capacity_factor=capacity_factor,
                availability=availability,
                efficiency=efficiency,
                ramp_rates=ramp_rates,
                response_times={},  # TODO: Implement response time analysis
                constraint_violations={},  # TODO: Track violations
                cost_metrics={},  # TODO: Implement cost analysis
                environmental_metrics={}  # TODO: Implement environmental metrics
            )
            
        except Exception as e:
            raise AnalysisError(f"Resource analysis failed: {str(e)}")
    
    def analyze_system(
        self,
        states: List[SimulationState],
        optimization_results: List[OptimizationResult]
    ) -> SystemPerformance:
        """Analyze overall system performance."""
        try:
            # Extract power outputs
            total_power = [
                sum(state.power_output.values())
                for state in states
            ]
            
            # Basic metrics
            avg_power = np.mean(total_power)
            peak_power = np.max(total_power)
            load_factor = avg_power / peak_power if peak_power else 0
            
            # Optimization performance
            opt_success_rate = sum(1 for r in optimization_results if r.success) / len(optimization_results)
            target_deviation = np.mean([
                abs(r.actual_power - r.target_power) / r.target_power
                for r in optimization_results
                if r.target_power > 0
            ])
            
            return SystemPerformance(
                total_energy=sum(total_power) * (states[1].timestamp - states[0].timestamp).total_seconds() / 3600,
                average_power=avg_power,
                peak_power=peak_power,
                load_factor=load_factor,
                optimization_metrics={
                    "success_rate": opt_success_rate,
                    "target_deviation": target_deviation
                },
                reliability_metrics={
                    "availability": sum(1 for s in states if any(s.power_output.values())) / len(states)
                },
                economic_metrics={},  # TODO: Implement economic analysis
                environmental_metrics={}  # TODO: Implement environmental analysis
            )
            
        except Exception as e:
            raise AnalysisError(f"System analysis failed: {str(e)}")

class StrategyComparator:
    """Compares different optimization strategies."""
    
    def compare_metrics(
        self,
        results: Dict[str, List[OptimizationResult]]
    ) -> pd.DataFrame:
        """Compare metrics across different strategies."""
        metrics = {}
        
        for strategy_name, strategy_results in results.items():
            success_rate = np.mean([r.success for r in strategy_results])
            avg_deviation = np.mean([
                abs(r.actual_power - r.target_power) / r.target_power
                for r in strategy_results
                if r.target_power > 0 and r.success
            ])
            
            metrics[strategy_name] = {
                "success_rate": success_rate,
                "avg_deviation": avg_deviation,
                "total_power": sum(r.actual_power for r in strategy_results),
                "computation_time": np.mean([
                    r.metadata.get("computation_time", 0)
                    for r in strategy_results
                ])
            }
        
        return pd.DataFrame.from_dict(metrics, orient='index')
    
    def analyze_constraints(
        self,
        results: Dict[str, List[OptimizationResult]]
    ) -> Dict[str, Dict[str, int]]:
        """Analyze constraint violations across strategies."""
        violations = {}
        
        for strategy_name, strategy_results in results.items():
            strategy_violations = {}
            for result in strategy_results:
                if not result.success:
                    continue
                for constraint, satisfied in result.constraints_satisfied.items():
                    if not satisfied:
                        strategy_violations[constraint] = strategy_violations.get(constraint, 0) + 1
            violations[strategy_name] = strategy_violations
        
        return violations

class TimeSeriesAnalyzer:
    """Analyzes time series data from simulations."""
    
    def create_power_series(
        self,
        states: List[SimulationState]
    ) -> pd.DataFrame:
        """Create power output time series."""
        data = []
        for state in states:
            row = {
                "timestamp": state.timestamp,
                **state.power_output
            }
            data.append(row)
        
        return pd.DataFrame(data).set_index("timestamp")
    
    def analyze_patterns(
        self,
        power_series: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze temporal patterns in power output."""
        results = {}
        
        # Daily patterns
        daily_avg = power_series.groupby(power_series.index.hour).mean()
        results["daily_patterns"] = daily_avg.to_dict()
        
        # Weekly patterns if data spans weeks
        if (power_series.index.max() - power_series.index.min()).days >= 7:
            weekly_avg = power_series.groupby(power_series.index.dayofweek).mean()
            results["weekly_patterns"] = weekly_avg.to_dict()
        
        # Volatility
        results["volatility"] = power_series.std().to_dict()
        
        # Correlation between resources
        results["correlations"] = power_series.corr().to_dict()
        
        return results

def create_performance_report(
    system_performance: SystemPerformance,
    resource_performances: Dict[str, ResourcePerformance],
    time_patterns: Dict[str, Any]
) -> Dict[str, Any]:
    """Create comprehensive performance report."""
    return {
        "system": {
            "total_energy": system_performance.total_energy,
            "average_power": system_performance.average_power,
            "peak_power": system_performance.peak_power,
            "load_factor": system_performance.load_factor,
            "optimization": system_performance.optimization_metrics,
            "reliability": system_performance.reliability_metrics
        },
        "resources": {
            name: {
                "capacity_factor": perf.capacity_factor,
                "availability": perf.availability,
                "efficiency": perf.efficiency,
                "ramp_rates": perf.ramp_rates
            }
            for name, perf in resource_performances.items()
        },
        "patterns": time_patterns,
        "timestamp": datetime.utcnow().isoformat()
    }
