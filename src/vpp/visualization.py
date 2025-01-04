"""Visualization tools for VPP analysis and research."""

from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from .simulation import SimulationState
from .optimization import OptimizationResult
from .analysis import ResourcePerformance, SystemPerformance

class PowerPlotter:
    """Plots power-related visualizations."""
    
    def __init__(self, style: str = "seaborn"):
        """Initialize plotter with style."""
        plt.style.use(style)
    
    def plot_power_output(
        self,
        states: List[SimulationState],
        resource_names: Optional[List[str]] = None,
        rolling_window: Optional[int] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """Plot power output over time."""
        # Create DataFrame
        data = []
        for state in states:
            row = {
                "timestamp": state.timestamp,
                **state.power_output
            }
            data.append(row)
        df = pd.DataFrame(data).set_index("timestamp")
        
        if rolling_window:
            df = df.rolling(window=rolling_window).mean()
        
        if resource_names:
            df = df[resource_names]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        df.plot(ax=ax)
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Power (kW)")
        ax.set_title("Resource Power Output")
        ax.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        return fig
    
    def plot_daily_pattern(
        self,
        states: List[SimulationState],
        resource_name: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """Plot daily power patterns."""
        data = []
        for state in states:
            if resource_name:
                power = state.power_output.get(resource_name, 0)
            else:
                power = sum(state.power_output.values())
            
            data.append({
                "hour": state.timestamp.hour,
                "power": power
            })
        
        df = pd.DataFrame(data)
        hourly_avg = df.groupby("hour")["power"].agg(['mean', 'std']).reset_index()
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(hourly_avg["hour"], hourly_avg["mean"], 'b-', label='Mean Power')
        ax.fill_between(
            hourly_avg["hour"],
            hourly_avg["mean"] - hourly_avg["std"],
            hourly_avg["mean"] + hourly_avg["std"],
            alpha=0.2
        )
        
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Power (kW)")
        ax.set_title(f"Daily Power Pattern{f' - {resource_name}' if resource_name else ''}")
        ax.grid(True)
        
        return fig

class PerformancePlotter:
    """Plots performance-related visualizations."""
    
    def plot_resource_comparison(
        self,
        performances: Dict[str, ResourcePerformance],
        metrics: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """Plot resource performance comparison."""
        if not metrics:
            metrics = ['capacity_factor', 'availability', 'efficiency']
        
        data = []
        for name, perf in performances.items():
            for metric in metrics:
                value = getattr(perf, metric)
                if isinstance(value, (int, float)):
                    data.append({
                        'Resource': name,
                        'Metric': metric,
                        'Value': value
                    })
        
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(data=df, x='Resource', y='Value', hue='Metric', ax=ax)
        
        ax.set_title("Resource Performance Comparison")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def plot_optimization_results(
        self,
        results: List[OptimizationResult],
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """Plot optimization results analysis."""
        data = pd.DataFrame([
            {
                'target': r.target_power,
                'actual': r.actual_power,
                'deviation': abs(r.actual_power - r.target_power) / r.target_power if r.target_power > 0 else 0,
                'success': r.success
            }
            for r in results
        ])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Target vs Actual
        ax1.scatter(data['target'], data['actual'], alpha=0.5)
        max_power = max(data['target'].max(), data['actual'].max())
        ax1.plot([0, max_power], [0, max_power], 'r--', label='Perfect Match')
        ax1.set_xlabel("Target Power (kW)")
        ax1.set_ylabel("Actual Power (kW)")
        ax1.set_title("Target vs Actual Power")
        ax1.grid(True)
        ax1.legend()
        
        # Deviation histogram
        sns.histplot(data=data, x='deviation', ax=ax2)
        ax2.set_xlabel("Relative Deviation")
        ax2.set_ylabel("Count")
        ax2.set_title("Distribution of Power Deviations")
        
        plt.tight_layout()
        return fig

class StrategyPlotter:
    """Plots strategy comparison visualizations."""
    
    def plot_strategy_comparison(
        self,
        results: Dict[str, List[OptimizationResult]],
        metrics: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """Plot strategy comparison."""
        if not metrics:
            metrics = ['success_rate', 'avg_deviation']
        
        data = []
        for strategy, strategy_results in results.items():
            success_rate = np.mean([r.success for r in strategy_results])
            avg_deviation = np.mean([
                abs(r.actual_power - r.target_power) / r.target_power
                for r in strategy_results
                if r.target_power > 0 and r.success
            ])
            
            data.append({
                'Strategy': strategy,
                'Success Rate': success_rate,
                'Average Deviation': avg_deviation
            })
        
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=figsize)
        df.plot(x='Strategy', y=metrics, kind='bar', ax=ax)
        
        ax.set_title("Strategy Performance Comparison")
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def plot_computation_time(
        self,
        results: Dict[str, List[OptimizationResult]],
        figsize: Tuple[int, int] = (8, 6)
    ) -> plt.Figure:
        """Plot computation time comparison."""
        times = {
            strategy: [
                r.metadata.get('computation_time', 0)
                for r in strategy_results
            ]
            for strategy, strategy_results in results.items()
        }
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(data=pd.DataFrame(times), ax=ax)
        
        ax.set_ylabel("Computation Time (seconds)")
        ax.set_title("Strategy Computation Time Comparison")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig

def create_dashboard(
    system_performance: SystemPerformance,
    resource_performances: Dict[str, ResourcePerformance],
    optimization_results: List[OptimizationResult],
    states: List[SimulationState],
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """Create comprehensive performance dashboard."""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2)
    
    # Power output
    ax1 = fig.add_subplot(gs[0, :])
    power_data = pd.DataFrame([
        {
            'timestamp': state.timestamp,
            'power': sum(state.power_output.values())
        }
        for state in states
    ]).set_index('timestamp')
    power_data.plot(ax=ax1, title='Total Power Output')
    ax1.set_ylabel('Power (kW)')
    
    # Resource performance
    ax2 = fig.add_subplot(gs[1, 0])
    perf_data = pd.DataFrame([
        {
            'Resource': name,
            'Capacity Factor': perf.capacity_factor
        }
        for name, perf in resource_performances.items()
    ])
    sns.barplot(data=perf_data, x='Resource', y='Capacity Factor', ax=ax2)
    ax2.set_title('Resource Capacity Factors')
    plt.xticks(rotation=45)
    
    # Optimization performance
    ax3 = fig.add_subplot(gs[1, 1])
    opt_data = pd.DataFrame([
        {
            'target': r.target_power,
            'actual': r.actual_power
        }
        for r in optimization_results
    ])
    ax3.scatter(opt_data['target'], opt_data['actual'], alpha=0.5)
    max_power = max(opt_data['target'].max(), opt_data['actual'].max())
    ax3.plot([0, max_power], [0, max_power], 'r--')
    ax3.set_title('Target vs Actual Power')
    ax3.set_xlabel('Target Power (kW)')
    ax3.set_ylabel('Actual Power (kW)')
    
    # System metrics
    ax4 = fig.add_subplot(gs[2, :])
    metrics = {
        'Load Factor': system_performance.load_factor,
        'Optimization Success': system_performance.optimization_metrics['success_rate'],
        'Availability': system_performance.reliability_metrics['availability']
    }
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), ax=ax4)
    ax4.set_title('System Metrics')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig
