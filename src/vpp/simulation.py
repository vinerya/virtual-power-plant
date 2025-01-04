"""Simulation module for testing and evaluating VPP strategies."""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np
import json
import time

from .resources import EnergyResource, Battery, Solar, WindTurbine
from .optimization import OptimizationStrategy, OptimizationResult
from .forecasting import Forecaster, ForecastResult
from .exceptions import SimulationError

@dataclass
class SimulationConfig:
    """Configuration for simulation runs."""
    start_time: datetime
    end_time: datetime
    time_step: timedelta = timedelta(minutes=15)
    random_seed: Optional[int] = None
    record_metrics: bool = True
    record_states: bool = True
    include_weather: bool = True
    include_market: bool = True

@dataclass
class SimulationMetrics:
    """Metrics collected during simulation."""
    total_energy: float  # kWh
    average_power: float  # kW
    peak_power: float  # kW
    capacity_factor: float
    optimization_success_rate: float
    average_computation_time: float  # seconds
    resource_utilization: Dict[str, float]
    constraint_violations: Dict[str, int]
    costs: Dict[str, float]
    emissions: Dict[str, float]

@dataclass
class SimulationState:
    """State snapshot during simulation."""
    timestamp: datetime
    power_output: Dict[str, float]
    resource_states: Dict[str, Dict[str, Any]]
    weather_conditions: Optional[Dict[str, float]] = None
    market_prices: Optional[Dict[str, float]] = None

class WeatherGenerator:
    """Generates realistic weather patterns for simulation."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize weather generator."""
        self.rng = np.random.RandomState(seed)
        
        # Base patterns
        self.base_temp = 20  # °C
        self.temp_amplitude = 5
        self.base_irradiance = 500  # W/m²
        self.base_wind = 5  # m/s
    
    def generate(
        self,
        timestamp: datetime,
        location: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Generate weather conditions for given timestamp."""
        hour = timestamp.hour
        day_progress = (hour / 24) * 2 * np.pi
        
        # Temperature with daily cycle
        temp = (
            self.base_temp +
            self.temp_amplitude * np.sin(day_progress) +
            self.rng.normal(0, 1)
        )
        
        # Irradiance with daily cycle and cloud effects
        if 6 <= hour <= 18:  # Daylight hours
            cloud_cover = max(0, min(1, self.rng.beta(2, 2)))
            irradiance = (
                self.base_irradiance *
                np.sin(np.pi * (hour - 6) / 12) *
                (1 - 0.7 * cloud_cover) +
                self.rng.normal(0, 50)
            )
        else:
            irradiance = 0
        
        # Wind speed with some randomness
        wind_speed = max(0, self.base_wind + self.rng.normal(0, 2))
        
        return {
            "temperature": max(-10, min(40, temp)),
            "irradiance": max(0, irradiance),
            "wind_speed": wind_speed,
            "cloud_cover": cloud_cover if 6 <= hour <= 18 else 1.0
        }

class MarketSimulator:
    """Simulates energy market conditions."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize market simulator."""
        self.rng = np.random.RandomState(seed)
        
        # Base prices ($/kWh)
        self.base_price = 0.1
        self.peak_multiplier = 2.5
        self.valley_multiplier = 0.6
    
    def generate_prices(self, timestamp: datetime) -> Dict[str, float]:
        """Generate market prices for given timestamp."""
        hour = timestamp.hour
        
        # Base price with daily pattern
        if 17 <= hour <= 21:  # Peak hours
            multiplier = self.peak_multiplier
        elif 0 <= hour <= 5:  # Valley hours
            multiplier = self.valley_multiplier
        else:  # Normal hours
            multiplier = 1.0
        
        # Add some random variation
        noise = self.rng.normal(0, 0.02)
        price = self.base_price * multiplier * (1 + noise)
        
        return {
            "energy_price": max(0.05, price),
            "demand_response_price": price * 1.2,
            "reserve_price": price * 0.5
        }

class Simulator:
    """Main simulation controller."""
    
    def __init__(
        self,
        config: SimulationConfig,
        resources: List[EnergyResource],
        strategy: OptimizationStrategy,
        forecaster: Optional[Forecaster] = None
    ):
        """Initialize simulator with configuration."""
        self.config = config
        self.resources = resources
        self.strategy = strategy
        self.forecaster = forecaster
        
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
        
        self.weather_gen = WeatherGenerator(config.random_seed)
        self.market_sim = MarketSimulator(config.random_seed)
        
        self.states: List[SimulationState] = []
        self.optimization_results: List[OptimizationResult] = []
        self.forecast_results: List[ForecastResult] = []
    
    def run(
        self,
        target_power_func: Callable[[datetime], float]
    ) -> SimulationMetrics:
        """Run simulation."""
        try:
            current_time = self.config.start_time
            total_steps = int((self.config.end_time - self.config.start_time) / self.config.time_step)
            
            total_energy = 0
            peak_power = 0
            computation_times = []
            optimization_successes = 0
            constraint_violations = {}
            
            while current_time <= self.config.end_time:
                # Generate conditions
                weather = (
                    self.weather_gen.generate(current_time)
                    if self.config.include_weather else None
                )
                market = (
                    self.market_sim.generate_prices(current_time)
                    if self.config.include_market else None
                )
                
                # Update resource conditions
                for resource in self.resources:
                    if isinstance(resource, Solar) and weather:
                        resource.update_conditions(
                            weather['irradiance'],
                            weather['temperature']
                        )
                    elif isinstance(resource, WindTurbine) and weather:
                        resource.update_wind(weather['wind_speed'])
                
                # Get target power and optimize
                target = target_power_func(current_time)
                start_time = time.time()
                result = self.strategy.optimize(self.resources, target)
                computation_time = time.time() - start_time
                
                # Record results
                if self.config.record_states:
                    state = SimulationState(
                        timestamp=current_time,
                        power_output={r.__class__.__name__: r._current_power for r in self.resources},
                        resource_states={r.__class__.__name__: r.get_metrics() for r in self.resources},
                        weather_conditions=weather,
                        market_prices=market
                    )
                    self.states.append(state)
                
                # Update metrics
                total_energy += result.actual_power * self.config.time_step.total_seconds() / 3600
                peak_power = max(peak_power, result.actual_power)
                computation_times.append(computation_time)
                if result.success:
                    optimization_successes += 1
                
                current_time += self.config.time_step
            
            # Calculate final metrics
            total_time_hours = (self.config.end_time - self.config.start_time).total_seconds() / 3600
            total_capacity = sum(r.rated_power for r in self.resources)
            
            metrics = SimulationMetrics(
                total_energy=total_energy,
                average_power=total_energy / total_time_hours,
                peak_power=peak_power,
                capacity_factor=total_energy / (total_capacity * total_time_hours),
                optimization_success_rate=optimization_successes / total_steps,
                average_computation_time=np.mean(computation_times),
                resource_utilization={
                    r.__class__.__name__: r._current_power / r.rated_power
                    for r in self.resources
                },
                constraint_violations=constraint_violations,
                costs={},  # TODO: Implement cost tracking
                emissions={}  # TODO: Implement emissions tracking
            )
            
            return metrics
            
        except Exception as e:
            raise SimulationError(f"Simulation failed: {str(e)}")
    
    def save_results(self, path: str) -> None:
        """Save simulation results to file."""
        results = {
            "config": {
                "start_time": self.config.start_time.isoformat(),
                "end_time": self.config.end_time.isoformat(),
                "time_step": str(self.config.time_step),
                "random_seed": self.config.random_seed
            },
            "states": [
                {
                    "timestamp": state.timestamp.isoformat(),
                    "power_output": state.power_output,
                    "resource_states": state.resource_states,
                    "weather_conditions": state.weather_conditions,
                    "market_prices": state.market_prices
                }
                for state in self.states
            ],
            "optimization_results": [
                {
                    "success": result.success,
                    "target_power": result.target_power,
                    "actual_power": result.actual_power,
                    "resource_allocation": result.resource_allocation,
                    "metadata": result.metadata
                }
                for result in self.optimization_results
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
    
    @classmethod
    def load_results(cls, path: str) -> Dict[str, Any]:
        """Load simulation results from file."""
        with open(path, 'r') as f:
            return json.load(f)
