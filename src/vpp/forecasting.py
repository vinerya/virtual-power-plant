"""Forecasting module for predicting power generation and demand."""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from .resources import EnergyResource
from .exceptions import ForecastError

@dataclass
class ForecastConfig:
    """Configuration for forecasting."""
    horizon: int = 24  # hours
    resolution: int = 1  # hour
    confidence_level: float = 0.95
    use_weather: bool = True
    use_historical: bool = True
    min_history_points: int = 168  # 1 week of hourly data

@dataclass
class ForecastResult:
    """Result of a forecasting operation."""
    timestamps: List[datetime]
    values: List[float]
    confidence_intervals: List[Tuple[float, float]]
    metadata: Dict[str, Any]

class Forecaster(ABC):
    """Base class for forecasting implementations."""
    
    def __init__(self, config: ForecastConfig):
        """Initialize forecaster with configuration."""
        self.config = config
    
    @abstractmethod
    def forecast(
        self,
        resource: EnergyResource,
        current_time: datetime,
        historical_data: Optional[Dict[datetime, float]] = None,
        weather_data: Optional[Dict[datetime, Dict[str, float]]] = None
    ) -> ForecastResult:
        """Generate forecast for resource."""
        pass

class SimpleForecaster(Forecaster):
    """Simple forecasting using moving averages and basic patterns."""
    
    def forecast(
        self,
        resource: EnergyResource,
        current_time: datetime,
        historical_data: Optional[Dict[datetime, float]] = None,
        weather_data: Optional[Dict[datetime, Dict[str, float]]] = None
    ) -> ForecastResult:
        """Generate forecast using simple methods."""
        try:
            timestamps = []
            values = []
            intervals = []
            
            # Generate hourly timestamps
            for i in range(self.config.horizon):
                timestamps.append(current_time + timedelta(hours=i))
            
            if historical_data:
                # Use moving average if historical data available
                window = min(168, len(historical_data))  # 1 week window
                hist_values = list(historical_data.values())[-window:]
                baseline = np.mean(hist_values)
                std_dev = np.std(hist_values)
                
                for _ in timestamps:
                    value = baseline + np.random.normal(0, std_dev * 0.1)
                    ci_width = std_dev * 1.96  # 95% confidence interval
                    values.append(max(0, value))
                    intervals.append((max(0, value - ci_width), value + ci_width))
            else:
                # Fallback to rated power with daily pattern
                for ts in timestamps:
                    hour = ts.hour
                    # Simple daily pattern
                    if 6 <= hour <= 18:  # Daytime
                        value = resource.rated_power * 0.7
                        ci_width = resource.rated_power * 0.2
                    else:  # Nighttime
                        value = resource.rated_power * 0.3
                        ci_width = resource.rated_power * 0.1
                    values.append(value)
                    intervals.append((max(0, value - ci_width), value + ci_width))
            
            return ForecastResult(
                timestamps=timestamps,
                values=values,
                confidence_intervals=intervals,
                metadata={
                    "method": "simple",
                    "has_historical": bool(historical_data),
                    "has_weather": bool(weather_data)
                }
            )
            
        except Exception as e:
            raise ForecastError(f"Forecasting failed: {str(e)}")

class WeatherBasedForecaster(Forecaster):
    """Weather-aware forecasting for renewable resources."""
    
    def forecast(
        self,
        resource: EnergyResource,
        current_time: datetime,
        historical_data: Optional[Dict[datetime, float]] = None,
        weather_data: Optional[Dict[datetime, Dict[str, float]]] = None
    ) -> ForecastResult:
        """Generate forecast using weather data."""
        try:
            if not weather_data:
                raise ForecastError("Weather data required for this forecaster")
            
            timestamps = []
            values = []
            intervals = []
            
            # Generate forecasts based on weather
            for i in range(self.config.horizon):
                ts = current_time + timedelta(hours=i)
                timestamps.append(ts)
                
                if ts in weather_data:
                    weather = weather_data[ts]
                    # Calculate power based on weather
                    if hasattr(resource, 'panel_area'):  # Solar
                        irradiance = weather.get('irradiance', 0)
                        temp = weather.get('temperature', 25)
                        value = (
                            resource.rated_power *
                            (irradiance / 1000) *
                            (1 - 0.004 * (temp - 25))
                        )
                    elif hasattr(resource, 'rotor_diameter'):  # Wind
                        wind_speed = weather.get('wind_speed', 0)
                        if wind_speed < resource.cut_in_speed:
                            value = 0
                        elif wind_speed > resource.cut_out_speed:
                            value = 0
                        else:
                            value = min(
                                resource.rated_power,
                                resource.rated_power * (wind_speed / resource.rated_speed) ** 3
                            )
                    else:
                        value = resource.rated_power * 0.5
                    
                    uncertainty = value * 0.2  # 20% uncertainty
                    values.append(value)
                    intervals.append((max(0, value - uncertainty), value + uncertainty))
                else:
                    # Fallback for missing weather data
                    value = resource.rated_power * 0.5
                    values.append(value)
                    intervals.append((value * 0.3, value * 0.7))
            
            return ForecastResult(
                timestamps=timestamps,
                values=values,
                confidence_intervals=intervals,
                metadata={
                    "method": "weather_based",
                    "has_historical": bool(historical_data),
                    "has_weather": True,
                    "weather_parameters": list(next(iter(weather_data.values())).keys())
                }
            )
            
        except Exception as e:
            raise ForecastError(f"Weather-based forecasting failed: {str(e)}")

def get_forecaster(name: str, config: Optional[ForecastConfig] = None) -> Forecaster:
    """Factory function to create forecaster."""
    config = config or ForecastConfig()
    
    forecasters = {
        "simple": SimpleForecaster,
        "weather": WeatherBasedForecaster
    }
    
    if name not in forecasters:
        raise ValueError(f"Unknown forecaster type: {name}")
    
    return forecasters[name](config)
