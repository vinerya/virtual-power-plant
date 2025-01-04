"""
Advanced simulation example demonstrating realistic VPP behavior with:
- Physical modeling of components
- Weather effects on renewable generation
- Battery degradation over time
- Grid constraints and power quality
- Market price dynamics
"""

import asyncio
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

from vpp import VirtualPowerPlant
from vpp.resources import Battery, Solar, WindTurbine
from vpp.models import WeatherData, GridModel, MarketModel
from vpp.config import VPPConfig
from vpp.optimization import create_strategy, OptimizationConstraint

async def simulate_day(vpp: VirtualPowerPlant, 
                      grid: GridModel,
                      market: MarketModel,
                      start_time: datetime,
                      weather_data: List[WeatherData]):
    """Simulate one day of VPP operation."""
    
    results = {
        'time': [],
        'total_power': [],
        'solar_power': [],
        'wind_power': [],
        'battery_power': [],
        'battery_soc': [],
        'market_price': [],
        'grid_metrics': [],
        'revenue': 0.0
    }
    
    # Time steps (5-minute intervals)
    time_steps = 24 * 12
    dt = 300  # 5 minutes in seconds
    
    for step in range(time_steps):
        current_time = start_time + timedelta(minutes=5*step)
        hour = current_time.hour
        weather = weather_data[step]
        
        # Update resource conditions
        for resource in vpp.resources:
            if isinstance(resource, (Solar, WindTurbine)):
                resource.update_weather(weather)
            elif isinstance(resource, Battery):
                resource.update_temperature(weather.temperature)
        
        # Get market price and demand
        price = market.get_price(
            hour=hour,
            demand=market.forecast_demand(
                hour=hour,
                temperature=weather.temperature,
                is_weekend=current_time.weekday() >= 5
            ),
            renewable_ratio=0.6  # Example renewable energy ratio
        )
        
        # Optimize dispatch based on market price
        if price > 0.15:  # High price - discharge battery
            target_power = vpp.total_capacity * 0.9  # 90% of capacity
        elif price < 0.05:  # Low price - charge battery
            target_power = vpp.total_capacity * 0.2  # 20% of capacity
        else:  # Medium price - balance generation
            target_power = vpp.total_capacity * 0.6  # 60% of capacity
        
        vpp.optimize_dispatch(target_power)
        
        # Get power outputs
        total_power = vpp.get_total_power()
        solar_power = sum(r.get_available_power() for r in vpp.resources if isinstance(r, Solar))
        wind_power = sum(r.get_available_power() for r in vpp.resources if isinstance(r, WindTurbine))
        battery_power = sum(r._current_power for r in vpp.resources if isinstance(r, Battery))
        battery_soc = np.mean([r.get_metrics()['state_of_charge'] 
                             for r in vpp.resources if isinstance(r, Battery)])
        
        # Check grid constraints
        grid_metrics = grid.power_quality_metrics(total_power)
        if grid.check_constraints(total_power):
            # Calculate revenue
            revenue = total_power * price * (dt / 3600)  # Convert to kWh
            results['revenue'] += revenue
        else:
            # Reduce power if grid constraints violated
            vpp.optimize_dispatch(grid.max_power * 0.95)
        
        # Store results
        results['time'].append(current_time)
        results['total_power'].append(total_power)
        results['solar_power'].append(solar_power)
        results['wind_power'].append(wind_power)
        results['battery_power'].append(battery_power)
        results['battery_soc'].append(battery_soc)
        results['market_price'].append(price)
        results['grid_metrics'].append(grid_metrics)
        
        # Small delay to prevent CPU overload
        await asyncio.sleep(0.01)
    
    return results

def plot_results(results: Dict):
    """Plot simulation results."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Power output plot
    ax1.plot(results['time'], results['total_power'], 'k-', label='Total')
    ax1.plot(results['time'], results['solar_power'], 'y-', label='Solar')
    ax1.plot(results['time'], results['wind_power'], 'b-', label='Wind')
    ax1.plot(results['time'], results['battery_power'], 'r-', label='Battery')
    ax1.set_ylabel('Power (kW)')
    ax1.legend()
    ax1.grid(True)
    
    # Battery SOC and market price
    ax2.plot(results['time'], results['battery_soc'], 'g-', label='Battery SOC')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(results['time'], results['market_price'], 'r--', label='Price')
    ax2.set_ylabel('Battery SOC (%)')
    ax2_twin.set_ylabel('Price ($/kWh)')
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2)
    ax2.grid(True)
    
    # Grid metrics
    voltage_deviation = [m['voltage_deviation'] * 100 for m in results['grid_metrics']]
    frequency_deviation = [m['frequency_deviation'] * 100 for m in results['grid_metrics']]
    power_factor = [m['power_factor'] for m in results['grid_metrics']]
    
    ax3.plot(results['time'], voltage_deviation, 'b-', label='Voltage Dev.')
    ax3.plot(results['time'], frequency_deviation, 'r-', label='Frequency Dev.')
    ax3.plot(results['time'], power_factor, 'g-', label='Power Factor')
    ax3.set_ylabel('Grid Metrics')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

async def main():
    # Initialize VPP with configuration
    config = VPPConfig(
        name="Simulation VPP",
        optimization={"strategy": "dynamic", "update_interval": 300},
        storage={"backend": "sqlite", "connection_string": "sqlite:///simulation.db"}
    )
    
    vpp = VirtualPowerPlant(config)
    
    # Create resources with realistic parameters
    battery = Battery(
        capacity=2000,      # 2 MWh
        current_charge=1500, # 75% initial charge
        max_power=500,      # 500 kW
        nominal_voltage=400, # 400V system
        max_c_rate=0.5      # 0.5C max charge/discharge rate
    )
    
    solar = Solar(
        peak_power=1000,    # 1 MW
        panel_area=5000,    # 5000 m²
        efficiency=0.22     # 22% efficient panels
    )
    
    wind = WindTurbine(
        rated_power=2000,   # 2 MW
        rotor_diameter=90,  # 90m rotor
        hub_height=100,     # 100m hub height
        cut_in_speed=3.0,
        cut_out_speed=25.0,
        rated_speed=12.0
    )
    
    vpp.add_resource(battery)
    vpp.add_resource(solar)
    vpp.add_resource(wind)
    
    # Initialize grid model
    grid = GridModel(
        nominal_voltage=400,
        nominal_frequency=50,
        max_power=5000,
        impedance=complex(0.1, 0.1)
    )
    
    # Initialize market model
    market = MarketModel(
        base_price=0.10,    # $0.10/kWh base price
        peak_price=0.25,    # $0.25/kWh peak price
        peak_hours=[9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    )
    
    # Generate synthetic weather data
    def generate_weather(steps: int) -> List[WeatherData]:
        weather_data = []
        for i in range(steps):
            hour = (i * 5) // 60  # 5-minute steps
            
            # Diurnal temperature variation
            temp = 20 + 5 * np.sin(2 * np.pi * (hour - 6) / 24)
            
            # Solar irradiance pattern
            if 6 <= hour < 20:  # Daylight hours
                irr = 1000 * np.sin(np.pi * (hour - 6) / 14) * (0.7 + 0.3 * np.random.random())
            else:
                irr = 0
            
            # Wind pattern with some randomness
            wind = 8 + 4 * np.sin(2 * np.pi * hour / 24) + 2 * np.random.randn()
            wind = max(0, wind)
            
            weather_data.append(WeatherData(
                temperature=temp,
                irradiance=max(0, irr),
                wind_speed=wind,
                wind_direction=np.random.uniform(0, 360),
                cloud_cover=np.random.uniform(0, 0.5),
                humidity=0.7,
                pressure=1013.25
            ))
        
        return weather_data
    
    # Run simulation
    start_time = datetime(2023, 6, 21, 0, 0)  # Summer solstice
    weather_data = generate_weather(24 * 12)  # 5-minute intervals for 24 hours
    
    print("Starting simulation...")
    results = await simulate_day(vpp, grid, market, start_time, weather_data)
    
    print(f"\nSimulation completed!")
    print(f"Total revenue: ${results['revenue']:.2f}")
    print("\nResource final metrics:")
    for resource in vpp.resources:
        print(f"\n{resource.__class__.__name__}:")
        for metric, value in resource.get_metrics().items():
            print(f"  {metric}: {value:.2f}")
    
    # Plot results
    plot_results(results)

if __name__ == "__main__":
    asyncio.run(main())
