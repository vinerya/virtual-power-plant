"""
Basic usage example of the Virtual Power Plant library.
This example demonstrates core functionality including:
- Configuring a VPP system
- Adding and managing resources
- Basic monitoring and optimization
- Event handling
"""

import asyncio
from datetime import datetime
from pathlib import Path

from vpp import VirtualPowerPlant
from vpp.config import VPPConfig, LogConfig
from vpp.resources import Battery, Solar, WindTurbine
from vpp.models import WeatherData
from vpp.events import EventHandler, EventType, Event

class SimpleEventHandler(EventHandler):
    """Simple event handler that prints events."""
    
    def handle_event(self, event: Event) -> None:
        """Print the event details."""
        print(f"\nEvent received: {event.type.name}")
        print(f"Priority: {event.priority.name}")
        print(f"Message: {event.message}")
        if event.data:
            print("Data:", event.data)

async def main():
    # Create basic configuration
    config = VPPConfig(
        name="Basic VPP Example",
        logging=LogConfig(
            level="INFO",
            file_path="basic_vpp.log"
        )
    )
    
    # Initialize VPP
    vpp = VirtualPowerPlant(config)
    
    # Add event handler
    vpp.event_manager.add_handler(SimpleEventHandler())
    
    print("Initializing Virtual Power Plant system...")
    print(f"Name: {config.name}")
    print(f"Logging to: {config.logging.file_path}")
    
    # Create and add resources
    print("\nAdding resources...")
    
    # Battery storage
    battery = Battery(
        capacity=1000,      # 1 MWh
        current_charge=700, # 70% initial charge
        max_power=250,      # 250 kW
        nominal_voltage=400 # 400V system
    )
    vpp.add_resource(battery)
    print("Added Battery Storage:")
    print(f"  Capacity: {battery.capacity} kWh")
    print(f"  Current Charge: {battery.current_charge} kWh")
    print(f"  Max Power: {battery.max_power} kW")
    
    # Solar PV
    solar = Solar(
        peak_power=500,    # 500 kW
        panel_area=2500,   # 2500 m²
        efficiency=0.20    # 20% efficient panels
    )
    vpp.add_resource(solar)
    print("\nAdded Solar PV:")
    print(f"  Peak Power: {solar.peak_power} kW")
    print(f"  Panel Area: {solar.panel_area} m²")
    print(f"  Efficiency: {solar.efficiency * 100}%")
    
    # Wind turbine
    wind = WindTurbine(
        rated_power=1000,  # 1 MW
        rotor_diameter=70, # 70m rotor
        hub_height=80,     # 80m hub height
        cut_in_speed=3.0,
        cut_out_speed=25.0,
        rated_speed=12.0
    )
    vpp.add_resource(wind)
    print("\nAdded Wind Turbine:")
    print(f"  Rated Power: {wind.rated_power} kW")
    print(f"  Rotor Diameter: {wind.rotor_diameter}m")
    print(f"  Hub Height: {wind.hub_height}m")
    
    # Simulate changing weather conditions
    print("\nSimulating weather conditions...")
    weather_data = WeatherData(
        temperature=25.0,     # 25°C
        irradiance=800.0,     # 800 W/m²
        wind_speed=8.0,       # 8 m/s
        wind_direction=180.0, # South
        cloud_cover=0.2,      # 20% cloud cover
        humidity=0.7,         # 70% humidity
        pressure=1013.25      # Standard pressure
    )
    
    # Update resource conditions
    solar.update_weather(weather_data)
    wind.update_weather(weather_data)
    battery.update_temperature(weather_data.temperature)
    
    # Get current system status
    print("\nCurrent System Status:")
    total_power = vpp.get_total_power()
    print(f"Total Power Output: {total_power:.1f} kW")
    print("\nResource Status:")
    for resource in vpp.resources:
        metrics = resource.get_metrics()
        print(f"\n{resource.__class__.__name__}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.2f}")
            else:
                print(f"  {metric}: {value}")
    
    # Optimize power dispatch
    print("\nOptimizing power dispatch...")
    target_power = 1500  # Target 1.5 MW
    success = vpp.optimize_dispatch(target_power)
    
    if success:
        print(f"Optimization successful!")
        print(f"Target Power: {target_power} kW")
        print(f"Actual Power: {vpp.get_total_power():.1f} kW")
        
        print("\nResource Allocation:")
        for resource in vpp.resources:
            print(f"{resource.__class__.__name__}: {resource._current_power:.1f} kW")
    else:
        print("Optimization failed - target power not achievable")
    
    # Demonstrate event handling
    print("\nDemonstrating event handling...")
    
    # Trigger a low battery event
    if isinstance(battery, Battery) and battery.current_charge < battery.capacity * 0.2:
        vpp.event_manager.emit_event(
            event_type=EventType.RESOURCE_WARNING,
            message="Battery charge level low",
            data={"charge_level": battery.current_charge}
        )
    
    # Trigger a high power event
    if total_power > vpp.total_capacity * 0.9:
        vpp.event_manager.emit_event(
            event_type=EventType.POWER_THRESHOLD,
            message="High power output",
            data={"power_level": total_power}
        )
    
    print("\nBasic usage demonstration completed!")

if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
