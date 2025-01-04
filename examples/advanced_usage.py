"""
Advanced usage example of the Virtual Power Plant library.
This example demonstrates the full capabilities including:
- Configuration management
- Event handling
- Multiple optimization strategies
- Data persistence
- Resource monitoring
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Set

from vpp import VirtualPowerPlant, Battery, Solar, WindTurbine
from vpp.config import VPPConfig
from vpp.events import (
    Event, EventType, EventHandler, EventPriority,
    WebhookEventHandler
)
from vpp.optimization import (
    OptimizationConstraint,
    create_strategy
)

class CustomEventHandler(EventHandler):
    """Custom event handler for demonstration."""
    
    def __init__(self, event_types: Set[EventType]):
        super().__init__(event_types, EventPriority.LOW)
        self.events = []
    
    async def handle_event(self, event: Event) -> None:
        """Store events for later analysis."""
        self.events.append(event)
        print(f"Event received: {event.type.value} - {json.dumps(event.data)}")

async def main():
    # Create custom configuration
    config = VPPConfig(
        name="Demo VPP",
        location="San Francisco",
        timezone="America/Los_Angeles",
        optimization={"strategy": "dynamic", "update_interval": 300},
        storage={"backend": "sqlite", "connection_string": "sqlite:///demo_vpp.db"},
        logging={"level": "INFO", "file_path": "demo_vpp.log"},
        resource_limits={"max_resources": 100, "max_power_per_resource": 5000.0}
    )
    
    # Save configuration
    config_path = Path("demo_config.json")
    config.save_to_file(str(config_path))
    
    # Initialize VPP with configuration
    vpp = VirtualPowerPlant(str(config_path))
    
    # Add custom event handler
    custom_handler = CustomEventHandler({
        EventType.RESOURCE_ADDED,
        EventType.POWER_OPTIMIZATION,
        EventType.SYSTEM_ERROR
    })
    vpp.event_manager.add_handler(custom_handler)
    
    # Add webhook handler (commented out - replace URL with actual endpoint)
    # webhook_handler = WebhookEventHandler(
    #     "https://your-endpoint/vpp-events",
    #     event_types={EventType.POWER_THRESHOLD, EventType.SYSTEM_ERROR},
    #     min_priority=EventPriority.HIGH
    # )
    # vpp.event_manager.add_handler(webhook_handler)
    
    # Create energy resources with different characteristics
    battery = Battery(
        capacity=2000,      # 2 MWh capacity
        current_charge=1500, # 75% charged
        max_power=500       # 500 kW max power
    )
    
    solar = Solar(
        peak_power=1000,    # 1 MW peak power
        current_irradiance=800.0  # Current solar irradiance (W/m²)
    )
    
    wind = WindTurbine(
        rated_power=2000,   # 2 MW rated power
        cut_in_speed=3.0,   # Minimum wind speed for generation
        rated_speed=12.0,   # Wind speed for rated power
        cut_out_speed=25.0  # Maximum wind speed for generation
    )
    
    # Add resources to VPP
    vpp.add_resource(battery)
    vpp.add_resource(solar)
    vpp.add_resource(wind)
    
    # Simulate changing conditions
    print("\nSimulating changing conditions...")
    
    # Update wind conditions
    wind.update_wind_speed(8.0)  # Set current wind speed to 8 m/s
    
    # Update solar conditions
    solar.update_irradiance(900.0)  # Increase solar irradiance
    
    # Get system status
    status = vpp.get_resource_status()
    print("\nVPP Status:")
    print(f"Total Capacity: {status['total_capacity']:.1f} kW")
    print(f"Available Power: {status['available_power']:.1f} kW")
    print(f"Resource Count: {status['resource_count']}")
    print(f"Active Resources: {status['active_resources']}")
    
    # Try different optimization strategies
    print("\nTesting different optimization strategies...")
    
    # Cost-based optimization
    cost_strategy = create_strategy('cost', cost_weights={
        battery: 0.15,  # Cost per kWh
        solar: 0.05,
        wind: 0.08
    })
    vpp.optimization_strategy = cost_strategy
    vpp.optimize_dispatch(target_power=2000)  # Target 2 MW
    
    # Priority-based optimization
    priority_strategy = create_strategy('priority', priorities={
        battery: 1.0,  # Lowest priority
        solar: 3.0,    # Highest priority
        wind: 2.0
    })
    vpp.optimization_strategy = priority_strategy
    vpp.optimize_dispatch(target_power=2000)
    
    # Dynamic optimization
    dynamic_strategy = create_strategy('dynamic', constraints={
        battery: OptimizationConstraint(
            min_power=0,
            max_power=500,
            priority=1.0,
            cost_per_kwh=0.15
        ),
        solar: OptimizationConstraint(
            min_power=0,
            max_power=1000,
            priority=3.0,
            cost_per_kwh=0.05
        ),
        wind: OptimizationConstraint(
            min_power=0,
            max_power=2000,
            priority=2.0,
            cost_per_kwh=0.08
        )
    })
    vpp.optimization_strategy = dynamic_strategy
    vpp.optimize_dispatch(target_power=2000)
    
    # Get efficiency metrics
    metrics = vpp.get_efficiency_metrics()
    print("\nEfficiency Metrics:")
    print(f"Capacity Factor: {metrics['capacity_factor']:.2%}")
    print(f"Utilization Rate: {metrics['utilization_rate']:.2%}")
    
    # Get power forecast
    forecast = vpp.forecast_production(hours=24)
    print("\nPower Forecast (next 24 hours):")
    for hour, power in enumerate(forecast):
        if hour % 6 == 0:  # Print every 6 hours
            print(f"Hour {hour:2d}: {power:.1f} kW")
    
    # Battery specific information
    print("\nBattery Status:")
    print(f"State of Charge: {battery.state_of_charge:.1f}%")
    print(f"Available Power: {battery.get_available_power():.1f} kW")
    
    # Demonstrate event handling
    print("\nEvents received:")
    for event in custom_handler.events:
        print(f"- {event.type.value}: Priority {event.priority.value}")
    
    # Proper shutdown
    print("\nShutting down VPP...")
    vpp.shutdown()

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
