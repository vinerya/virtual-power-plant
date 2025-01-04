"""
Enterprise-level demonstration of the Virtual Power Plant library.
This example showcases advanced features including:
- Real-time monitoring and alerts
- Data validation
- Logging and metrics
- Frontend dashboard integration
- Multiple optimization strategies
- Grid stability management
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from vpp import VirtualPowerPlant
from vpp.config import VPPConfig, LogConfig, OptimizationConfig, StorageConfig
from vpp.resources import Battery, Solar, WindTurbine
from vpp.models import WeatherData, GridModel, MarketModel
from vpp.monitoring import SystemMonitor
from vpp.validation import validate_weather_data, validate_grid_metrics
from vpp.optimization import create_strategy, OptimizationConstraint
from vpp.events import EventType, EventPriority, WebhookEventHandler

async def main():
    # Create directory structure
    base_dir = Path("enterprise_vpp")
    logs_dir = base_dir / "logs"
    data_dir = base_dir / "data"
    config_dir = base_dir / "config"
    
    for directory in [logs_dir, data_dir, config_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    log_config = LogConfig(
        level="INFO",
        file_path=str(logs_dir / "vpp.log"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        rotate_size=10 * 1024 * 1024,  # 10MB
        backup_count=5
    )
    
    # Configure optimization
    opt_config = OptimizationConfig(
        strategy="dynamic",
        update_interval=300,
        forecast_horizon=24,
        min_adjustment=0.1,
        max_iterations=100
    )
    
    # Configure storage
    storage_config = StorageConfig(
        backend="sqlite",
        connection_string=f"sqlite:///{data_dir}/vpp.db",
        backup_enabled=True,
        retention_days=90
    )
    
    # Create main configuration
    config = VPPConfig(
        name="Enterprise VPP Demo",
        location="Data Center 1",
        timezone="UTC",
        logging=log_config,
        optimization=opt_config,
        storage=storage_config
    )
    
    # Save configuration
    config_path = config_dir / "vpp_config.json"
    config.save_to_file(str(config_path))
    
    # Initialize VPP with configuration
    vpp = VirtualPowerPlant(str(config_path))
    
    # Add webhook handler for critical events
    webhook_handler = WebhookEventHandler(
        "http://localhost:8000/webhooks/vpp",  # Replace with actual endpoint
        event_types={
            EventType.SYSTEM_ERROR,
            EventType.RESOURCE_ERROR,
            EventType.POWER_THRESHOLD,
            EventType.SECURITY_BREACH
        },
        min_priority=EventPriority.HIGH
    )
    vpp.event_manager.add_handler(webhook_handler)
    
    # Create and add resources
    battery = Battery(
        capacity=2000,      # 2 MWh
        current_charge=1500, # 75% charged
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
    
    # Add resources to VPP
    vpp.add_resource(battery)
    vpp.add_resource(solar)
    vpp.add_resource(wind)
    
    # Initialize monitoring
    monitor = SystemMonitor(vpp.logger)
    monitor.add_resource(battery)
    monitor.add_resource(solar)
    monitor.add_resource(wind)
    
    # Start monitoring in background
    monitoring_task = asyncio.create_task(monitor.start_monitoring())
    
    try:
        print("Starting VPP simulation...")
        print(f"Monitoring logs will be written to: {log_config.file_path}")
        print(f"Data will be stored in: {storage_config.connection_string}")
        
        # Simulate changing conditions
        for hour in range(24):
            # Generate weather data
            weather_data = {
                "temperature": 20 + 5 * np.sin(2 * np.pi * hour / 24),  # Daily temperature cycle
                "irradiance": max(0, 1000 * np.sin(np.pi * (hour - 6) / 12)) if 6 <= hour < 18 else 0,
                "wind_speed": 8 + 4 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 1),
                "wind_direction": np.random.uniform(0, 360),
                "cloud_cover": np.random.uniform(0, 0.5),
                "humidity": 0.7,
                "pressure": 1013.25
            }
            
            # Validate weather data
            weather = validate_weather_data(weather_data)
            
            # Update resource conditions
            solar.update_weather(weather)
            wind.update_weather(weather)
            battery.update_temperature(weather.temperature)
            
            # Get market price for the hour
            price = market.get_price(
                hour=hour,
                demand=market.forecast_demand(hour, weather.temperature, False),
                renewable_ratio=0.6
            )
            
            # Optimize dispatch based on price
            if price > 0.15:  # High price - discharge battery
                target_power = vpp.total_capacity * 0.9
            elif price < 0.05:  # Low price - charge battery
                target_power = vpp.total_capacity * 0.2
            else:  # Medium price - balance generation
                target_power = vpp.total_capacity * 0.6
            
            # Check grid constraints
            grid_metrics = grid.power_quality_metrics(vpp.get_total_power())
            validated_metrics = validate_grid_metrics(grid_metrics)
            
            if grid.check_constraints(target_power):
                vpp.optimize_dispatch(target_power)
            else:
                # Reduce power if grid constraints violated
                vpp.optimize_dispatch(grid.max_power * 0.95)
            
            # Get system health status
            health = monitor.get_system_health()
            print(f"\nHour {hour:02d}:00 Status:")
            print(f"System Status: {health['status']}")
            print(f"Total Power: {health['power_metrics']['mean']:.1f} kW")
            print(f"Healthy Resources: {health['resource_health']['healthy']}/{health['resource_health']['total']}")
            print(f"Active Alerts: {health['active_alerts']}")
            
            # Simulate time passing
            await asyncio.sleep(1)  # Speed up simulation
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Clean shutdown
        monitor.stop_monitoring()
        await monitoring_task
        vpp.shutdown()
        print("VPP shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
