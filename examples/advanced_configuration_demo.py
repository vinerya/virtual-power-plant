"""
Advanced Configuration System Demonstration

This example shows how to use the enhanced configuration system with:
- Physics-based battery models
- Multi-objective optimization
- Rule-based systems
- Comprehensive validation
- Configuration management
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Add the src directory to the path so we can import vpp modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vpp.config import (
    VPPConfig, OptimizationObjective, ConstraintConfig, 
    RuleConfig, ConfigFormat, ValidationLevel
)
from vpp.models.battery import (
    BatteryParameters, BatteryModel, create_battery_model
)


def setup_logging():
    """Setup logging for the demonstration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def demonstrate_configuration_loading():
    """Demonstrate loading and validating configuration from file."""
    print("=" * 60)
    print("CONFIGURATION LOADING DEMONSTRATION")
    print("=" * 60)
    
    # Load configuration from YAML file
    config_path = Path(__file__).parent.parent / "configs" / "advanced_vpp_config.yaml"
    
    try:
        config = VPPConfig.load_from_file(config_path)
        print(f"✓ Successfully loaded configuration from {config_path}")
        print(f"  VPP Name: {config.name}")
        print(f"  Location: {config.location}")
        print(f"  Resources: {len(config.resources)}")
        print(f"  Optimization Strategy: {config.optimization.strategy}")
        print(f"  Heuristic Algorithm: {config.heuristics.algorithm}")
        print(f"  Rules: {len(config.rules.rules)}")
        
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return None
    
    return config


def demonstrate_configuration_validation(config: VPPConfig):
    """Demonstrate comprehensive configuration validation."""
    print("\n" + "=" * 60)
    print("CONFIGURATION VALIDATION DEMONSTRATION")
    print("=" * 60)
    
    # Validate the configuration
    validation_result = config.validate()
    
    if validation_result.is_valid:
        print("✓ Configuration validation passed!")
    else:
        print("✗ Configuration validation failed!")
        print("\nErrors:")
        for error in validation_result.errors:
            print(f"  - {error}")
    
    if validation_result.warnings:
        print("\nWarnings:")
        for warning in validation_result.warnings:
            print(f"  - {warning}")
    
    return validation_result.is_valid


def demonstrate_programmatic_configuration():
    """Demonstrate creating configuration programmatically."""
    print("\n" + "=" * 60)
    print("PROGRAMMATIC CONFIGURATION DEMONSTRATION")
    print("=" * 60)
    
    # Create a new VPP configuration programmatically
    config = VPPConfig(
        name="Programmatic VPP",
        description="Created programmatically for demonstration",
        location="Demo Location",
        timezone="UTC"
    )
    
    # Add optimization objectives
    config.optimization.objectives = [
        OptimizationObjective(
            name="cost_minimization",
            weight=0.6,
            priority=1,
            parameters={"include_demand_charges": True}
        ),
        OptimizationObjective(
            name="emissions_reduction",
            weight=0.4,
            priority=2,
            parameters={"carbon_price": 50.0}
        )
    ]
    
    # Add constraints
    config.optimization.constraints = [
        ConstraintConfig(
            name="power_balance",
            parameters={"tolerance": 0.01},
            violation_penalty=10000.0
        ),
        ConstraintConfig(
            name="ramp_limits",
            parameters={"max_ramp": 100.0},
            violation_penalty=1000.0
        )
    ]
    
    # Add rules
    config.rules.rules = [
        RuleConfig(
            name="emergency_shutdown",
            priority=1,
            conditions={"system_fault": True},
            actions={"shutdown_all": True, "notify_operator": True}
        ),
        RuleConfig(
            name="peak_demand_response",
            priority=5,
            conditions={"peak_demand_signal": True, "battery_soc": "> 0.3"},
            actions={"discharge_battery": True, "target_power": "max_discharge"}
        )
    ]
    
    # Add resources
    config.add_resource(
        name="demo_battery",
        resource_type="battery",
        parameters={
            "nominal_capacity": 1000.0,
            "nominal_voltage": 400.0,
            "max_current": 250.0,
            "model_type": "simple"
        },
        constraints={
            "max_soc": 0.9,
            "min_soc": 0.1
        }
    )
    
    config.add_resource(
        name="demo_solar",
        resource_type="solar",
        parameters={
            "peak_power": 500.0,
            "panel_area": 2500.0,
            "panel_efficiency": 0.20
        }
    )
    
    print("✓ Created programmatic configuration")
    print(f"  Objectives: {len(config.optimization.objectives)}")
    print(f"  Constraints: {len(config.optimization.constraints)}")
    print(f"  Rules: {len(config.rules.rules)}")
    print(f"  Resources: {len(config.resources)}")
    
    # Validate the programmatic configuration
    if config.validate_and_log():
        print("✓ Programmatic configuration is valid")
    else:
        print("✗ Programmatic configuration has validation errors")
    
    return config


def demonstrate_battery_model_creation(config: VPPConfig):
    """Demonstrate creating advanced battery models from configuration."""
    print("\n" + "=" * 60)
    print("BATTERY MODEL CREATION DEMONSTRATION")
    print("=" * 60)
    
    # Find battery resource in configuration
    battery_config = None
    for resource in config.resources:
        if resource.type == "battery":
            battery_config = resource
            break
    
    if not battery_config:
        print("✗ No battery resource found in configuration")
        return
    
    print(f"Found battery resource: {battery_config.name}")
    
    # Create battery parameters from configuration
    params = battery_config.parameters
    battery_params = BatteryParameters(
        nominal_capacity=params.get("nominal_capacity", 1000.0),
        nominal_voltage=params.get("nominal_voltage", 400.0),
        max_voltage=params.get("max_voltage", 420.0),
        min_voltage=params.get("min_voltage", 320.0),
        max_current=params.get("max_current", 250.0),
        internal_resistance=params.get("internal_resistance", 0.01),
        capacity_fade_rate=params.get("capacity_fade_rate", 0.0002),
        resistance_growth_rate=params.get("resistance_growth_rate", 0.0001),
        calendar_fade_rate=params.get("calendar_fade_rate", 0.00005),
        charge_efficiency=params.get("charge_efficiency", 0.95),
        discharge_efficiency=params.get("discharge_efficiency", 0.95)
    )
    
    # Create battery model
    model_type = params.get("model_type", "simple")
    try:
        battery_model = create_battery_model(model_type, battery_params, battery_config)
        print(f"✓ Created {model_type} battery model")
        print(f"  Initial SOC: {battery_model.state.soc:.2f}")
        print(f"  Initial SOH: {battery_model.state.soh:.2f}")
        print(f"  Temperature: {battery_model.state.temperature:.1f}°C")
        print(f"  Available Energy: {battery_model.get_available_energy():.1f} kWh")
        print(f"  Storage Capacity: {battery_model.get_storage_capacity():.1f} kWh")
        
        # Demonstrate battery operation
        print("\nDemonstrating battery operation:")
        
        # Charge the battery
        print("  Charging at 100 kW for 1 hour...")
        for _ in range(60):  # 60 minutes
            battery_model.update(100.0, 60.0)  # 100 kW for 60 seconds
        
        print(f"  After charging - SOC: {battery_model.state.soc:.3f}, "
              f"Temperature: {battery_model.state.temperature:.1f}°C")
        
        # Discharge the battery
        print("  Discharging at 150 kW for 30 minutes...")
        for _ in range(30):  # 30 minutes
            battery_model.update(-150.0, 60.0)  # -150 kW for 60 seconds
        
        print(f"  After discharging - SOC: {battery_model.state.soc:.3f}, "
              f"Temperature: {battery_model.state.temperature:.1f}°C")
        
        # Check safety
        if battery_model.is_safe_to_operate():
            print("  ✓ Battery is operating within safe limits")
        else:
            print("  ✗ Battery is outside safe operating limits")
            
    except Exception as e:
        print(f"✗ Failed to create battery model: {e}")


def demonstrate_configuration_management(config: VPPConfig):
    """Demonstrate configuration management features."""
    print("\n" + "=" * 60)
    print("CONFIGURATION MANAGEMENT DEMONSTRATION")
    print("=" * 60)
    
    # Save configuration in different formats
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Save as YAML
    yaml_path = output_dir / "demo_config.yaml"
    config.save_to_file(yaml_path, ConfigFormat.YAML)
    print(f"✓ Saved configuration as YAML: {yaml_path}")
    
    # Save as JSON
    json_path = output_dir / "demo_config.json"
    config.save_to_file(json_path, ConfigFormat.JSON)
    print(f"✓ Saved configuration as JSON: {json_path}")
    
    # Create backup
    backup_path = config.backup_to_file()
    print(f"✓ Created configuration backup: {backup_path}")
    
    # Demonstrate configuration merging
    print("\nDemonstrating configuration merging:")
    
    # Create a partial configuration for merging
    partial_config = VPPConfig(name="Partial Config")
    partial_config.optimization.solver_timeout = 600  # Different timeout
    partial_config.add_resource(
        name="additional_battery",
        resource_type="battery",
        parameters={"nominal_capacity": 500.0}
    )
    
    # Merge configurations
    merged_config = config.merge(partial_config)
    print(f"  Original resources: {len(config.resources)}")
    print(f"  Partial resources: {len(partial_config.resources)}")
    print(f"  Merged resources: {len(merged_config.resources)}")
    print(f"  Merged solver timeout: {merged_config.optimization.solver_timeout}")


def demonstrate_hot_reload_simulation():
    """Demonstrate hot reload capability simulation."""
    print("\n" + "=" * 60)
    print("HOT RELOAD SIMULATION DEMONSTRATION")
    print("=" * 60)
    
    print("Simulating configuration hot reload...")
    print("1. Initial configuration loaded")
    print("2. Configuration file modified (simulated)")
    print("3. Hot reload triggered (simulated)")
    print("4. New configuration validated and applied")
    print("✓ Hot reload completed successfully (simulated)")
    print("\nNote: Actual hot reload would require file system monitoring")


def main():
    """Main demonstration function."""
    setup_logging()
    
    print("ADVANCED VPP CONFIGURATION SYSTEM DEMONSTRATION")
    print("This demo showcases the enhanced configuration capabilities")
    print("including validation, physics-based models, and management features.\n")
    
    # Load configuration from file
    config = demonstrate_configuration_loading()
    if not config:
        return
    
    # Validate configuration
    is_valid = demonstrate_configuration_validation(config)
    if not is_valid:
        print("Stopping demo due to configuration validation errors.")
        return
    
    # Create programmatic configuration
    programmatic_config = demonstrate_programmatic_configuration()
    
    # Demonstrate battery model creation
    demonstrate_battery_model_creation(config)
    
    # Demonstrate configuration management
    demonstrate_configuration_management(programmatic_config)
    
    # Demonstrate hot reload simulation
    demonstrate_hot_reload_simulation()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nKey features demonstrated:")
    print("✓ Configuration loading from YAML/JSON files")
    print("✓ Comprehensive validation with detailed error reporting")
    print("✓ Programmatic configuration creation")
    print("✓ Physics-based battery model integration")
    print("✓ Configuration management (save, backup, merge)")
    print("✓ Hot reload capability (simulated)")
    print("\nThe advanced VPP configuration system provides:")
    print("- Type-safe configuration with validation")
    print("- Hierarchical configuration structure")
    print("- Multiple file format support")
    print("- Configuration merging and inheritance")
    print("- Integration with physics-based models")
    print("- Hot reload for dynamic reconfiguration")


if __name__ == "__main__":
    main()
