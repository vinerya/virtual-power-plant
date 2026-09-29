"""
Advanced Configuration System Demonstration

This example shows how to use the enhanced configuration system with:
- Physics-based battery models
- Multi-objective optimization
- Rule-based systems
- Comprehensive validation
- Configuration management

Run from the repository root (no server or external service needed)::

    pip install -e .            # or: export PYTHONPATH=src
    python examples/advanced_configuration_demo.py [--output-dir DIR]

Files the demo writes (saved configs, a backup, and the ``vpp_advanced.log``
that ``configs/advanced_vpp_config.yaml`` asks for) go to ``--output-dir``,
or to a temporary directory that is removed afterwards.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

from vpp.config import (
    ConfigFormat,
    ConstraintConfig,
    OptimizationConfig,
    OptimizationObjective,
    ResourceConfig,
    RuleConfig,
    VPPConfig,
)
from vpp.models.battery import BatteryParameters, create_battery_model

CONFIG_PATH = (Path(__file__).parent.parent / "configs" / "advanced_vpp_config.yaml").resolve()


def demonstrate_configuration_loading():
    """Demonstrate loading and validating configuration from file."""
    print("=" * 60)
    print("CONFIGURATION LOADING DEMONSTRATION")
    print("=" * 60)

    # Load configuration from YAML file
    config = VPPConfig.load_from_file(CONFIG_PATH)
    print(f"✓ Successfully loaded configuration from {CONFIG_PATH}")
    print(f"  VPP Name: {config.name}")
    print(f"  Location: {config.location}")
    print(f"  Resources: {len(config.resources)}")
    print(f"  Optimization Strategy: {config.optimization.strategy}")
    print(f"  Heuristic Algorithm: {config.heuristics.algorithm}")
    print(f"  Rules: {len(config.rules.rules)}")

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
        timezone="UTC",
    )

    # Add optimization objectives
    config.optimization.objectives = [
        OptimizationObjective(
            name="cost_minimization",
            weight=0.6,
            priority=1,
            parameters={"include_demand_charges": True},
        ),
        OptimizationObjective(
            name="emissions_reduction", weight=0.4, priority=2, parameters={"carbon_price": 50.0}
        ),
    ]

    # Add constraints
    config.optimization.constraints = [
        ConstraintConfig(
            name="power_balance", parameters={"tolerance": 0.01}, violation_penalty=10000.0
        ),
        ConstraintConfig(
            name="ramp_limits", parameters={"max_ramp": 100.0}, violation_penalty=1000.0
        ),
    ]

    # Add rules
    config.rules.rules = [
        RuleConfig(
            name="emergency_shutdown",
            priority=1,
            conditions={"system_fault": True},
            actions={"shutdown_all": True, "notify_operator": True},
        ),
        RuleConfig(
            name="peak_demand_response",
            priority=5,
            conditions={"peak_demand_signal": True, "battery_soc": "> 0.3"},
            actions={"discharge_battery": True, "target_power": "max_discharge"},
        ),
    ]

    # Add resources
    config.add_resource(
        name="demo_battery",
        resource_type="battery",
        parameters={
            "nominal_capacity": 1000.0,
            "nominal_voltage": 400.0,
            "max_current": 250.0,
            "model_type": "simple",
        },
        constraints={"max_soc": 0.9, "min_soc": 0.1},
    )

    config.add_resource(
        name="demo_solar",
        resource_type="solar",
        parameters={"peak_power": 500.0, "panel_area": 2500.0, "panel_efficiency": 0.20},
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
        raise ValueError("no battery resource found in configuration")

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
        discharge_efficiency=params.get("discharge_efficiency", 0.95),
    )

    # Create battery model
    model_type = params.get("model_type", "simple")
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

    print(
        f"  After charging - SOC: {battery_model.state.soc:.3f}, "
        f"Temperature: {battery_model.state.temperature:.1f}°C"
    )

    # Discharge the battery
    print("  Discharging at 150 kW for 30 minutes...")
    for _ in range(30):  # 30 minutes
        battery_model.update(-150.0, 60.0)  # -150 kW for 60 seconds

    print(
        f"  After discharging - SOC: {battery_model.state.soc:.3f}, "
        f"Temperature: {battery_model.state.temperature:.1f}°C"
    )

    # Check safety
    if battery_model.is_safe_to_operate():
        print("  ✓ Battery is operating within safe limits")
    else:
        print("  ✗ Battery is outside safe operating limits")


def demonstrate_configuration_management(config: VPPConfig, output_dir: Path):
    """Demonstrate configuration management features."""
    print("\n" + "=" * 60)
    print("CONFIGURATION MANAGEMENT DEMONSTRATION")
    print("=" * 60)

    # Save configuration in different formats

    # Save as YAML
    yaml_path = output_dir / "demo_config.yaml"
    config.save_to_file(yaml_path, ConfigFormat.YAML)
    print(f"✓ Saved configuration as YAML: {yaml_path}")

    # Save as JSON
    json_path = output_dir / "demo_config.json"
    config.save_to_file(json_path, ConfigFormat.JSON)
    print(f"✓ Saved configuration as JSON: {json_path}")

    # Create backup (without a path it lands in the current directory)
    backup_path = config.backup_to_file(str(output_dir / "demo_config_backup.yaml"))
    print(f"✓ Created configuration backup: {backup_path}")

    # Round-trip: reload what was saved and check nothing was lost
    reloaded = VPPConfig.load_from_file(json_path)
    assert reloaded.to_dict() == config.to_dict(), "JSON round-trip changed the configuration"
    print("✓ Reloaded the JSON file; it matches the in-memory configuration")

    # Apply overrides on top of the existing configuration. merge() applies
    # only the fields set on the override (nested configs field by field);
    # lists are replaced whole, so extend the existing resource list.
    print("\nApplying configuration overrides:")
    overrides = VPPConfig(
        optimization=OptimizationConfig(solver_timeout=600),
        resources=[
            *config.resources,
            ResourceConfig(
                name="additional_battery",
                type="battery",
                parameters={"nominal_capacity": 500.0},
            ),
        ],
    )
    overridden = config.merge(overrides)
    assert isinstance(overridden, VPPConfig)
    assert overridden.name == config.name, "merge() dropped a field the override left unset"
    print(f"  Original resources: {len(config.resources)}")
    print(f"  Overridden resources: {len(overridden.resources)}")
    print(
        f"  Solver timeout: {config.optimization.solver_timeout} -> "
        f"{overridden.optimization.solver_timeout}"
    )
    print(f"  Still valid: {overridden.validate().is_valid}")


def main(argv: list[str] | None = None) -> int:
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="keep the files the demo writes here (default: a temporary directory)",
    )
    args = parser.parse_args(argv)

    print("ADVANCED VPP CONFIGURATION SYSTEM DEMONSTRATION")
    print("This demo showcases the enhanced configuration capabilities")
    print("including validation, physics-based models, and management features.\n")

    with tempfile.TemporaryDirectory(prefix="vpp-config-demo-") as tmp:
        output_dir = (args.output_dir or Path(tmp)).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        # The sample config logs to the relative path ``vpp_advanced.log``;
        # run from the output directory so that file lands there too.
        previous_cwd = Path.cwd()
        os.chdir(output_dir)
        try:
            return run_demo(output_dir)
        finally:
            os.chdir(previous_cwd)


def run_demo(output_dir: Path) -> int:
    # Load configuration from file
    config = demonstrate_configuration_loading()

    # Validate configuration
    if not demonstrate_configuration_validation(config):
        print("Stopping demo due to configuration validation errors.")
        return 1

    # Create programmatic configuration
    programmatic_config = demonstrate_programmatic_configuration()

    # Demonstrate battery model creation
    demonstrate_battery_model_creation(config)

    # Demonstrate configuration management
    demonstrate_configuration_management(programmatic_config, output_dir)

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nKey features demonstrated:")
    print("✓ Configuration loading from YAML/JSON files")
    print("✓ Comprehensive validation with detailed error reporting")
    print("✓ Programmatic configuration creation")
    print("✓ Physics-based battery model integration")
    print("✓ Configuration management (save, backup, reload, overrides)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
