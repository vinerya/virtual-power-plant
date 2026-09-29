"""
Core library walkthrough: resources, dispatch and events.

This example uses the in-process building blocks of the ``vpp`` package
(no API server, database or external service):

- Describing the plant with ``VPPConfig``
- ``Battery`` / ``Solar`` / ``WindTurbine`` resources and their physics hooks
- ``VirtualPowerPlant`` proportional dispatch to a power target
- The typed in-process ``EventBus`` (type filters, predicate filters, history)
- Resource metrics, state history and (optionally) battery degradation

Run from the repository root::

    pip install -e .            # or: export PYTHONPATH=src
    python examples/advanced_usage.py

The degradation step needs the ``degradation`` extra (``rainflow``); it is
skipped with a note when that is not installed.
"""

import asyncio
import importlib.util
import sys

from vpp import VirtualPowerPlant, VPPConfig
from vpp.events import Event, EventBus, EventType
from vpp.exceptions import ResourceError
from vpp.resources import Battery, EnergyResource, Solar, WindTurbine


def build_resources() -> tuple[Battery, Solar, WindTurbine]:
    """Create one resource of each kind with realistic-looking ratings."""
    battery = Battery(
        capacity=2000.0,  # kWh
        current_charge=1500.0,  # kWh (75% charged)
        max_power=500.0,  # kW
        nominal_voltage=800.0,  # V
    )
    solar = Solar(
        peak_power=1000.0,  # kW
        panel_area=5000.0,  # m²
        efficiency=0.20,
    )
    wind = WindTurbine(
        rated_power=2000.0,  # kW
        rotor_diameter=90.0,  # m
        hub_height=80.0,  # m
        cut_in_speed=3.0,  # m/s
        cut_out_speed=25.0,  # m/s
        rated_speed=12.0,  # m/s
    )
    battery.name, solar.name, wind.name = "battery", "solar", "wind"
    return battery, solar, wind


def print_outputs(resources: list[EnergyResource]) -> None:
    for resource in resources:
        metrics = resource.get_metrics()
        print(
            f"  {metrics['name']:<8} {metrics['current_power']:8.1f} kW"
            f" of {metrics['rated_power']:7.1f} kW rated"
        )


async def main() -> int:
    # Event bus: one subscriber for plant activity, one for faults only.
    bus = EventBus()
    activity: list[Event] = []

    async def on_activity(event: Event) -> None:
        activity.append(event)
        print(f"  [event] {event.event_type.value}: {event.data}")

    async def on_fault(event: Event) -> None:
        print(f"  [ALERT] {event.source}: {event.data['error']}")

    bus.subscribe(
        on_activity,
        event_types={
            EventType.RESOURCE_ADDED,
            EventType.DISPATCH_EXECUTED,
            EventType.OPTIMIZATION_FAILED,
        },
    )
    bus.subscribe(
        on_fault,
        event_types={EventType.RESOURCE_FAULT},
        filter_fn=lambda event: event.severity in {"error", "critical"},
    )

    # Plant description
    config = VPPConfig(name="Demo VPP", location="San Francisco", timezone="America/Los_Angeles")
    battery, solar, wind = build_resources()
    config.add_resource("battery", "battery", {"capacity_kwh": battery.capacity})
    config.add_resource("solar", "solar", {"peak_power_kw": solar.rated_power})
    config.add_resource("wind", "wind", {"rated_power_kw": wind.rated_power})
    validation = config.validate()
    print(f"Configuration '{config.name}' valid: {validation.is_valid}")

    vpp = VirtualPowerPlant(config)

    print("\nAdding resources...")
    for resource in (battery, solar, wind):
        vpp.add_resource(resource)
        await bus.publish(
            Event(
                event_type=EventType.RESOURCE_ADDED,
                data={"name": resource.name, "rated_power_kw": resource.rated_power},
                source="demo",
            )
        )
    print(f"Total capacity: {vpp.total_capacity:.1f} kW")

    # Weather drives the renewable output.
    print("\nUpdating conditions (irradiance 900 W/m², 35 °C; wind 8 m/s)...")
    solar.update_conditions(irradiance=900.0, temperature=35.0)
    wind.update_wind(wind_speed=8.0)
    print_outputs(vpp.resources)
    print(f"Current output: {vpp.get_total_power():.1f} kW")

    # Dispatch to a target: VirtualPowerPlant splits it pro rata to rated
    # power (it does not look at weather-limited availability; the
    # vpp.optimization package does real dispatch optimisation).
    print("\nDispatching to 1500 kW...")
    if not vpp.optimize_dispatch(target_power=1500.0):
        raise RuntimeError("a 1500 kW target should be feasible")
    print_outputs(vpp.resources)
    await bus.publish(
        Event(
            event_type=EventType.DISPATCH_EXECUTED,
            data={"target_kw": vpp.target_power, "total_kw": round(vpp.get_total_power(), 1)},
            source="vpp",
        )
    )

    print("\nDispatching to 10 MW (above capacity)...")
    if vpp.optimize_dispatch(target_power=10_000.0):
        raise RuntimeError("a target above total capacity should be rejected")
    await bus.publish(
        Event(
            event_type=EventType.OPTIMIZATION_FAILED,
            data={"target_kw": 10_000.0, "capacity_kw": vpp.total_capacity},
            source="vpp",
            severity="warning",
        )
    )

    # Battery operations and limit enforcement.
    print("\nBattery operations:")
    battery.discharge(power=400.0, duration=0.5)
    print(f"  After 400 kW for 30 min: SOC {battery.get_metrics()['state_of_charge']:.1f}%")
    try:
        battery.charge(power=500.0, duration=10.0)
    except ResourceError as exc:
        await bus.publish(
            Event(
                event_type=EventType.RESOURCE_FAULT,
                data={"error": str(exc)},
                source=battery.name,
                severity="error",
            )
        )
    else:
        raise RuntimeError("charging 5 MWh into a 2 MWh battery should fail")

    # Degradation over a daily cycle (optional dependency).
    if importlib.util.find_spec("rainflow") is not None:
        soc_trace = [0.2, 0.5, 0.9, 0.9, 0.6, 0.2] * 30  # 30 daily cycles, 4 h steps
        loss = battery.apply_realized_dispatch(soc_trace, dt_hours=4.0)
        print(
            f"  30 simulated cycles: capacity loss {loss:.4%}, "
            f"SOH {battery.state_of_health:.4f}, "
            f"throughput {battery.cumulative_throughput_kwh:,.0f} kWh"
        )
    else:
        print("  (degradation step skipped: pip install -e '.[degradation]')")

    # Custom power curves and state history for analysis.
    solar.set_power_curve(lambda power: power * 0.97)  # e.g. 3% inverter loss
    solar.set_power(500.0)
    history = solar.get_state_history()
    print(f"\nSolar set to 500 kW through a 3% loss curve: {history[-1]['power']:.1f} kW")

    # Event history
    print(f"\nEvents published: {bus.publish_count}, delivered to activity log: {len(activity)}")
    for event in bus.get_history(limit=10):
        print(f"  {event.severity:<7} {event.event_type.value:<22} from {event.source}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
