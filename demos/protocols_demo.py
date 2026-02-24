"""Multi-Protocol Demo — OpenADR + OCPP + MQTT + Modbus working together.

Demonstrates:
- Protocol registry for managing multiple adapters
- OpenADR demand response event handling
- OCPP charge point management and V2G coordination
- MQTT topic structure
- Modbus register map definitions
- Cross-protocol event coordination
"""

from __future__ import annotations

import asyncio
import time

import numpy as np

from vpp.protocols.base import ProtocolRegistry, ProtocolMessage, ProtocolStatus
from vpp.protocols.openadr import OpenADRAdapter, DREvent, DRSignalType
from vpp.protocols.ocpp import OCPPAdapter, ChargePoint, ChargePointStatus
from vpp.protocols.mqtt import MQTTAdapter
from vpp.protocols.modbus import ModbusAdapter, INVERTER_MAPS


def run() -> None:
    """Run the multi-protocol demo."""
    print("=" * 70)
    print("  MULTI-PROTOCOL DEMO — OpenADR + OCPP + MQTT + Modbus")
    print("=" * 70)

    # --- 1. Protocol Registry Setup ---
    print("\n--- 1. Protocol Registry Setup ---")
    registry = ProtocolRegistry()

    openadr = OpenADRAdapter()
    ocpp = OCPPAdapter()
    mqtt = MQTTAdapter()
    modbus = ModbusAdapter()

    registry.register(openadr)
    registry.register(ocpp)
    registry.register(mqtt)
    registry.register(modbus)

    adapters = registry.list_adapters()
    print(f"  Registered {len(adapters)} protocol adapters:")
    for a in adapters:
        print(f"    [{a.name}] {a.__class__.__name__} v{a.version} — status: {a.status.value}")

    # --- 2. OpenADR Demand Response ---
    print("\n--- 2. OpenADR Demand Response ---")

    # Simulate receiving a DR event from utility
    dr_event = DREvent(
        event_id="EVT-2025-001",
        signal_type=DRSignalType.LOAD_DISPATCH,
        signal_level=0.5,  # reduce to 50%
        start_time=time.time() + 3600,
        duration_seconds=7200,
        market_context="http://utility.example.com/dr",
    )

    print(f"  Received DR event: {dr_event.event_id}")
    print(f"    Signal: {dr_event.signal_type.value} = {dr_event.signal_level}")
    print(f"    Duration: {dr_event.duration_seconds / 3600:.1f} hours")

    # Auto opt-in response (handle_incoming_event is async)
    response = asyncio.get_event_loop().run_until_complete(
        openadr.handle_incoming_event(dr_event)
    ) if asyncio.get_event_loop().is_running() is False else None

    # Fallback: run in a new loop for sync contexts
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already in an async context — just show simulated response
        print(f"    Response: optIn (auto)")
    else:
        response = asyncio.run(openadr.handle_incoming_event(dr_event))
        print(f"    Response: {response.opt_type} (auto)")

    print(f"    Events tracked: {len(openadr.get_active_events())}")

    # --- 3. OCPP Charge Point Management ---
    print("\n--- 3. OCPP Charge Point Management ---")

    # Register charge points
    cps = [
        ChargePoint(
            charge_point_id=f"CP-{i:03d}",
            vendor="ABB",
            model="Terra AC",
            num_connectors=2,
            status=ChargePointStatus.AVAILABLE,
            current_power_kw=0,
            v2g_capable=(i % 3 == 0),
        )
        for i in range(1, 7)
    ]
    for cp in cps:
        ocpp.register_charge_point(cp)

    v2g_count = sum(1 for cp in cps if cp.v2g_capable)
    print(f"  Registered {len(cps)} charge points ({v2g_count} V2G capable)")

    # Simulate charging session (set fields directly instead of async remote_start)
    cp1 = cps[0]
    cp1.status = ChargePointStatus.CHARGING
    cp1.current_power_kw = 7.4
    print(f"  Started charging on {cp1.charge_point_id}: {cp1.current_power_kw} kW")

    # Simulate V2G discharge request during DR event
    v2g_cp = next(cp for cp in cps if cp.v2g_capable)
    v2g_cp.status = ChargePointStatus.CHARGING
    v2g_cp.current_soc = 0.75
    v2g_cp.current_power_kw = -5.0  # V2G discharge
    print(f"  V2G discharge on {v2g_cp.charge_point_id}: "
          f"{abs(v2g_cp.current_power_kw)} kW (SOC={v2g_cp.current_soc*100:.0f}%)")

    total_power = ocpp.total_charging_power()
    v2g_capacity = ocpp.available_v2g_capacity()
    print(f"  Fleet: total_power={total_power:.1f}kW, v2g_capacity={v2g_capacity:.1f}kW")

    # --- 4. MQTT Topic Structure ---
    print("\n--- 4. MQTT IoT Telemetry ---")

    topics = [
        "vpp/site1/battery/+/soc",
        "vpp/site1/solar/+/power",
        "vpp/site1/meter/+/energy",
    ]
    print(f"  Topic patterns ({len(topics)}):")
    for topic in topics:
        print(f"    {topic}")

    # Show sample messages
    messages = [
        ("vpp/site1/battery/B001/soc", {"soc": 0.72, "voltage": 48.2}),
        ("vpp/site1/battery/B002/soc", {"soc": 0.45, "voltage": 47.8}),
        ("vpp/site1/solar/S001/power", {"power_kw": 4.2, "irradiance": 680}),
        ("vpp/site1/meter/M001/energy", {"import_kwh": 125.3, "export_kwh": 32.1}),
    ]
    print(f"\n  Sample messages ({len(messages)}):")
    for topic, payload in messages:
        print(f"    {topic} -> {payload}")

    # --- 5. Modbus Inverter Data ---
    print("\n--- 5. Modbus Inverter Data ---")

    print(f"  Adapter: {modbus.name} v{modbus.version}")
    print(f"  Register maps available:")
    for name, rmap in INVERTER_MAPS.items():
        print(f"    {name}: {len(rmap.registers)} registers ({rmap.name})")

    # Simulated register reads
    print(f"\n  Simulated register reads (SMA Sunny Boy):")
    sim_values = {
        "ac_power": (4250.0, "W"),
        "dc_power": (4380.0, "W"),
        "ac_voltage": (240.5, "V"),
        "ac_current": (17.7, "A"),
        "frequency": (60.01, "Hz"),
        "total_energy": (15234.5, "kWh"),
        "daily_energy": (28.3, "kWh"),
        "temperature": (42.0, "C"),
        "status": (1.0, ""),
    }
    for name, (val, unit) in sim_values.items():
        print(f"    {name:<20}: {val:>10.1f} {unit}")

    # --- 6. Cross-Protocol Coordination ---
    print("\n--- 6. Cross-Protocol Coordination ---")
    print("  Scenario: DR event triggers coordinated response across protocols")
    print()

    # DR event -> calculate required reduction
    current_load = 42.0  # kW
    target_load = current_load * dr_event.signal_level
    reduction_needed = current_load - target_load

    print(f"  Current load   : {current_load:.1f} kW")
    print(f"  Target load    : {target_load:.1f} kW")
    print(f"  Reduction needed: {reduction_needed:.1f} kW")
    print()

    # Step 1: Curtail EV charging
    ev_curtailment = min(abs(total_power), reduction_needed * 0.4)
    print(f"  Step 1 [OCPP]: Curtail EV charging by {ev_curtailment:.1f} kW")

    # Step 2: Activate V2G discharge
    v2g_dispatch = min(v2g_capacity, reduction_needed * 0.3)
    print(f"  Step 2 [OCPP]: Activate V2G discharge: {v2g_dispatch:.1f} kW")

    # Step 3: Battery discharge via Modbus
    batt_dispatch = min(5.0, max(0, reduction_needed - ev_curtailment - v2g_dispatch))
    print(f"  Step 3 [Modbus]: Battery discharge: {batt_dispatch:.1f} kW")

    # Step 4: Status update via MQTT
    print(f"  Step 4 [MQTT]: Publish DR response status to vpp/site1/dr/response")

    total_response = ev_curtailment + v2g_dispatch + batt_dispatch
    compliance = total_response / reduction_needed * 100 if reduction_needed > 0 else 100
    print(f"\n  Total response: {total_response:.1f} kW / {reduction_needed:.1f} kW needed "
          f"({compliance:.0f}%)")

    # Summary
    print("\n" + "-" * 70)
    print("  PROTOCOL SUMMARY")
    print("-" * 70)
    print(f"  OpenADR : 1 DR event handled, auto opt-in")
    print(f"  OCPP    : {len(cps)} charge points, {v2g_count} V2G, 1 active session")
    print(f"  MQTT    : {len(messages)} telemetry messages, {len(topics)} topic patterns")
    print(f"  Modbus  : {len(INVERTER_MAPS)} register maps, {len(sim_values)} registers read")
    print(f"  Cross-protocol DR response: {total_response:.1f} kW reduction achieved")
    print("-" * 70)

    print("\n[Demo complete]")


if __name__ == "__main__":
    run()
