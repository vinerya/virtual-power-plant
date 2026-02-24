"""Residential VPP Demo — 10 homes with solar + battery.

Simulates a full 24-hour cycle with:
- 10 residential homes each with 6kW solar + 13.5kWh battery
- Real-time optimisation: peak shaving + self-consumption maximisation
- Cost comparison: no optimisation vs rule-based dispatch
"""

from __future__ import annotations

import numpy as np


def run() -> None:
    """Run the residential VPP demo."""
    from benchmarks.datasets import IEEETestCase

    print("=" * 70)
    print("  RESIDENTIAL VPP DEMO — 10 Homes with Solar + Battery")
    print("=" * 70)

    # Generate synthetic data
    dataset = IEEETestCase()
    data = dataset.generate(seed=42)
    n_steps = len(data["load_kw"])
    resolution_h = 0.25

    load = data["load_kw"]
    solar = data["solar_kw"]
    prices = data["price_per_kwh"]

    print(f"\nDataset: 24h @ 15min resolution ({n_steps} steps)")
    print(f"  Peak load      : {np.max(load):.1f} kW")
    print(f"  Peak solar      : {np.max(solar):.1f} kW")
    print(f"  Price range     : ${np.min(prices):.3f} - ${np.max(prices):.3f}/kWh")

    # Scale for 10 homes
    n_homes = 10
    total_load = load * n_homes
    total_solar = solar * n_homes

    # Battery fleet: 10 x 13.5 kWh batteries
    batt_capacity = 13.5 * n_homes  # 135 kWh total
    batt_power = 5.0 * n_homes  # 50 kW total
    soc = 0.5

    print(f"\nFleet:")
    print(f"  {n_homes} homes x 6kW solar = {6*n_homes}kW total solar")
    print(f"  {n_homes} homes x 13.5kWh battery = {batt_capacity}kWh total storage")
    print(f"  {n_homes} homes x 5kW inverter = {batt_power}kW total battery power")

    # --- Scenario 1: No optimisation (grid serves all net load) ---
    net_load_noop = total_load - total_solar
    grid_import_noop = np.clip(net_load_noop, 0, None)
    grid_export_noop = np.clip(-net_load_noop, 0, None)
    cost_noop = np.sum(grid_import_noop * prices * resolution_h)
    export_revenue_noop = np.sum(grid_export_noop * prices * 0.5 * resolution_h)  # feed-in at 50%
    net_cost_noop = cost_noop - export_revenue_noop

    # --- Scenario 2: Rule-based dispatch ---
    threshold = np.percentile(total_load, 70)
    battery_power_arr = np.zeros(n_steps)
    soc_arr = np.zeros(n_steps + 1)
    soc_arr[0] = soc

    for i in range(n_steps):
        net = total_load[i] - total_solar[i]

        if net > threshold:
            # Discharge battery to shave peak
            needed = net - threshold
            max_discharge = min(batt_power, (soc_arr[i] - 0.1) * batt_capacity / resolution_h)
            discharge = min(needed, max(max_discharge, 0))
            battery_power_arr[i] = discharge
            soc_arr[i + 1] = soc_arr[i] - (discharge * resolution_h / batt_capacity)
        elif net < 0:
            # Solar excess — charge battery
            available = abs(net)
            max_charge = min(batt_power, (0.9 - soc_arr[i]) * batt_capacity / resolution_h)
            charge = min(available, max(max_charge, 0))
            battery_power_arr[i] = -charge
            soc_arr[i + 1] = soc_arr[i] + (charge * resolution_h / batt_capacity)
        else:
            soc_arr[i + 1] = soc_arr[i]

    optimised_net = total_load - total_solar - battery_power_arr
    grid_import_opt = np.clip(optimised_net, 0, None)
    grid_export_opt = np.clip(-optimised_net, 0, None)
    cost_opt = np.sum(grid_import_opt * prices * resolution_h)
    export_revenue_opt = np.sum(grid_export_opt * prices * 0.5 * resolution_h)
    net_cost_opt = cost_opt - export_revenue_opt

    # Metrics
    peak_noop = np.max(grid_import_noop)
    peak_opt = np.max(grid_import_opt)
    peak_reduction = (1 - peak_opt / peak_noop) * 100 if peak_noop > 0 else 0

    self_consumption_noop = np.sum(np.minimum(total_solar, total_load)) / np.sum(total_solar) * 100
    total_solar_used = np.sum(np.minimum(total_solar, total_load + np.clip(-battery_power_arr, 0, None)))
    self_consumption_opt = min(100, total_solar_used / np.sum(total_solar) * 100)

    soc_changes = np.abs(np.diff(soc_arr[:-1]))
    cycles = np.sum(soc_changes) / 2.0

    print("\n" + "-" * 70)
    print("  RESULTS")
    print("-" * 70)
    print(f"{'Metric':<35} {'No Optimisation':>18} {'Rule-Based':>18}")
    print("-" * 70)
    print(f"{'Grid import (kWh)':<35} {np.sum(grid_import_noop)*resolution_h:>17.1f} {np.sum(grid_import_opt)*resolution_h:>17.1f}")
    print(f"{'Grid export (kWh)':<35} {np.sum(grid_export_noop)*resolution_h:>17.1f} {np.sum(grid_export_opt)*resolution_h:>17.1f}")
    print(f"{'Peak demand (kW)':<35} {peak_noop:>17.1f} {peak_opt:>17.1f}")
    print(f"{'Peak reduction':<35} {'—':>18} {peak_reduction:>17.1f}%")
    print(f"{'Energy cost ($)':<35} {cost_noop:>17.2f} {cost_opt:>17.2f}")
    print(f"{'Export revenue ($)':<35} {export_revenue_noop:>17.2f} {export_revenue_opt:>17.2f}")
    print(f"{'Net cost ($)':<35} {net_cost_noop:>17.2f} {net_cost_opt:>17.2f}")
    print(f"{'Daily savings ($)':<35} {'—':>18} {net_cost_noop - net_cost_opt:>17.2f}")
    print(f"{'Self-consumption (%)':<35} {self_consumption_noop:>17.1f}% {self_consumption_opt:>17.1f}%")
    print(f"{'Battery cycles':<35} {'—':>18} {cycles:>17.2f}")
    print(f"{'Final SOC':<35} {'—':>18} {soc_arr[-1]*100:>17.1f}%")
    print("-" * 70)

    annual_savings = (net_cost_noop - net_cost_opt) * 365
    print(f"\nProjected annual savings: ${annual_savings:,.0f}")
    print(f"Battery investment ({n_homes}x $8,000): ${n_homes * 8000:,}")
    payback = (n_homes * 8000) / annual_savings if annual_savings > 0 else float("inf")
    print(f"Simple payback: {payback:.1f} years")

    print("\n[Demo complete]")


if __name__ == "__main__":
    run()
