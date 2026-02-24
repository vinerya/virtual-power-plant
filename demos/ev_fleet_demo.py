"""EV Fleet V2G Demo — 50-vehicle parking garage.

Demonstrates:
- Smart charging scheduling for a corporate parking garage
- V2G arbitrage: discharge to grid during peak prices
- Fleet flexibility assessment and ancillary service bidding
- Comparison: dumb charging vs smart V2G scheduling
"""

from __future__ import annotations

import numpy as np


def run() -> None:
    """Run the EV fleet V2G demo."""
    from benchmarks.datasets import EVFleetData

    print("=" * 70)
    print("  EV FLEET V2G DEMO — 50-Vehicle Parking Garage")
    print("=" * 70)

    dataset = EVFleetData(n_vehicles=50)
    data = dataset.generate(seed=42)

    n_vehicles = len(data["arrival_hour"])
    prices = data["energy_prices"]
    n_steps = len(prices)
    resolution_h = 0.25

    arrivals = data["arrival_hour"]
    departures = data["departure_hour"]
    initial_soc = data["initial_soc"]
    target_soc = data["target_soc"]
    capacity = data["capacity_kwh"]
    max_charge = data["max_charge_kw"]
    max_discharge = data["max_discharge_kw"]
    v2g = data["v2g_capable"]

    total_capacity = np.sum(capacity)
    v2g_count = int(np.sum(v2g))
    v2g_capacity = np.sum(capacity[v2g > 0])

    print(f"\nFleet: {n_vehicles} vehicles")
    print(f"  V2G capable    : {v2g_count} ({v2g_count/n_vehicles*100:.0f}%)")
    print(f"  Total capacity : {total_capacity:.0f} kWh")
    print(f"  V2G capacity   : {v2g_capacity:.0f} kWh")
    print(f"  Avg arrival    : {np.mean(arrivals):.1f}h")
    print(f"  Avg departure  : {np.mean(departures):.1f}h")
    print(f"  Avg initial SOC: {np.mean(initial_soc)*100:.1f}%")
    print(f"  Avg target SOC : {np.mean(target_soc)*100:.1f}%")
    print(f"\nPrice range: ${np.min(prices):.3f} - ${np.max(prices):.3f}/kWh")

    # --- Dumb charging: charge as fast as possible on arrival ---
    cost_dumb = 0.0
    peak_dumb = 0.0
    load_profile_dumb = np.zeros(n_steps)
    soc_final_dumb = np.zeros(n_vehicles)

    for v in range(n_vehicles):
        soc = initial_soc[v]
        arr = max(0, int(arrivals[v] * 4))  # 15-min steps
        dep = min(n_steps, int(departures[v] * 4))

        for t in range(arr, dep):
            if soc >= target_soc[v]:
                break
            charge_kw = min(max_charge[v], (target_soc[v] - soc) * capacity[v] / resolution_h)
            soc += charge_kw * resolution_h / capacity[v]
            load_profile_dumb[t] += charge_kw
            cost_dumb += charge_kw * resolution_h * prices[t]

        soc_final_dumb[v] = soc

    peak_dumb = np.max(load_profile_dumb)
    met_target_dumb = np.sum(soc_final_dumb >= target_soc - 0.01)

    # --- Smart V2G: charge cheap, discharge expensive ---
    cost_smart = 0.0
    revenue_smart = 0.0
    load_profile_smart = np.zeros(n_steps)
    discharge_profile = np.zeros(n_steps)
    soc_final_smart = np.zeros(n_vehicles)

    # Rank time slots by price
    price_rank = np.argsort(prices)
    n_cheap = n_steps // 3
    n_expensive = n_steps // 4
    cheap_set = set(price_rank[:n_cheap])
    expensive_set = set(price_rank[-n_expensive:])

    for v in range(n_vehicles):
        soc = initial_soc[v]
        arr = max(0, int(arrivals[v] * 4))
        dep = min(n_steps, int(departures[v] * 4))
        available_slots = dep - arr

        # Estimate energy needed
        energy_needed = max(0, (target_soc[v] - soc) * capacity[v])

        # Phase 1: plan charging in cheapest available slots
        vehicle_cheap = sorted(
            [t for t in range(arr, dep) if t in cheap_set],
            key=lambda t: prices[t],
        )

        charged = 0.0
        for t in vehicle_cheap:
            if charged >= energy_needed:
                break
            charge_kw = min(max_charge[v], (0.95 - soc) * capacity[v] / resolution_h)
            if charge_kw <= 0:
                break
            energy = charge_kw * resolution_h
            soc += energy / capacity[v]
            charged += energy
            load_profile_smart[t] += charge_kw
            cost_smart += energy * prices[t]

        # Phase 2: V2G discharge during expensive slots
        if v2g[v] > 0:
            vehicle_expensive = sorted(
                [t for t in range(arr, dep) if t in expensive_set],
                key=lambda t: -prices[t],
            )
            for t in vehicle_expensive:
                if soc <= target_soc[v] + 0.05:
                    break
                discharge_kw = min(
                    max_discharge[v],
                    (soc - target_soc[v]) * capacity[v] / resolution_h,
                )
                if discharge_kw <= 0:
                    break
                energy = discharge_kw * resolution_h
                soc -= energy / capacity[v]
                discharge_profile[t] += discharge_kw
                revenue_smart += energy * prices[t]

        soc_final_smart[v] = soc

    peak_smart = np.max(load_profile_smart)
    met_target_smart = np.sum(soc_final_smart >= target_soc - 0.01)
    net_cost_smart = cost_smart - revenue_smart
    peak_discharge = np.max(discharge_profile)

    print("\n" + "-" * 70)
    print("  RESULTS")
    print("-" * 70)
    print(f"{'Metric':<35} {'Dumb Charging':>18} {'Smart V2G':>18}")
    print("-" * 70)
    print(f"{'Charging cost ($)':<35} {cost_dumb:>17.2f} {cost_smart:>17.2f}")
    print(f"{'V2G revenue ($)':<35} {'—':>18} {revenue_smart:>17.2f}")
    print(f"{'Net cost ($)':<35} {cost_dumb:>17.2f} {net_cost_smart:>17.2f}")
    print(f"{'Savings ($)':<35} {'—':>18} {cost_dumb - net_cost_smart:>17.2f}")
    print(f"{'Peak charging load (kW)':<35} {peak_dumb:>17.1f} {peak_smart:>17.1f}")
    print(f"{'Peak V2G discharge (kW)':<35} {'—':>18} {peak_discharge:>17.1f}")
    print(f"{'Vehicles meeting SOC target':<35} {int(met_target_dumb):>17d} {int(met_target_smart):>17d}")
    print(f"{'SOC compliance (%)':<35} {met_target_dumb/n_vehicles*100:>17.1f}% {met_target_smart/n_vehicles*100:>17.1f}%")
    print("-" * 70)

    # Flexibility assessment
    print("\n--- Fleet Flexibility Window ---")
    active = np.zeros(n_steps)
    flex_up = np.zeros(n_steps)  # additional charge capacity
    flex_down = np.zeros(n_steps)  # discharge capacity

    for v in range(n_vehicles):
        arr = max(0, int(arrivals[v] * 4))
        dep = min(n_steps, int(departures[v] * 4))
        for t in range(arr, dep):
            active[t] += 1
            flex_up[t] += max_charge[v]
            if v2g[v] > 0:
                flex_down[t] += max_discharge[v]

    peak_active = int(np.max(active))
    peak_flex_up = np.max(flex_up)
    peak_flex_down = np.max(flex_down)

    print(f"  Peak connected EVs     : {peak_active}")
    print(f"  Peak charge flex (up)  : {peak_flex_up:.0f} kW")
    print(f"  Peak discharge flex    : {peak_flex_down:.0f} kW")
    print(f"  Avg connected (9-17h)  : {np.mean(active[36:68]):.0f} EVs")

    annual_savings = (cost_dumb - net_cost_smart) * 365
    print(f"\nProjected annual savings: ${annual_savings:,.0f}")
    print(f"Per-vehicle annual savings: ${annual_savings/n_vehicles:,.0f}")

    print("\n[Demo complete]")


if __name__ == "__main__":
    run()
