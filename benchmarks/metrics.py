"""Standardised benchmark metrics for VPP evaluation.

Provides consistent metric computation across all scenarios, ensuring
fair comparison between optimisation methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BenchmarkMetrics:
    """Container for computed benchmark metrics."""

    scenario_name: str = ""
    method_name: str = ""
    values: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> float:
        return self.values[key]

    def __contains__(self, key: str) -> bool:
        return key in self.values

    def summary(self) -> str:
        lines = [f"Metrics for {self.method_name} on {self.scenario_name}:"]
        for k, v in sorted(self.values.items()):
            if isinstance(v, float):
                lines.append(f"  {k:>35}: {v:>12.4f}")
            else:
                lines.append(f"  {k:>35}: {v!s:>12}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metric computation functions
# ---------------------------------------------------------------------------

def compute_peak_reduction(load: np.ndarray, optimised_load: np.ndarray) -> float:
    """Peak reduction percentage."""
    original_peak = np.max(load)
    if original_peak == 0:
        return 0.0
    optimised_peak = np.max(optimised_load)
    return (1.0 - optimised_peak / original_peak) * 100.0


def compute_total_cost(power_kw: np.ndarray, prices: np.ndarray,
                       resolution_hours: float = 0.25) -> float:
    """Total energy cost in currency units."""
    return float(np.sum(power_kw * prices * resolution_hours))


def compute_self_consumption(generation: np.ndarray, load: np.ndarray) -> float:
    """Self-consumption percentage (locally consumed / total generated)."""
    total_gen = np.sum(generation)
    if total_gen == 0:
        return 0.0
    consumed = np.sum(np.minimum(generation, load))
    return (consumed / total_gen) * 100.0


def compute_renewable_utilization(available: np.ndarray, used: np.ndarray) -> float:
    """Renewable utilization percentage (used / available)."""
    total_available = np.sum(available)
    if total_available == 0:
        return 0.0
    return (np.sum(used) / total_available) * 100.0


def compute_battery_cycles(soc_trajectory: np.ndarray, capacity_kwh: float) -> float:
    """Estimate equivalent full cycles from SOC trajectory."""
    if capacity_kwh == 0:
        return 0.0
    soc_changes = np.abs(np.diff(soc_trajectory))
    total_energy = np.sum(soc_changes) * capacity_kwh
    return total_energy / (2.0 * capacity_kwh)  # full cycle = charge + discharge


def compute_frequency_rmse(freq: np.ndarray, nominal: float = 60.0) -> float:
    """RMSE of frequency deviation from nominal."""
    return float(np.sqrt(np.mean((freq - nominal) ** 2)))


def compute_v2g_utilization(discharged_kwh: np.ndarray,
                            available_capacity_kwh: np.ndarray) -> float:
    """V2G utilization — fraction of available capacity actually discharged."""
    total_available = np.sum(available_capacity_kwh)
    if total_available == 0:
        return 0.0
    return (np.sum(discharged_kwh) / total_available) * 100.0


def compute_departure_soc_compliance(actual_soc: np.ndarray,
                                     target_soc: np.ndarray) -> float:
    """Percentage of vehicles meeting departure SOC target."""
    n = len(actual_soc)
    if n == 0:
        return 100.0
    met = np.sum(actual_soc >= target_soc - 0.01)  # 1% tolerance
    return (met / n) * 100.0


def compute_curtailment(available: np.ndarray, used: np.ndarray,
                        resolution_hours: float = 0.25) -> float:
    """Total curtailed energy in MWh."""
    curtailed = np.clip(available - used, 0, None)
    return float(np.sum(curtailed) * resolution_hours)


def compute_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Annualised Sharpe ratio of trading returns."""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate
    mean_ret = np.mean(excess)
    std_ret = np.std(excess, ddof=1)
    if std_ret < 1e-9:
        return 0.0
    # Annualise assuming 15-min intervals -> 35040 periods/year
    return float(mean_ret / std_ret * np.sqrt(35040))


def compute_max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum drawdown as a percentage."""
    if len(equity_curve) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / np.where(peak > 0, peak, 1.0)
    return float(np.max(drawdown) * 100.0)


def compute_co2_reduction(baseline_emissions: np.ndarray,
                          optimised_emissions: np.ndarray) -> float:
    """CO2 reduction percentage."""
    total_baseline = np.sum(baseline_emissions)
    if total_baseline == 0:
        return 0.0
    total_optimised = np.sum(optimised_emissions)
    return (1.0 - total_optimised / total_baseline) * 100.0


def compute_uptime(served: np.ndarray, demanded: np.ndarray) -> float:
    """Uptime percentage — fraction of time steps where all demand is met."""
    n = len(served)
    if n == 0:
        return 100.0
    met = np.sum(np.abs(served - demanded) < 0.01 * demanded + 0.1)
    return (met / n) * 100.0


# ---------------------------------------------------------------------------
# Aggregate metric builder
# ---------------------------------------------------------------------------

def compute_scenario_metrics(
    scenario_name: str,
    method_name: str,
    results: dict[str, np.ndarray],
    resolution_minutes: int = 15,
) -> BenchmarkMetrics:
    """Compute all applicable metrics from simulation results.

    The `results` dict should contain keys like:
    - load, optimised_load, generation, solar, wind, prices
    - soc_trajectory, capacity_kwh
    - frequency, discharged_kwh, available_v2g_kwh
    - actual_departure_soc, target_departure_soc
    - returns, equity_curve
    - baseline_emissions, optimised_emissions
    - served, demanded
    """
    metrics: dict[str, float] = {}
    res_h = resolution_minutes / 60.0

    if "load" in results and "optimised_load" in results:
        metrics["peak_reduction_pct"] = compute_peak_reduction(
            results["load"], results["optimised_load"])

    if "power_kw" in results and "prices" in results:
        metrics["total_cost"] = compute_total_cost(
            results["power_kw"], results["prices"], res_h)

    if "generation" in results and "load" in results:
        metrics["self_consumption_pct"] = compute_self_consumption(
            results["generation"], results["load"])

    if "available_renewable" in results and "used_renewable" in results:
        metrics["renewable_utilization_pct"] = compute_renewable_utilization(
            results["available_renewable"], results["used_renewable"])

    if "soc_trajectory" in results and "capacity_kwh" in results:
        metrics["battery_cycles"] = compute_battery_cycles(
            results["soc_trajectory"], float(results["capacity_kwh"]))

    if "frequency" in results:
        metrics["frequency_rmse"] = compute_frequency_rmse(results["frequency"])

    if "discharged_kwh" in results and "available_v2g_kwh" in results:
        metrics["v2g_utilization_pct"] = compute_v2g_utilization(
            results["discharged_kwh"], results["available_v2g_kwh"])

    if "actual_departure_soc" in results and "target_departure_soc" in results:
        metrics["departure_soc_compliance"] = compute_departure_soc_compliance(
            results["actual_departure_soc"], results["target_departure_soc"])

    if "curtailment_available" in results and "curtailment_used" in results:
        metrics["curtailment_mwh"] = compute_curtailment(
            results["curtailment_available"], results["curtailment_used"], res_h)

    if "returns" in results:
        metrics["sharpe_ratio"] = compute_sharpe_ratio(results["returns"])

    if "equity_curve" in results:
        metrics["max_drawdown_pct"] = compute_max_drawdown(results["equity_curve"])

    if "baseline_emissions" in results and "optimised_emissions" in results:
        metrics["co2_reduction_pct"] = compute_co2_reduction(
            results["baseline_emissions"], results["optimised_emissions"])

    if "served" in results and "demanded" in results:
        metrics["uptime_pct"] = compute_uptime(results["served"], results["demanded"])

    if "solve_time_ms" in results:
        metrics["solve_time_ms"] = float(results["solve_time_ms"])

    return BenchmarkMetrics(
        scenario_name=scenario_name,
        method_name=method_name,
        values=metrics,
    )
