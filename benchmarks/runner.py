"""Benchmark runner — execute scenarios with multiple methods and compare.

Runs VPP optimisation strategies against standardised scenarios, collects
metrics, performs statistical comparison, and generates reports.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from benchmarks.datasets import DatasetRegistry
from benchmarks.scenarios import ScenarioRegistry, Scenario
from benchmarks.metrics import BenchmarkMetrics


# ---------------------------------------------------------------------------
# Method protocol
# ---------------------------------------------------------------------------

class BenchmarkMethod(Protocol):
    """Protocol for optimisation methods to be benchmarked."""

    @property
    def name(self) -> str: ...

    def solve(self, scenario: Scenario, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run the method on the scenario data and return result arrays."""
        ...


# ---------------------------------------------------------------------------
# Built-in baseline methods
# ---------------------------------------------------------------------------

class NoOpMethod:
    """Baseline: no optimisation, pass-through."""

    name = "no_op"

    def solve(self, scenario: Scenario, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        results: dict[str, np.ndarray] = {}
        if "load_kw" in data:
            results["load"] = data["load_kw"]
            results["optimised_load"] = data["load_kw"].copy()
        if "solar_kw" in data:
            results["generation"] = data["solar_kw"]
        if "price_per_kwh" in data:
            results["prices"] = data["price_per_kwh"]
            if "load_kw" in data:
                results["power_kw"] = data["load_kw"]
        return results


class RuleBasedPeakShaving:
    """Rule-based peak shaving: charge when solar excess, discharge when above threshold."""

    name = "rule_based_peak_shaving"

    def __init__(self, threshold_pct: float = 0.7) -> None:
        self._threshold_pct = threshold_pct

    def solve(self, scenario: Scenario, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        load = data.get("load_kw", np.zeros(1))
        solar = data.get("solar_kw", np.zeros_like(load))
        prices = data.get("price_per_kwh", np.ones_like(load) * 0.1)

        n = len(load)
        battery_power = np.zeros(n)  # positive=discharge, negative=charge
        soc = np.zeros(n + 1)

        # Battery params from scenario
        batt_specs = [r for r in scenario.resources if r.resource_type == "battery"]
        if batt_specs:
            total_power = sum(r.rated_power_kw * r.count for r in batt_specs)
            total_cap = sum(r.capacity_kwh * r.count for r in batt_specs)
            soc[0] = batt_specs[0].initial_soc
        else:
            total_power = 10.0
            total_cap = 40.0
            soc[0] = 0.5

        threshold = np.max(load) * self._threshold_pct
        resolution_h = 0.25  # 15 min

        for i in range(n):
            net = load[i] - solar[i]

            if net > threshold:
                # Discharge to shave peak
                needed = net - threshold
                discharge = min(needed, total_power, (soc[i] - 0.1) * total_cap / resolution_h)
                discharge = max(discharge, 0.0)
                battery_power[i] = discharge
                soc[i + 1] = soc[i] - (discharge * resolution_h / total_cap)
            elif net < 0:
                # Excess solar — charge
                charge = min(abs(net), total_power, (0.9 - soc[i]) * total_cap / resolution_h)
                charge = max(charge, 0.0)
                battery_power[i] = -charge
                soc[i + 1] = soc[i] + (charge * resolution_h / total_cap)
            else:
                soc[i + 1] = soc[i]

        optimised = load - solar - battery_power
        grid_import = np.clip(optimised, 0, None)

        return {
            "load": load,
            "optimised_load": grid_import,
            "generation": solar,
            "power_kw": grid_import,
            "prices": prices,
            "soc_trajectory": soc[:-1],
            "capacity_kwh": np.array(total_cap),
        }


class SimpleV2GScheduler:
    """Simple V2G scheduling: charge during cheapest slots, discharge during most expensive."""

    name = "simple_v2g"

    def solve(self, scenario: Scenario, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        prices = data.get("energy_prices", np.ones(96) * 0.1)
        n_vehicles = int(data.get("arrival_hour", np.array([0])).shape[0])

        arrivals = data.get("arrival_hour", np.ones(n_vehicles) * 8)
        departures = data.get("departure_hour", np.ones(n_vehicles) * 17)
        initial_soc = data.get("initial_soc", np.ones(n_vehicles) * 0.5)
        target_soc = data.get("target_soc", np.ones(n_vehicles) * 0.8)
        capacity = data.get("capacity_kwh", np.ones(n_vehicles) * 60.0)
        max_charge = data.get("max_charge_kw", np.ones(n_vehicles) * 7.4)
        v2g = data.get("v2g_capable", np.ones(n_vehicles))

        n_steps = len(prices)
        resolution_h = 0.25
        total_discharged = np.zeros(n_vehicles)
        total_available = np.zeros(n_vehicles)
        final_soc = initial_soc.copy()
        returns = np.zeros(n_steps)

        # Price ranking
        price_rank = np.argsort(prices)
        cheap_slots = set(price_rank[:n_steps // 3])
        expensive_slots = set(price_rank[-n_steps // 3:])

        for v in range(n_vehicles):
            soc = initial_soc[v]
            arr_step = int(arrivals[v] * 60 / 15)
            dep_step = int(departures[v] * 60 / 15)
            arr_step = max(0, min(arr_step, n_steps - 1))
            dep_step = max(arr_step + 1, min(dep_step, n_steps))

            for t in range(arr_step, dep_step):
                if t in cheap_slots and soc < 0.95:
                    # Charge
                    charge_kw = min(max_charge[v], (0.95 - soc) * capacity[v] / resolution_h)
                    soc += charge_kw * resolution_h / capacity[v]
                    returns[t] -= charge_kw * resolution_h * prices[t]
                elif t in expensive_slots and v2g[v] > 0 and soc > target_soc[v] + 0.1:
                    # Discharge
                    discharge_kw = min(max_charge[v] * 0.8,
                                       (soc - target_soc[v]) * capacity[v] / resolution_h)
                    discharge_kw = max(discharge_kw, 0.0)
                    soc -= discharge_kw * resolution_h / capacity[v]
                    total_discharged[v] += discharge_kw * resolution_h
                    returns[t] += discharge_kw * resolution_h * prices[t]

            total_available[v] = max(0, (initial_soc[v] - target_soc[v])) * capacity[v] if v2g[v] > 0 else 0
            final_soc[v] = soc

        return {
            "actual_departure_soc": final_soc,
            "target_departure_soc": target_soc,
            "discharged_kwh": total_discharged,
            "available_v2g_kwh": np.maximum(total_available, 0.01),
            "returns": returns,
            "equity_curve": np.cumsum(returns),
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Result of running a single benchmark."""
    scenario_name: str
    method_name: str
    metrics: BenchmarkMetrics
    solve_time_s: float
    seed: int
    raw_results: dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    """Run and compare benchmark scenarios across methods."""

    def __init__(self) -> None:
        self._results: list[BenchmarkResult] = []

    def run(
        self,
        scenario_name: str,
        method: BenchmarkMethod,
        seed: int = 42,
        store_raw: bool = False,
    ) -> BenchmarkResult:
        """Run a single scenario with one method."""
        scenario = ScenarioRegistry.get(scenario_name)
        dataset = DatasetRegistry.get(scenario.dataset_name)
        data = dataset.generate(seed=seed)

        t0 = time.perf_counter()
        raw = method.solve(scenario, data)
        solve_time = time.perf_counter() - t0

        # Inject solve time
        raw["solve_time_ms"] = np.array(solve_time * 1000)

        from benchmarks.metrics import compute_scenario_metrics
        metrics = compute_scenario_metrics(
            scenario_name, method.name, raw,
            resolution_minutes=dataset.spec().resolution_minutes,
        )

        result = BenchmarkResult(
            scenario_name=scenario_name,
            method_name=method.name,
            metrics=metrics,
            solve_time_s=solve_time,
            seed=seed,
            raw_results=raw if store_raw else {},
        )
        self._results.append(result)
        return result

    def compare(
        self,
        scenario_name: str,
        methods: list[BenchmarkMethod],
        seeds: list[int] | None = None,
    ) -> dict[str, Any]:
        """Run multiple methods on the same scenario and compare."""
        seeds = seeds or [42]
        comparison: dict[str, list[BenchmarkMetrics]] = {}

        for method in methods:
            runs = []
            for seed in seeds:
                result = self.run(scenario_name, method, seed=seed)
                runs.append(result.metrics)
            comparison[method.name] = runs

        # Aggregate and find best
        all_metrics: set[str] = set()
        for runs in comparison.values():
            for m in runs:
                all_metrics.update(m.values.keys())

        summary: dict[str, dict[str, Any]] = {}
        for metric in sorted(all_metrics):
            summary[metric] = {}
            for method_name, runs in comparison.items():
                values = [r.values.get(metric, float("nan")) for r in runs]
                valid = [v for v in values if not np.isnan(v)]
                if valid:
                    summary[metric][method_name] = {
                        "mean": np.mean(valid),
                        "std": np.std(valid),
                        "min": np.min(valid),
                        "max": np.max(valid),
                        "n": len(valid),
                    }

        # Determine winner per metric
        best_per_metric: dict[str, str] = {}
        higher_is_better = {"self_consumption_pct", "renewable_utilization_pct",
                            "v2g_utilization_pct", "departure_soc_compliance",
                            "co2_reduction_pct", "uptime_pct", "sharpe_ratio",
                            "peak_reduction_pct", "win_rate", "load_served_pct",
                            "critical_load_served_pct"}
        for metric, method_stats in summary.items():
            if not method_stats:
                continue
            if metric in higher_is_better:
                best_per_metric[metric] = max(method_stats, key=lambda m: method_stats[m]["mean"])
            else:
                best_per_metric[metric] = min(method_stats, key=lambda m: method_stats[m]["mean"])

        return {
            "scenario": scenario_name,
            "methods": [m.name for m in methods],
            "seeds": seeds,
            "summary": summary,
            "best_per_metric": best_per_metric,
        }

    def generate_report(self, title: str = "VPP Benchmark Report") -> str:
        """Generate a markdown comparison report from all results."""
        lines = [f"# {title}\n"]

        # Group by scenario
        scenarios: dict[str, list[BenchmarkResult]] = {}
        for r in self._results:
            scenarios.setdefault(r.scenario_name, []).append(r)

        for scenario_name, results in scenarios.items():
            s = ScenarioRegistry.get(scenario_name)
            lines.append(f"## {scenario_name}\n")
            lines.append(f"_{s.description}_\n")

            # Collect all metrics
            all_metrics: set[str] = set()
            for r in results:
                all_metrics.update(r.metrics.values.keys())

            if not all_metrics:
                lines.append("_No metrics computed._\n")
                continue

            # Table header
            methods = sorted(set(r.method_name for r in results))
            header = "| Metric | " + " | ".join(methods) + " |"
            sep = "|" + "---|" * (len(methods) + 1)
            lines.append(header)
            lines.append(sep)

            for metric in sorted(all_metrics):
                row = f"| {metric} "
                for method in methods:
                    vals = [r.metrics.values.get(metric, float("nan"))
                            for r in results if r.method_name == method]
                    valid = [v for v in vals if not np.isnan(v)]
                    if valid:
                        mean = np.mean(valid)
                        row += f"| {mean:.4f} "
                    else:
                        row += "| — "
                row += "|"
                lines.append(row)

            lines.append("")

            # Solve times
            for method in methods:
                times = [r.solve_time_s for r in results if r.method_name == method]
                if times:
                    lines.append(f"- **{method}** solve time: {np.mean(times)*1000:.1f}ms (avg)")
            lines.append("")

        return "\n".join(lines)

    def get_results(self) -> list[BenchmarkResult]:
        return list(self._results)

    def clear(self) -> None:
        self._results.clear()
