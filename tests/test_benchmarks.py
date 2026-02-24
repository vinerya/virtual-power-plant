"""Tests for the benchmarking suite."""

import sys
import os

import numpy as np
import pytest

# Ensure benchmarks package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.datasets import (
    BenchmarkDataset,
    DatasetRegistry,
    IEEETestCase,
    CaliforniaISO,
    EUGridData,
    EVFleetData,
)
from benchmarks.scenarios import Scenario, ScenarioRegistry, ScenarioCategory
from benchmarks.metrics import (
    BenchmarkMetrics,
    compute_peak_reduction,
    compute_total_cost,
    compute_self_consumption,
    compute_renewable_utilization,
    compute_battery_cycles,
    compute_frequency_rmse,
    compute_v2g_utilization,
    compute_departure_soc_compliance,
    compute_curtailment,
    compute_sharpe_ratio,
    compute_max_drawdown,
    compute_co2_reduction,
    compute_uptime,
    compute_scenario_metrics,
)
from benchmarks.runner import (
    BenchmarkRunner,
    NoOpMethod,
    RuleBasedPeakShaving,
    SimpleV2GScheduler,
    BenchmarkResult,
)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class TestDatasets:
    def test_registry_list(self):
        available = DatasetRegistry.list_all()
        assert "ieee_test_case" in available
        assert "california_iso" in available
        assert "eu_grid_data" in available
        assert "ev_fleet_data" in available

    def test_registry_get(self):
        ds = DatasetRegistry.get("ieee_test_case")
        assert isinstance(ds, IEEETestCase)

    def test_registry_unknown(self):
        with pytest.raises(KeyError, match="Unknown dataset"):
            DatasetRegistry.get("nonexistent")

    def test_ieee_test_case(self):
        ds = IEEETestCase()
        spec = ds.spec()
        assert spec.name == "ieee_test_case"
        assert spec.duration_hours == 24
        assert spec.resolution_minutes == 15
        assert spec.n_steps == 96

        data = ds.generate(seed=42)
        assert "load_kw" in data
        assert "solar_kw" in data
        assert "price_per_kwh" in data
        assert "temperature_c" in data
        assert data["load_kw"].shape == (96,)
        assert np.all(data["load_kw"] > 0)
        assert np.all(data["solar_kw"] >= 0)

    def test_california_iso(self):
        ds = CaliforniaISO()
        spec = ds.spec()
        assert spec.n_steps == 672  # 7 days * 96
        data = ds.generate(seed=123)
        assert "net_load_mw" in data
        assert "solar_mw" in data
        assert "lmp_per_mwh" in data
        assert "frequency_hz" in data
        assert data["frequency_hz"].shape == (672,)

    def test_eu_grid_data(self):
        ds = EUGridData()
        data = ds.generate(seed=42)
        assert "da_price_zone_a" in data
        assert "da_price_zone_b" in data
        assert "cross_border_capacity_mw" in data
        assert len(data["load_mw"]) == 192  # 48h * 4

    def test_ev_fleet_data(self):
        ds = EVFleetData(n_vehicles=20)
        data = ds.generate(seed=42)
        assert data["arrival_hour"].shape == (20,)
        assert data["departure_hour"].shape == (20,)
        assert np.all(data["departure_hour"] > data["arrival_hour"])
        assert np.all(data["initial_soc"] >= 0)
        assert np.all(data["initial_soc"] <= 1)
        assert data["energy_prices"].shape == (96,)

    def test_reproducibility(self):
        ds = IEEETestCase()
        d1 = ds.generate(seed=42)
        d2 = ds.generate(seed=42)
        np.testing.assert_array_equal(d1["load_kw"], d2["load_kw"])

    def test_different_seeds(self):
        ds = IEEETestCase()
        d1 = ds.generate(seed=42)
        d2 = ds.generate(seed=99)
        assert not np.array_equal(d1["load_kw"], d2["load_kw"])

    def test_dataset_summary(self):
        ds = IEEETestCase()
        summary = ds.summary()
        assert "ieee_test_case" in summary
        assert "96 steps" in summary


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

class TestScenarios:
    def test_registry_list(self):
        available = ScenarioRegistry.list_all()
        assert "PEAK_SHAVING" in available
        assert "FREQUENCY_RESPONSE" in available
        assert "V2G_ARBITRAGE" in available
        assert "ISLANDING" in available
        assert "HIGH_RENEWABLE_PENETRATION" in available
        assert "MULTI_MARKET_TRADING" in available

    def test_registry_get(self):
        s = ScenarioRegistry.get("PEAK_SHAVING")
        assert isinstance(s, Scenario)
        assert s.category == ScenarioCategory.PEAK_SHAVING

    def test_registry_by_category(self):
        trading = ScenarioRegistry.by_category(ScenarioCategory.TRADING)
        assert len(trading) >= 1
        assert all(s.category == ScenarioCategory.TRADING for s in trading)

    def test_scenario_summary(self):
        s = ScenarioRegistry.get("V2G_ARBITRAGE")
        summary = s.summary()
        assert "V2G_ARBITRAGE" in summary
        assert "ev" in summary.lower()

    def test_all_scenarios_have_valid_datasets(self):
        for name in ScenarioRegistry.list_all():
            s = ScenarioRegistry.get(name)
            ds = DatasetRegistry.get(s.dataset_name)
            assert ds is not None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_peak_reduction(self):
        load = np.array([10, 20, 30, 40, 50])
        optimised = np.array([10, 20, 25, 30, 35])
        pct = compute_peak_reduction(load, optimised)
        assert abs(pct - 30.0) < 1e-6  # 50->35 = 30% reduction

    def test_peak_reduction_no_change(self):
        load = np.array([10, 20, 30])
        assert compute_peak_reduction(load, load) == 0.0

    def test_total_cost(self):
        power = np.array([10.0, 20.0, 10.0])
        prices = np.array([0.1, 0.2, 0.1])
        cost = compute_total_cost(power, prices, resolution_hours=1.0)
        assert abs(cost - 6.0) < 1e-6  # 1+4+1 = 6

    def test_self_consumption(self):
        gen = np.array([10, 20, 5])
        load = np.array([8, 15, 10])
        pct = compute_self_consumption(gen, load)
        # consumed = min(10,8)+min(20,15)+min(5,10) = 8+15+5=28, total_gen=35
        assert abs(pct - 28 / 35 * 100) < 1e-6

    def test_renewable_utilization(self):
        available = np.array([100, 200, 150])
        used = np.array([90, 180, 150])
        pct = compute_renewable_utilization(available, used)
        assert abs(pct - 420 / 450 * 100) < 1e-6

    def test_battery_cycles(self):
        soc = np.array([0.5, 0.6, 0.7, 0.6, 0.5, 0.6])  # charge, charge, discharge, discharge, charge
        cycles = compute_battery_cycles(soc, capacity_kwh=100)
        # total changes: 0.1+0.1+0.1+0.1+0.1 = 0.5 * 100 = 50 kWh, /200 = 0.25 cycles
        assert abs(cycles - 0.25) < 1e-6

    def test_frequency_rmse(self):
        freq = np.array([60.0, 60.01, 59.99, 60.02, 59.98])
        rmse = compute_frequency_rmse(freq, nominal=60.0)
        assert rmse < 0.02

    def test_v2g_utilization(self):
        discharged = np.array([10, 20, 5])
        available = np.array([50, 50, 50])
        pct = compute_v2g_utilization(discharged, available)
        assert abs(pct - 35 / 150 * 100) < 1e-6

    def test_departure_soc_compliance(self):
        actual = np.array([0.8, 0.75, 0.9, 0.6])
        target = np.array([0.8, 0.8, 0.8, 0.8])
        pct = compute_departure_soc_compliance(actual, target)
        assert abs(pct - 50.0) < 1e-6  # 2 out of 4 meet target

    def test_curtailment(self):
        available = np.array([100, 200, 150])
        used = np.array([80, 150, 150])
        mwh = compute_curtailment(available, used, resolution_hours=1.0)
        assert abs(mwh - 70.0) < 1e-6  # 20+50+0

    def test_sharpe_ratio(self):
        returns = np.array([1, 1, 1, 1, 1], dtype=float)
        sharpe = compute_sharpe_ratio(returns)
        # All same return -> std=0, should return 0
        assert sharpe == 0.0

    def test_sharpe_ratio_with_variance(self):
        rng = np.random.RandomState(42)
        returns = rng.randn(1000) * 0.1 + 0.01
        sharpe = compute_sharpe_ratio(returns)
        assert sharpe != 0.0

    def test_max_drawdown(self):
        equity = np.array([100, 110, 105, 120, 90, 95])
        dd = compute_max_drawdown(equity)
        # Peak=120, trough=90, dd=30/120=25%
        assert abs(dd - 25.0) < 1e-6

    def test_co2_reduction(self):
        baseline = np.array([100, 200, 300])
        optimised = np.array([80, 150, 210])
        pct = compute_co2_reduction(baseline, optimised)
        # total: 600 -> 440, reduction = 160/600 = 26.67%
        assert abs(pct - 26.667) < 0.01

    def test_uptime(self):
        served = np.array([10, 20, 30, 0])
        demanded = np.array([10, 20, 30, 30])
        pct = compute_uptime(served, demanded)
        assert abs(pct - 75.0) < 1e-6

    def test_compute_scenario_metrics(self):
        results = {
            "load": np.array([10, 20, 30]),
            "optimised_load": np.array([10, 15, 25]),
        }
        m = compute_scenario_metrics("test", "method1", results)
        assert "peak_reduction_pct" in m.values

    def test_benchmark_metrics_container(self):
        m = BenchmarkMetrics(scenario_name="test", method_name="m1",
                             values={"cost": 100.0, "pct": 50.0})
        assert m["cost"] == 100.0
        assert "pct" in m
        summary = m.summary()
        assert "test" in summary


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class TestRunner:
    def test_noop_method(self):
        method = NoOpMethod()
        assert method.name == "no_op"
        scenario = ScenarioRegistry.get("PEAK_SHAVING")
        ds = DatasetRegistry.get(scenario.dataset_name)
        data = ds.generate(seed=42)
        results = method.solve(scenario, data)
        assert "load" in results

    def test_rule_based_peak_shaving(self):
        method = RuleBasedPeakShaving(threshold_pct=0.7)
        scenario = ScenarioRegistry.get("PEAK_SHAVING")
        ds = DatasetRegistry.get(scenario.dataset_name)
        data = ds.generate(seed=42)
        results = method.solve(scenario, data)
        assert "optimised_load" in results
        assert "soc_trajectory" in results
        # Optimised peak should be <= original peak
        assert np.max(results["optimised_load"]) <= np.max(results["load"]) + 0.01

    def test_simple_v2g_scheduler(self):
        method = SimpleV2GScheduler()
        scenario = ScenarioRegistry.get("V2G_ARBITRAGE")
        ds = DatasetRegistry.get(scenario.dataset_name)
        data = ds.generate(seed=42)
        results = method.solve(scenario, data)
        assert "actual_departure_soc" in results
        assert "returns" in results
        assert "equity_curve" in results

    def test_runner_run(self):
        runner = BenchmarkRunner()
        result = runner.run("PEAK_SHAVING", NoOpMethod(), seed=42)
        assert isinstance(result, BenchmarkResult)
        assert result.scenario_name == "PEAK_SHAVING"
        assert result.method_name == "no_op"
        assert result.solve_time_s > 0

    def test_runner_compare(self):
        runner = BenchmarkRunner()
        comparison = runner.compare(
            "PEAK_SHAVING",
            [NoOpMethod(), RuleBasedPeakShaving()],
            seeds=[42],
        )
        assert comparison["scenario"] == "PEAK_SHAVING"
        assert len(comparison["methods"]) == 2
        assert "summary" in comparison
        assert "best_per_metric" in comparison

    def test_runner_report(self):
        runner = BenchmarkRunner()
        runner.run("PEAK_SHAVING", NoOpMethod(), seed=42)
        runner.run("PEAK_SHAVING", RuleBasedPeakShaving(), seed=42)
        report = runner.generate_report()
        assert "PEAK_SHAVING" in report
        assert "no_op" in report
        assert "rule_based_peak_shaving" in report

    def test_runner_clear(self):
        runner = BenchmarkRunner()
        runner.run("PEAK_SHAVING", NoOpMethod())
        assert len(runner.get_results()) == 1
        runner.clear()
        assert len(runner.get_results()) == 0

    def test_runner_multiple_seeds(self):
        runner = BenchmarkRunner()
        comparison = runner.compare(
            "PEAK_SHAVING",
            [RuleBasedPeakShaving()],
            seeds=[42, 123, 456],
        )
        stats = comparison["summary"]
        for metric, method_data in stats.items():
            for method_name, data in method_data.items():
                assert data["n"] == 3  # 3 seeds
