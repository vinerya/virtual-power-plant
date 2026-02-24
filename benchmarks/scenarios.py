"""Benchmark scenarios — predefined VPP use cases for standardised evaluation.

Each scenario defines a dataset, resource fleet, optimisation strategy,
and evaluation metrics. Scenarios can be run individually or compared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

class ScenarioCategory(str, Enum):
    PEAK_SHAVING = "peak_shaving"
    FREQUENCY_RESPONSE = "frequency_response"
    V2G_ARBITRAGE = "v2g_arbitrage"
    MULTI_SITE = "multi_site_coordination"
    ISLANDING = "islanding"
    HIGH_RENEWABLE = "high_renewable_penetration"
    TRADING = "trading"


@dataclass
class ResourceSpec:
    """Specification of a resource for a benchmark scenario."""
    resource_type: str  # battery, solar, wind, ev, load
    count: int = 1
    rated_power_kw: float = 0.0
    capacity_kwh: float = 0.0
    initial_soc: float = 0.5
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scenario:
    """A complete benchmark scenario."""

    name: str
    category: ScenarioCategory
    description: str
    dataset_name: str
    resources: list[ResourceSpec]
    objective: str  # e.g. "minimize_cost", "maximize_self_consumption", "minimize_peak"
    constraints: dict[str, Any] = field(default_factory=dict)
    evaluation_metrics: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Scenario: {self.name} [{self.category.value}]",
            f"  {self.description}",
            f"  Dataset   : {self.dataset_name}",
            f"  Objective : {self.objective}",
            f"  Resources : {len(self.resources)} specs",
        ]
        for r in self.resources:
            lines.append(f"    - {r.count}x {r.resource_type} ({r.rated_power_kw} kW)")
        lines.append(f"  Metrics   : {', '.join(self.evaluation_metrics)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ScenarioRegistry:
    """Registry of available benchmark scenarios."""

    _scenarios: dict[str, Scenario] = {}

    @classmethod
    def register(cls, scenario: Scenario) -> None:
        cls._scenarios[scenario.name] = scenario

    @classmethod
    def get(cls, name: str) -> Scenario:
        if name not in cls._scenarios:
            raise KeyError(f"Unknown scenario: {name}. Available: {list(cls._scenarios)}")
        return cls._scenarios[name]

    @classmethod
    def list_all(cls) -> list[str]:
        return sorted(cls._scenarios.keys())

    @classmethod
    def by_category(cls, category: ScenarioCategory) -> list[Scenario]:
        return [s for s in cls._scenarios.values() if s.category == category]


# ---------------------------------------------------------------------------
# Built-in scenarios
# ---------------------------------------------------------------------------

# 1. Peak Shaving
ScenarioRegistry.register(Scenario(
    name="PEAK_SHAVING",
    category=ScenarioCategory.PEAK_SHAVING,
    description=(
        "Residential VPP with 10 homes: minimize demand charges by shaving "
        "evening peak using coordinated battery dispatch and solar self-consumption."
    ),
    dataset_name="ieee_test_case",
    resources=[
        ResourceSpec("solar", count=10, rated_power_kw=6.0),
        ResourceSpec("battery", count=10, rated_power_kw=5.0, capacity_kwh=13.5, initial_soc=0.5),
        ResourceSpec("load", count=10, rated_power_kw=8.0),
    ],
    objective="minimize_peak",
    constraints={"max_grid_import_kw": 50.0, "min_battery_soc": 0.1},
    evaluation_metrics=["peak_reduction_pct", "total_cost", "self_consumption_pct",
                        "battery_cycles", "solve_time_ms"],
    tags=["residential", "demand_charge", "solar_battery"],
))

# 2. Frequency Response
ScenarioRegistry.register(Scenario(
    name="FREQUENCY_RESPONSE",
    category=ScenarioCategory.FREQUENCY_RESPONSE,
    description=(
        "Grid-scale BESS providing primary frequency response: track frequency "
        "deviations and inject/absorb power within 200ms deadband."
    ),
    dataset_name="california_iso",
    resources=[
        ResourceSpec("battery", count=1, rated_power_kw=50000, capacity_kwh=200000, initial_soc=0.5),
    ],
    objective="minimize_frequency_deviation",
    constraints={"deadband_hz": 0.015, "response_time_s": 0.2, "soc_range": [0.2, 0.8]},
    evaluation_metrics=["frequency_rmse", "energy_throughput_mwh", "revenue_per_mw",
                        "soc_deviation", "regulation_score"],
    tags=["grid_scale", "ancillary_services", "frequency_regulation"],
))

# 3. V2G Arbitrage
ScenarioRegistry.register(Scenario(
    name="V2G_ARBITRAGE",
    category=ScenarioCategory.V2G_ARBITRAGE,
    description=(
        "50-EV parking garage performing energy arbitrage: charge during low-price "
        "periods, discharge to grid during peaks while meeting departure SOC targets."
    ),
    dataset_name="ev_fleet_data",
    resources=[
        ResourceSpec("ev", count=50, rated_power_kw=11.0, capacity_kwh=60.0, initial_soc=0.4,
                     parameters={"v2g_capable_pct": 0.7, "target_soc": 0.8}),
    ],
    objective="maximize_v2g_revenue",
    constraints={"departure_soc_min": 0.8, "max_cycles_per_day": 1.5,
                 "degradation_cost_per_kwh": 0.02},
    evaluation_metrics=["net_revenue", "v2g_utilization_pct", "departure_soc_compliance",
                        "peak_discharge_kw", "avg_battery_degradation"],
    tags=["ev_fleet", "v2g", "arbitrage", "scheduling"],
))

# 4. Multi-Site Coordination
ScenarioRegistry.register(Scenario(
    name="MULTI_SITE_COORDINATION",
    category=ScenarioCategory.MULTI_SITE,
    description=(
        "Coordinated dispatch across 3 geographically distributed sites: industrial, "
        "commercial, and residential. Cross-site power sharing with grid constraints."
    ),
    dataset_name="eu_grid_data",
    resources=[
        # Site A: Industrial
        ResourceSpec("solar", count=1, rated_power_kw=500, parameters={"site": "industrial"}),
        ResourceSpec("battery", count=1, rated_power_kw=250, capacity_kwh=1000, initial_soc=0.6,
                     parameters={"site": "industrial"}),
        ResourceSpec("load", count=1, rated_power_kw=800, parameters={"site": "industrial"}),
        # Site B: Commercial
        ResourceSpec("solar", count=1, rated_power_kw=200, parameters={"site": "commercial"}),
        ResourceSpec("wind", count=1, rated_power_kw=100, parameters={"site": "commercial"}),
        ResourceSpec("battery", count=1, rated_power_kw=100, capacity_kwh=400, initial_soc=0.5,
                     parameters={"site": "commercial"}),
        ResourceSpec("load", count=1, rated_power_kw=300, parameters={"site": "commercial"}),
        # Site C: Residential
        ResourceSpec("solar", count=10, rated_power_kw=6.0, parameters={"site": "residential"}),
        ResourceSpec("battery", count=10, rated_power_kw=5.0, capacity_kwh=13.5, initial_soc=0.5,
                     parameters={"site": "residential"}),
        ResourceSpec("load", count=10, rated_power_kw=8.0, parameters={"site": "residential"}),
    ],
    objective="minimize_total_cost",
    constraints={"interconnection_limit_kw": 200, "min_battery_soc": 0.1},
    evaluation_metrics=["total_cost", "cross_site_transfers_kwh", "peak_reduction_pct",
                        "renewable_utilization_pct", "solve_time_ms"],
    tags=["multi_site", "distributed", "cross_site_sharing"],
))

# 5. Islanding
ScenarioRegistry.register(Scenario(
    name="ISLANDING",
    category=ScenarioCategory.ISLANDING,
    description=(
        "Microgrid islanding resilience test: grid fault at hour 4, island for 8h, "
        "then reconnect. Grid-forming inverter must maintain frequency/voltage "
        "while shedding non-critical loads."
    ),
    dataset_name="ieee_test_case",
    resources=[
        ResourceSpec("solar", count=4, rated_power_kw=10.0),
        ResourceSpec("battery", count=2, rated_power_kw=25.0, capacity_kwh=100.0, initial_soc=0.9,
                     parameters={"grid_forming": True}),
        ResourceSpec("load", count=6, rated_power_kw=5.0,
                     parameters={"priorities": [1, 1, 3, 5, 7, 10]}),
    ],
    objective="maximize_uptime",
    constraints={"frequency_band_hz": [59.5, 60.5], "voltage_band_pu": [0.95, 1.05],
                 "island_start_h": 4, "island_duration_h": 8},
    evaluation_metrics=["uptime_pct", "load_served_pct", "critical_load_served_pct",
                        "frequency_deviation_max", "voltage_deviation_max",
                        "reconnection_success"],
    tags=["microgrid", "islanding", "resilience", "grid_forming"],
))

# 6. High Renewable Penetration
ScenarioRegistry.register(Scenario(
    name="HIGH_RENEWABLE_PENETRATION",
    category=ScenarioCategory.HIGH_RENEWABLE,
    description=(
        "CAISO-like grid with >70% renewable penetration: manage duck curve, "
        "curtailment, and evening ramp with storage and demand response."
    ),
    dataset_name="california_iso",
    resources=[
        ResourceSpec("solar", count=1, rated_power_kw=15000),
        ResourceSpec("wind", count=1, rated_power_kw=8000),
        ResourceSpec("battery", count=1, rated_power_kw=5000, capacity_kwh=20000, initial_soc=0.5),
        ResourceSpec("load", count=1, rated_power_kw=35000,
                     parameters={"dr_available_pct": 0.15}),
    ],
    objective="minimize_curtailment",
    constraints={"max_ramp_rate_mw_per_min": 100, "min_renewable_pct": 0.7},
    evaluation_metrics=["curtailment_mwh", "renewable_utilization_pct", "ramp_violations",
                        "storage_cycles", "total_cost", "co2_reduction_pct"],
    tags=["grid_scale", "high_renewable", "duck_curve", "curtailment"],
))

# 7. Multi-Market Trading
ScenarioRegistry.register(Scenario(
    name="MULTI_MARKET_TRADING",
    category=ScenarioCategory.TRADING,
    description=(
        "Multi-market trading strategy: simultaneous participation in day-ahead, "
        "intraday, and balancing markets with cross-border arbitrage opportunities."
    ),
    dataset_name="eu_grid_data",
    resources=[
        ResourceSpec("battery", count=1, rated_power_kw=100, capacity_kwh=400, initial_soc=0.5),
        ResourceSpec("solar", count=1, rated_power_kw=200),
        ResourceSpec("wind", count=1, rated_power_kw=150),
    ],
    objective="maximize_trading_profit",
    constraints={"max_position_mw": 100, "var_limit": 10000, "min_battery_soc": 0.15},
    evaluation_metrics=["net_profit", "sharpe_ratio", "max_drawdown",
                        "win_rate", "avg_spread_captured", "var_utilization"],
    tags=["trading", "multi_market", "arbitrage", "risk_management"],
))
