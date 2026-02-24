"""Benchmark datasets — synthetic and standard test cases for VPP evaluation.

Each dataset produces time-series data (load, generation, prices, etc.) that
can be consumed by scenarios and the benchmark runner.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

@dataclass
class DatasetSpec:
    """Metadata describing a benchmark dataset."""

    name: str
    description: str
    duration_hours: int
    resolution_minutes: int
    features: list[str]
    source: str = "synthetic"
    tags: list[str] = field(default_factory=list)

    @property
    def n_steps(self) -> int:
        return self.duration_hours * 60 // self.resolution_minutes


class BenchmarkDataset(ABC):
    """Abstract benchmark dataset."""

    @abstractmethod
    def spec(self) -> DatasetSpec:
        ...

    @abstractmethod
    def generate(self, seed: int = 42) -> dict[str, np.ndarray]:
        """Return {feature_name: array_of_values} with shape (n_steps,)."""

    def summary(self) -> str:
        s = self.spec()
        lines = [
            f"Dataset: {s.name}",
            f"  {s.description}",
            f"  Duration : {s.duration_hours}h @ {s.resolution_minutes}min resolution ({s.n_steps} steps)",
            f"  Features : {', '.join(s.features)}",
            f"  Source   : {s.source}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class DatasetRegistry:
    """Registry of available benchmark datasets."""

    _datasets: dict[str, type[BenchmarkDataset]] = {}

    @classmethod
    def register(cls, name: str, dataset_cls: type[BenchmarkDataset]) -> None:
        cls._datasets[name] = dataset_cls

    @classmethod
    def get(cls, name: str) -> BenchmarkDataset:
        if name not in cls._datasets:
            raise KeyError(f"Unknown dataset: {name}. Available: {list(cls._datasets)}")
        return cls._datasets[name]()

    @classmethod
    def list_all(cls) -> list[str]:
        return sorted(cls._datasets.keys())


# ---------------------------------------------------------------------------
# Helper generators
# ---------------------------------------------------------------------------

def _daily_profile(n_steps: int, peak_hour: float = 14.0, base: float = 0.3,
                   peak: float = 1.0, resolution_min: int = 15) -> np.ndarray:
    """Create a smooth daily profile peaking at peak_hour."""
    hours = np.arange(n_steps) * resolution_min / 60.0
    daily_phase = hours % 24.0
    profile = base + (peak - base) * np.exp(-0.5 * ((daily_phase - peak_hour) / 3.0) ** 2)
    return profile


def _solar_profile(n_steps: int, rated_kw: float = 100.0,
                   resolution_min: int = 15, rng: np.random.RandomState | None = None) -> np.ndarray:
    """Simulate solar generation with cloud variability."""
    if rng is None:
        rng = np.random.RandomState(42)
    hours = np.arange(n_steps) * resolution_min / 60.0
    daily_hour = hours % 24.0
    # Bell curve around noon
    solar_base = np.clip(np.cos((daily_hour - 12.0) * np.pi / 12.0), 0, 1) ** 1.5
    # Cloud cover noise
    clouds = 1.0 - 0.3 * np.abs(rng.randn(n_steps))
    clouds = np.clip(clouds, 0.2, 1.0)
    return solar_base * clouds * rated_kw


def _wind_profile(n_steps: int, rated_kw: float = 200.0,
                  resolution_min: int = 15, rng: np.random.RandomState | None = None) -> np.ndarray:
    """Simulate wind generation with Weibull-like pattern."""
    if rng is None:
        rng = np.random.RandomState(42)
    # Autocorrelated wind speed
    wind_speed = np.zeros(n_steps)
    wind_speed[0] = 7.0
    for i in range(1, n_steps):
        wind_speed[i] = 0.95 * wind_speed[i - 1] + 0.05 * rng.weibull(2.0) * 14.0
    wind_speed = np.clip(wind_speed, 0, 25)
    # Cubic power curve with cut-in/cut-out
    power = np.where(
        (wind_speed >= 3) & (wind_speed <= 25),
        np.clip((wind_speed / 12.0) ** 3, 0, 1) * rated_kw,
        0.0,
    )
    return power


def _price_profile(n_steps: int, base_price: float = 50.0,
                   resolution_min: int = 15, rng: np.random.RandomState | None = None) -> np.ndarray:
    """Simulate energy prices with diurnal pattern and volatility."""
    if rng is None:
        rng = np.random.RandomState(42)
    hours = np.arange(n_steps) * resolution_min / 60.0
    daily_hour = hours % 24.0
    # Higher prices during peak hours (8-20)
    diurnal = base_price * (1.0 + 0.5 * np.sin((daily_hour - 6.0) * np.pi / 12.0))
    diurnal = np.clip(diurnal, base_price * 0.5, base_price * 2.5)
    # Random spikes
    noise = rng.lognormal(0, 0.15, n_steps)
    spikes = rng.choice([1.0, 1.0, 1.0, 1.0, 2.0, 3.0], size=n_steps, p=[0.8, 0.05, 0.05, 0.04, 0.04, 0.02])
    return diurnal * noise * spikes


# ---------------------------------------------------------------------------
# Built-in datasets
# ---------------------------------------------------------------------------

class IEEETestCase(BenchmarkDataset):
    """IEEE-inspired test case: residential load + solar + battery + grid prices.

    Simulates a 24h period at 15-min resolution typical of IEEE distribution
    test feeders with DER integration.
    """

    def spec(self) -> DatasetSpec:
        return DatasetSpec(
            name="ieee_test_case",
            description="IEEE-inspired residential DER test case (24h)",
            duration_hours=24,
            resolution_minutes=15,
            features=["load_kw", "solar_kw", "price_per_kwh", "temperature_c"],
            source="synthetic (IEEE-inspired)",
            tags=["residential", "solar", "battery", "peak_shaving"],
        )

    def generate(self, seed: int = 42) -> dict[str, np.ndarray]:
        rng = np.random.RandomState(seed)
        s = self.spec()
        n = s.n_steps

        load = _daily_profile(n, peak_hour=18.0, base=2.0, peak=8.0, resolution_min=s.resolution_minutes)
        load += rng.randn(n) * 0.3

        solar = _solar_profile(n, rated_kw=6.0, resolution_min=s.resolution_minutes, rng=rng)
        prices = _price_profile(n, base_price=0.12, resolution_min=s.resolution_minutes, rng=rng)

        hours = np.arange(n) * s.resolution_minutes / 60.0
        temp = 20.0 + 8.0 * np.sin((hours % 24 - 6) * np.pi / 12) + rng.randn(n) * 1.5

        return {
            "load_kw": np.clip(load, 0.5, None),
            "solar_kw": solar,
            "price_per_kwh": prices,
            "temperature_c": temp,
        }


class CaliforniaISO(BenchmarkDataset):
    """CAISO-inspired dataset: multi-day grid-scale with high renewable penetration.

    Simulates 7 days of CAISO-like conditions: duck curve, negative pricing,
    evening ramps, and frequency regulation needs.
    """

    def spec(self) -> DatasetSpec:
        return DatasetSpec(
            name="california_iso",
            description="CAISO-inspired grid-scale dataset (7 days, duck curve)",
            duration_hours=168,
            resolution_minutes=15,
            features=["net_load_mw", "solar_mw", "wind_mw", "lmp_per_mwh",
                       "regulation_up_price", "regulation_down_price", "frequency_hz"],
            source="synthetic (CAISO-inspired)",
            tags=["grid_scale", "duck_curve", "ancillary_services", "renewable"],
        )

    def generate(self, seed: int = 42) -> dict[str, np.ndarray]:
        rng = np.random.RandomState(seed)
        s = self.spec()
        n = s.n_steps

        gross_load = _daily_profile(n, peak_hour=16.0, base=18000, peak=35000, resolution_min=s.resolution_minutes)
        gross_load += rng.randn(n) * 500

        solar = _solar_profile(n, rated_kw=15000, resolution_min=s.resolution_minutes, rng=rng)
        wind = _wind_profile(n, rated_kw=8000, resolution_min=s.resolution_minutes, rng=rng)
        net_load = gross_load - solar - wind

        lmp = _price_profile(n, base_price=40.0, resolution_min=s.resolution_minutes, rng=rng)
        # Negative pricing during solar peak
        hours = np.arange(n) * s.resolution_minutes / 60.0
        midday = ((hours % 24) > 10) & ((hours % 24) < 14)
        lmp[midday] *= rng.uniform(-0.3, 0.5, midday.sum())

        reg_up = np.abs(rng.lognormal(2.5, 0.5, n))
        reg_down = np.abs(rng.lognormal(1.5, 0.6, n))

        freq = 60.0 + rng.randn(n) * 0.02
        freq = np.clip(freq, 59.9, 60.1)

        return {
            "net_load_mw": net_load,
            "solar_mw": solar,
            "wind_mw": wind,
            "lmp_per_mwh": lmp,
            "regulation_up_price": reg_up,
            "regulation_down_price": reg_down,
            "frequency_hz": freq,
        }


class EUGridData(BenchmarkDataset):
    """European grid dataset: cross-border trading, multi-market.

    Simulates 48h of European-style energy markets with day-ahead,
    intraday, and balancing prices across two zones.
    """

    def spec(self) -> DatasetSpec:
        return DatasetSpec(
            name="eu_grid_data",
            description="European multi-market cross-border dataset (48h)",
            duration_hours=48,
            resolution_minutes=15,
            features=["load_mw", "wind_mw", "solar_mw",
                       "da_price_zone_a", "da_price_zone_b",
                       "intraday_price", "balancing_price",
                       "cross_border_capacity_mw"],
            source="synthetic (EU-inspired)",
            tags=["european", "multi_market", "cross_border", "trading"],
        )

    def generate(self, seed: int = 42) -> dict[str, np.ndarray]:
        rng = np.random.RandomState(seed)
        s = self.spec()
        n = s.n_steps

        load = _daily_profile(n, peak_hour=12.0, base=5000, peak=9000, resolution_min=s.resolution_minutes)
        wind = _wind_profile(n, rated_kw=3000, resolution_min=s.resolution_minutes, rng=rng)
        solar = _solar_profile(n, rated_kw=2000, resolution_min=s.resolution_minutes, rng=rng)

        da_a = _price_profile(n, base_price=55.0, resolution_min=s.resolution_minutes, rng=rng)
        da_b = da_a * (1.0 + rng.randn(n) * 0.15)  # correlated but different

        intraday = da_a * (1.0 + rng.randn(n) * 0.2)
        balancing = da_a * (1.0 + rng.randn(n) * 0.5)  # more volatile

        capacity = 500 + rng.randn(n) * 50
        capacity = np.clip(capacity, 100, 800)

        return {
            "load_mw": load + rng.randn(n) * 200,
            "wind_mw": wind,
            "solar_mw": solar,
            "da_price_zone_a": da_a,
            "da_price_zone_b": np.clip(da_b, 0, None),
            "intraday_price": np.clip(intraday, 0, None),
            "balancing_price": balancing,
            "cross_border_capacity_mw": capacity,
        }


class EVFleetData(BenchmarkDataset):
    """EV fleet dataset: 50-vehicle parking garage with arrival/departure patterns.

    Generates per-vehicle arrival times, departure times, initial SOC,
    target SOC, and V2G capability for a full 24h cycle.
    """

    def __init__(self, n_vehicles: int = 50) -> None:
        self._n_vehicles = n_vehicles

    def spec(self) -> DatasetSpec:
        return DatasetSpec(
            name="ev_fleet_data",
            description=f"EV fleet dataset ({self._n_vehicles} vehicles, 24h)",
            duration_hours=24,
            resolution_minutes=15,
            features=["arrival_hour", "departure_hour", "initial_soc",
                       "target_soc", "capacity_kwh", "max_charge_kw",
                       "max_discharge_kw", "v2g_capable",
                       "energy_prices"],
            source="synthetic",
            tags=["ev_fleet", "v2g", "scheduling", "flexibility"],
        )

    def generate(self, seed: int = 42) -> dict[str, np.ndarray]:
        rng = np.random.RandomState(seed)
        s = self.spec()
        n_steps = s.n_steps
        n = self._n_vehicles

        # Arrival/departure patterns (office parking garage)
        arrivals = rng.normal(8.5, 1.0, n)  # arrive ~8:30
        arrivals = np.clip(arrivals, 6.0, 12.0)
        durations = rng.normal(8.0, 1.5, n)  # stay ~8h
        departures = arrivals + np.clip(durations, 4.0, 12.0)
        departures = np.clip(departures, arrivals + 2.0, 22.0)

        initial_soc = rng.uniform(0.2, 0.8, n)
        target_soc = rng.uniform(0.7, 0.95, n)
        target_soc = np.maximum(target_soc, initial_soc + 0.1)
        target_soc = np.clip(target_soc, 0, 1.0)

        capacity = rng.choice([40.0, 60.0, 75.0, 100.0], n, p=[0.2, 0.35, 0.3, 0.15])
        max_charge = np.where(capacity >= 75, 11.0, 7.4)
        max_discharge = np.where(rng.random(n) > 0.3, max_charge * 0.8, 0.0)
        v2g = (max_discharge > 0).astype(float)

        prices = _price_profile(n_steps, base_price=0.15, resolution_min=s.resolution_minutes, rng=rng)

        return {
            "arrival_hour": arrivals,
            "departure_hour": departures,
            "initial_soc": initial_soc,
            "target_soc": target_soc,
            "capacity_kwh": capacity,
            "max_charge_kw": max_charge,
            "max_discharge_kw": max_discharge,
            "v2g_capable": v2g,
            "energy_prices": prices,
        }


# ---------------------------------------------------------------------------
# Register built-in datasets
# ---------------------------------------------------------------------------

DatasetRegistry.register("ieee_test_case", IEEETestCase)
DatasetRegistry.register("california_iso", CaliforniaISO)
DatasetRegistry.register("eu_grid_data", EUGridData)
DatasetRegistry.register("ev_fleet_data", EVFleetData)
