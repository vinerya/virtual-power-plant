"""Regression tests for the rule-based stochastic fallback (SimpleStochasticRules)."""

from __future__ import annotations

import numpy as np

from vpp.optimization import SimpleStochasticRules, create_stochastic_problem


def _problem(seed: int = 0):
    np.random.seed(seed)
    # Cheap night, expensive evening: a clear arbitrage signal.
    prices = [0.05] * 8 + [0.15] * 8 + [0.30] * 8
    return create_stochastic_problem(
        {
            "base_prices": prices,
            "renewable_forecast": [0.0] * 24,
            "load_forecast": [500.0] * 24,
            "battery_capacity": 1000.0,
            "max_power": 250.0,
        },
        num_scenarios=20,
        uncertainty_config={"price_volatility": 0.1},
    )


def test_conservative_rules_dispatch_on_clear_price_spread():
    """The fallback used to compare a period's p10 price with the same period's
    p90 price, which can never trigger, so it always returned an all-idle plan."""
    result = SimpleStochasticRules().solve(_problem())

    power = result.solution["battery_power"]
    assert len(power) == 24
    # Charges in the cheap hours, discharges in the expensive ones.
    assert any(p > 0 for p in power[:8])
    assert any(p < 0 for p in power[16:])
    assert all(p >= 0 for p in power[:8])
    assert all(p <= 0 for p in power[16:])
    # Arbitrage on this spread earns money (negative cost).
    assert result.objective_value < 0


def test_conservative_rules_respect_soc_limits():
    result = SimpleStochasticRules().solve(_problem(seed=1))
    soc = result.solution["battery_soc"]
    assert all(0.1 <= s <= 0.95 for s in soc)
    assert all(abs(p) <= 250.0 for p in result.solution["battery_power"])
