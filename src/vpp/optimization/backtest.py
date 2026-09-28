"""
Backtest harness for the MPC controller — Milestone 3.

Drives an :class:`MPCController` over a historical window. Each tick:

  1. Calls ``forecast_fn(timestamp, horizon_steps)`` to obtain the forecast
     dict that the controller sees (perfect-foresight, naive, or noisy).
  2. Asks the controller for a decision.
  3. Applies the first-step decision to a simulated battery model using the
     *ground-truth* price/load/solar series and integrates SOC.
  4. Records realized cost, SOC trajectory, solver time, and fallback usage.

The simulator is intentionally tiny — it mirrors the formulation's SOC dynamics
so that perfect-foresight MPC == offline optimum (within solver tolerance).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from .mpc import MPCController, MPCDecision, MPCStep

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class BacktestConfig:
    start: datetime
    end: datetime
    interval_minutes: int


@dataclass
class BacktestResult:
    realized_dispatch: list[dict[str, Any]] = field(default_factory=list)
    soc_trajectory: list[float] = field(default_factory=list)
    realized_cost: float = 0.0
    cumulative_solve_time_ms: float = 0.0
    cumulative_solver_iterations: int = 0
    """Sum of MPCDecision.solver_iterations across ticks where it was
    reported (appsi HiGHS only). Deterministic given the same inputs --
    unlike cumulative_solve_time_ms, safe to compare across runs without
    being sensitive to system load."""
    fallback_count: int = 0
    wall_time_s: float = 0.0


def run_backtest(
    controller: MPCController,
    config: BacktestConfig,
    price_series: list[float],
    load_series: list[float],
    solar_series: list[float],
    forecast_fn: Callable[[datetime, int], dict[str, list[float]]],
) -> BacktestResult:
    """Run a closed-loop MPC backtest.

    Args:
        controller: pre-configured :class:`MPCController`.
        config: backtest window + tick cadence.
        price_series: ground-truth prices, one per tick (length >= num_ticks).
        load_series, solar_series: ground-truth load/solar, same length.
        forecast_fn: ``(now, horizon_steps) -> {'prices': [...], 'load_kw': [...],
            'solar_kw': [...]}``. The harness does not enforce that forecasts
            equal ground truth — that's how we test forecast-error sensitivity.

    Returns:
        :class:`BacktestResult` with per-step decisions and cumulative stats.
    """
    bp = controller.battery_params
    cfg = controller.config
    dt_h = cfg.interval_minutes / 60.0
    cap = float(bp["battery_capacity_kwh"])
    soc_min = float(bp["soc_min"]) * cap
    soc_max = float(bp["soc_max"]) * cap
    eta_c = float(bp["eta_charge"])
    eta_d = float(bp["eta_discharge"])

    duration = config.end - config.start
    total_minutes = duration.total_seconds() / 60.0
    num_ticks = int(total_minutes // config.interval_minutes)

    if num_ticks <= 0:
        return BacktestResult()

    if not (
        len(price_series) >= num_ticks
        and len(load_series) >= num_ticks
        and len(solar_series) >= num_ticks
    ):
        raise ValueError(f"price/load/solar series must each cover >= {num_ticks} ticks")

    # Initial SOC is whatever the controller's battery_params says.
    soc = float(bp.get("soc_init", 0.5)) * cap

    result = BacktestResult()
    wall_start = time.time()

    for k in range(num_ticks):
        now = config.start + timedelta(minutes=k * config.interval_minutes)
        horizon = cfg.horizon_steps
        forecast = forecast_fn(now, horizon)

        step = MPCStep(
            timestamp=now,
            soc_init=soc / cap,  # back to fraction
            forecast=forecast,
        )
        decision: MPCDecision = controller.step(step)
        if decision.fallback_used:
            result.fallback_count += 1
        result.cumulative_solve_time_ms += decision.solve_time_ms
        if decision.solver_iterations is not None:
            result.cumulative_solver_iterations += decision.solver_iterations

        # Apply first-step decision against ground-truth, integrate SOC.
        p_chg = max(0.0, decision.p_charge_kw)
        p_dis = max(0.0, decision.p_discharge_kw)
        # Clip to physical SOC bounds (rule-based fallbacks may overshoot).
        new_soc = soc + eta_c * p_chg * dt_h - p_dis * dt_h / eta_d
        if new_soc > soc_max:
            # Trim charge.
            excess = new_soc - soc_max
            p_chg = max(0.0, p_chg - excess / (eta_c * dt_h))
            new_soc = soc + eta_c * p_chg * dt_h - p_dis * dt_h / eta_d
        if new_soc < soc_min:
            excess = soc_min - new_soc
            p_dis = max(0.0, p_dis - excess * eta_d / dt_h)
            new_soc = soc + eta_c * p_chg * dt_h - p_dis * dt_h / eta_d
        new_soc = min(soc_max, max(soc_min, new_soc))

        realized_price = float(price_series[k])
        net_kw = p_chg - p_dis
        step_cost = realized_price * net_kw * dt_h
        result.realized_cost += step_cost

        result.realized_dispatch.append(
            {
                "timestamp": now,
                "p_charge_kw": p_chg,
                "p_discharge_kw": p_dis,
                "is_charging": decision.is_charging,
                "soc_kwh": new_soc,
                "realized_price": realized_price,
                "step_cost": step_cost,
                "solve_time_ms": decision.solve_time_ms,
                "fallback_used": decision.fallback_used,
            }
        )
        result.soc_trajectory.append(new_soc)
        soc = new_soc

    result.wall_time_s = time.time() - wall_start
    return result


__all__ = ["BacktestConfig", "BacktestResult", "run_backtest"]
