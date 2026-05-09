"""
Model Predictive Control (MPC) wrapper — Milestone 3.

Receding-horizon controller that wraps the deterministic / stochastic dispatch
formulations from M1/M2. Each ``step()`` call:

  1. Builds the dispatch model with the supplied forecast (or scenarios).
  2. Forwards any caller-supplied ``additional_objective_terms`` and
     ``additional_constraint_builders`` into the formulation hooks. This is the
     plumbing that lets downstream tracks (degradation, tariff, multi-resource)
     compose into the MPC loop without modifying the formulations.
  3. Optionally seeds the MILP with a warm-start hint derived from the previous
     tick's solution (shifted forward by one step).
  4. Solves with a HiGHS time limit. If the solver fails or times out and
     ``fallback_on_failure`` is True, falls back to the rule-based dispatcher.
  5. Returns only the first-step decision (``MPCDecision``) plus the full plan
     for diagnostics.

The controller does NOT advance physical SOC itself — callers (or the
backtest harness) provide a fresh ``soc_init`` from telemetry on each tick.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence

from .formulations.dispatch import build_battery_dispatch_model
from .formulations.stochastic import build_stochastic_dispatch_model
from .solvers.pyomo_plugin import (
    SimpleBatteryDispatchRules,
    _try_import_pyomo,
)
from .base import (
    OptimizationProblem,
    OptimizationStatus,
)


# Try Scenario import only for typing; not strictly required at runtime since
# the stochastic builder accepts dicts too.
try:
    from .stochastic import Scenario  # type: ignore
except Exception:  # pragma: no cover
    Scenario = Any  # type: ignore


@dataclass
class MPCConfig:
    horizon_steps: int
    interval_minutes: int
    warm_start: bool = True
    stochastic: bool = False
    num_scenarios: int = 1
    cvar_alpha: float = 0.95
    cvar_lambda: float = 0.0
    solver_timeout_ms: int = 5_000
    fallback_on_failure: bool = True


@dataclass
class MPCStep:
    """Inputs to one MPC tick."""
    timestamp: datetime
    soc_init: float
    forecast: Dict[str, List[float]] = field(default_factory=dict)
    scenarios: Optional[List[Any]] = None
    additional_objective_terms: List[Callable] = field(default_factory=list)
    additional_constraint_builders: List[Callable] = field(default_factory=list)


@dataclass
class MPCDecision:
    """Output of one MPC tick — only the first-step decision is binding."""
    timestamp: datetime
    p_charge_kw: float
    p_discharge_kw: float
    is_charging: bool
    expected_cost_remaining: float
    solve_time_ms: float
    fallback_used: bool
    full_horizon_plan: Dict[str, Any] = field(default_factory=dict)


class MPCController:
    """Receding-horizon MPC driver around the M1/M2 dispatch formulations."""

    def __init__(self, config: MPCConfig, battery_params: Dict[str, Any]) -> None:
        self.config = config
        self.battery_params = dict(battery_params)
        self._pyo, self._solver_factory = _try_import_pyomo()
        # Warm-start cache: list of dicts {p_charge, p_discharge, is_charging, soc}
        # taken from the previous solve, shifted by one step on use.
        self._last_plan: Optional[Dict[str, List[float]]] = None
        self._fallback = SimpleBatteryDispatchRules()

    # ------------------------------------------------------------------ public

    def reset(self) -> None:
        self._last_plan = None

    @property
    def warm_start_size(self) -> int:
        if self._last_plan is None:
            return 0
        return len(self._last_plan.get("is_charging", []))

    def step(self, mpc_step: MPCStep) -> MPCDecision:
        t_start = time.time()
        cfg = self.config
        H = cfg.horizon_steps
        dt_h = cfg.interval_minutes / 60.0

        # Build params for the formulation.
        params = dict(self.battery_params)
        params["soc_init"] = float(mpc_step.soc_init)
        params["dt_hours"] = dt_h

        # If terminal SOC isn't pinned by caller, default to soc_init for stability.
        params.setdefault("terminal_soc", params["soc_init"])

        prices = list(mpc_step.forecast.get("prices", []))
        load = list(mpc_step.forecast.get("load_kw", []))
        solar = list(mpc_step.forecast.get("solar_kw", []))

        if cfg.stochastic and mpc_step.scenarios:
            return self._step_stochastic(mpc_step, params, t_start)

        # Deterministic path
        if not prices:
            raise ValueError("MPCStep.forecast['prices'] is required (deterministic mode)")
        if len(prices) < H:
            # Pad with last value; we keep full H to keep the warm-start shapes stable.
            prices = prices + [prices[-1]] * (H - len(prices))
        prices = prices[:H]
        params["prices"] = prices

        # Try optimal solve.
        decision = self._solve_deterministic(
            params=params,
            mpc_step=mpc_step,
            t_start=t_start,
            load=load,
            solar=solar,
        )
        return decision

    # ------------------------------------------------------------------ helpers

    def _solve_deterministic(
        self,
        params: Dict[str, Any],
        mpc_step: MPCStep,
        t_start: float,
        load: List[float],
        solar: List[float],
    ) -> MPCDecision:
        cfg = self.config

        if self._pyo is None or self._solver_factory is None:
            return self._do_fallback(mpc_step, params, t_start, reason="no_pyomo")

        # Build model with hooks
        try:
            model = build_battery_dispatch_model(
                params,
                objective_terms=mpc_step.additional_objective_terms or None,
                constraint_builders=mpc_step.additional_constraint_builders or None,
            )
            # Optional load/solar exposure as M2-style vars on the model.
            self._attach_load_solar(model, load, solar, len(params["prices"]))
        except Exception:
            return self._do_fallback(mpc_step, params, t_start, reason="build_failed")

        # Warm start
        if cfg.warm_start and self._last_plan is not None:
            self._apply_warm_start(model, len(params["prices"]))

        # Solve
        time_limit_s = max(0.001, cfg.solver_timeout_ms / 1000.0)
        try:
            solver = self._solver_factory(time_limit_s)
            results = solver.solve(model)
        except Exception:
            return self._do_fallback(mpc_step, params, t_start, reason="solve_exc")

        # Status check
        from .solvers.pyomo_plugin import PyomoPlugin
        status, _ = PyomoPlugin._extract_status(results)
        if status != OptimizationStatus.SUCCESS:
            if cfg.fallback_on_failure:
                return self._do_fallback(
                    mpc_step, params, t_start, reason=f"status={status.value}"
                )
            return MPCDecision(
                timestamp=mpc_step.timestamp,
                p_charge_kw=0.0,
                p_discharge_kw=0.0,
                is_charging=False,
                expected_cost_remaining=float("inf"),
                solve_time_ms=(time.time() - t_start) * 1000.0,
                fallback_used=False,
                full_horizon_plan={"status": status.value},
            )

        pyo = self._pyo
        T = len(params["prices"])
        try:
            p_chg = [float(pyo.value(model.p_charge[t])) for t in range(T)]
            p_dis = [float(pyo.value(model.p_discharge[t])) for t in range(T)]
            is_chg = [int(round(float(pyo.value(model.is_charging[t])))) for t in range(T)]
            soc = [float(pyo.value(model.soc[t])) for t in range(T)]
            obj = float(pyo.value(model.cost))
        except Exception:
            return self._do_fallback(mpc_step, params, t_start, reason="extract_failed")

        # Cache plan for next warm start.
        self._last_plan = {
            "p_charge": p_chg,
            "p_discharge": p_dis,
            "is_charging": [float(x) for x in is_chg],
            "soc": soc,
        }

        return MPCDecision(
            timestamp=mpc_step.timestamp,
            p_charge_kw=p_chg[0],
            p_discharge_kw=p_dis[0],
            is_charging=bool(is_chg[0]),
            expected_cost_remaining=obj,
            solve_time_ms=(time.time() - t_start) * 1000.0,
            fallback_used=False,
            full_horizon_plan={
                "p_charge": p_chg,
                "p_discharge": p_dis,
                "is_charging": is_chg,
                "soc": soc,
                "objective": obj,
                "horizon": T,
            },
        )

    def _step_stochastic(
        self,
        mpc_step: MPCStep,
        params: Dict[str, Any],
        t_start: float,
    ) -> MPCDecision:
        cfg = self.config
        scenarios = list(mpc_step.scenarios or [])
        if not scenarios:
            raise ValueError("stochastic mode requires mpc_step.scenarios")

        # Truncate / pad scenarios to horizon length H.
        H = cfg.horizon_steps
        norm_scen = []
        for sc in scenarios:
            data = getattr(sc, "data", None)
            if data is None and isinstance(sc, dict):
                data = sc.get("data") or sc
            prices = (data or {}).get("prices") if isinstance(data, dict) else None
            if prices is None and isinstance(sc, dict):
                prices = sc.get("prices")
            if prices is None:
                raise ValueError("scenario missing prices")
            prices = list(prices)
            if len(prices) < H:
                prices = prices + [prices[-1]] * (H - len(prices))
            prices = prices[:H]
            prob = getattr(sc, "probability", None)
            if prob is None and isinstance(sc, dict):
                prob = sc.get("probability", 1.0 / len(scenarios))
            norm_scen.append({"probability": float(prob), "prices": prices})

        params["cvar_alpha"] = cfg.cvar_alpha
        params["cvar_lambda"] = cfg.cvar_lambda

        if self._pyo is None or self._solver_factory is None:
            return self._do_fallback(mpc_step, params, t_start, reason="no_pyomo")

        try:
            model = build_stochastic_dispatch_model(params, norm_scen)
        except Exception:
            return self._do_fallback(mpc_step, params, t_start, reason="stoch_build_failed")

        time_limit_s = max(0.001, cfg.solver_timeout_ms / 1000.0)
        try:
            solver = self._solver_factory(time_limit_s)
            results = solver.solve(model)
        except Exception:
            return self._do_fallback(mpc_step, params, t_start, reason="solve_exc")

        from .solvers.pyomo_plugin import PyomoPlugin
        status, _ = PyomoPlugin._extract_status(results)
        if status != OptimizationStatus.SUCCESS:
            if cfg.fallback_on_failure:
                return self._do_fallback(
                    mpc_step, params, t_start, reason=f"status={status.value}"
                )

        pyo = self._pyo
        # Stage-1 (here-and-now) — non-anticipativity ties all scenarios at t=0.
        p_chg0 = float(pyo.value(model.p_charge[0, 0]))
        p_dis0 = float(pyo.value(model.p_discharge[0, 0]))
        is_chg0 = int(round(float(pyo.value(model.is_charging[0, 0]))))
        obj = float(pyo.value(model.total_cost))

        return MPCDecision(
            timestamp=mpc_step.timestamp,
            p_charge_kw=p_chg0,
            p_discharge_kw=p_dis0,
            is_charging=bool(is_chg0),
            expected_cost_remaining=obj,
            solve_time_ms=(time.time() - t_start) * 1000.0,
            fallback_used=False,
            full_horizon_plan={
                "expected_cost": float(pyo.value(model.expected_cost)),
                "scenarios": len(norm_scen),
                "horizon": cfg.horizon_steps,
            },
        )

    def _do_fallback(
        self,
        mpc_step: MPCStep,
        params: Dict[str, Any],
        t_start: float,
        reason: str,
    ) -> MPCDecision:
        # Prepare a one-step problem for the rule-based fallback (it uses the
        # full horizon, which is still cheap; we pass the available forecast or
        # a single-step degenerate horizon).
        prices = list(params.get("prices") or [])
        if not prices:
            # Stochastic mode or no forecast — use scenario-0 prices if available.
            prices = [0.0] * max(1, self.config.horizon_steps)
        fb_params = dict(params)
        fb_params["prices"] = prices
        fb_params.setdefault("dt_hours", self.config.interval_minutes / 60.0)
        fb_params.setdefault("terminal_soc", params.get("soc_init", 0.5))

        problem = OptimizationProblem(
            variables={},
            objectives=[{"type": "minimize_cost"}],
            constraints=[],
            parameters=fb_params,
            metadata={"type": "battery_dispatch"},
        )
        result = self._fallback.solve(problem)
        sol = result.solution or {}
        p_chg = (sol.get("p_charge") or [0.0])[0]
        p_dis = (sol.get("p_discharge") or [0.0])[0]
        return MPCDecision(
            timestamp=mpc_step.timestamp,
            p_charge_kw=float(p_chg),
            p_discharge_kw=float(p_dis),
            is_charging=bool(p_chg > p_dis),
            expected_cost_remaining=float(result.objective_value or 0.0),
            solve_time_ms=(time.time() - t_start) * 1000.0,
            fallback_used=True,
            full_horizon_plan={
                "fallback_reason": reason,
                "p_charge": sol.get("p_charge", []),
                "p_discharge": sol.get("p_discharge", []),
                "soc": sol.get("soc", []),
            },
        )

    def _apply_warm_start(self, model, T: int) -> None:
        """Seed binary is_charging from the previous plan, shifted forward by 1 step.

        HiGHS appsi accepts MIP starts via the variable ``.value`` attribute set
        before calling ``solve()``. Continuous vars are also seeded — the solver
        treats these as hints and ignores them if infeasible.
        """
        if self._last_plan is None:
            return
        prev_chg = self._last_plan.get("p_charge", [])
        prev_dis = self._last_plan.get("p_discharge", [])
        prev_bin = self._last_plan.get("is_charging", [])
        prev_soc = self._last_plan.get("soc", [])
        if not prev_bin:
            return
        # Shift by 1; for the tail, replicate the last decision.
        # Helper to clip into [lb, ub] respecting Var bounds (avoids Pyomo
        # warnings when prior solutions sit on boundary with float jitter).
        def _clip(var, val):
            try:
                lb, ub = var.bounds
                if lb is not None and val < lb:
                    val = lb
                if ub is not None and val > ub:
                    val = ub
            except Exception:
                pass
            return float(val)

        for t in range(T):
            src = min(t + 1, len(prev_bin) - 1)
            try:
                model.is_charging[t].value = _clip(model.is_charging[t], prev_bin[src])
                if src < len(prev_chg):
                    model.p_charge[t].value = _clip(model.p_charge[t], prev_chg[src])
                if src < len(prev_dis):
                    model.p_discharge[t].value = _clip(model.p_discharge[t], prev_dis[src])
                if src < len(prev_soc):
                    model.soc[t].value = _clip(model.soc[t], prev_soc[src])
            except Exception:
                # Pyomo doesn't error on stale indices, but be defensive.
                continue

    def _attach_load_solar(self, model, load: List[float], solar: List[float], T: int) -> None:
        """Expose ``m.load_kw[t]`` and ``m.solar_kw[t]`` Params if forecast supplied.

        These are part of the M2 stable surface; downstream hooks may reference
        them. If the caller didn't supply load/solar we silently skip.
        """
        try:
            import pyomo.environ as pyo
            if load and len(load) >= T and not hasattr(model, "load_kw"):
                model.load_kw = pyo.Param(
                    model.T,
                    initialize={t: float(load[t]) for t in range(T)},
                    mutable=True,
                )
            if solar and len(solar) >= T and not hasattr(model, "solar_kw"):
                model.solar_kw = pyo.Param(
                    model.T,
                    initialize={t: float(solar[t]) for t in range(T)},
                    mutable=True,
                )
        except Exception:
            pass


__all__ = [
    "MPCConfig",
    "MPCStep",
    "MPCDecision",
    "MPCController",
]
