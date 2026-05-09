"""
Pyomo + HiGHS optimization plugin for VPP (Milestone 1).

This plugin solves the deterministic single-battery dispatch problem defined
in :mod:`vpp.optimization.formulations.dispatch`. It is registered as the
preferred plugin for problems whose ``metadata['type']`` is
``"battery_dispatch"`` or ``"deterministic"``.

Problem parameter contract (passed via ``OptimizationProblem.parameters``):

    battery_capacity_kwh : float, > 0
    max_charge_kw        : float, > 0
    max_discharge_kw     : float, > 0
    soc_init             : float, fraction in [soc_min, soc_max]
    soc_min              : float, fraction in [0, 1)
    soc_max              : float, fraction in (soc_min, 1]
    eta_charge           : float, in (0, 1]
    eta_discharge        : float, in (0, 1]
    prices               : list[float], length T (currency / kWh)
    dt_hours             : float, > 0
    terminal_soc         : float, optional fraction (defaults to soc_init)

The plugin honours ``timeout_ms`` by passing it as ``time_limit`` (seconds) to
the HiGHS solver.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..base import (
    OptimizationPlugin,
    OptimizationProblem,
    OptimizationResult,
    OptimizationStatus,
    RuleBasedOptimizer,
)
from ..formulations.dispatch import REQUIRED_KEYS, build_battery_dispatch_model


def _try_import_pyomo():
    """Attempt to import pyomo and a HiGHS solver factory.

    Returns a tuple ``(pyo_module, solver_factory_callable)`` or ``(None, None)``
    if Pyomo or HiGHS bindings are unavailable. ``solver_factory_callable``
    takes a ``time_limit_s`` arg and returns a configured solver object with a
    ``solve(model)`` method that returns a Pyomo results object.
    """
    try:
        import pyomo.environ as pyo  # noqa: F401
    except Exception:
        return None, None

    # Prefer the appsi HiGHS interface (pure-python via highspy, no PATH lookup).
    try:
        from pyomo.contrib.appsi.solvers.highs import Highs as _AppsiHighs

        # Probe availability once; appsi returns an Availability enum.
        probe = _AppsiHighs()
        avail = probe.available()
        # Truthy if any flavour of license / availability detected.
        if not avail:
            raise RuntimeError("appsi Highs reports unavailable")

        def _factory(time_limit_s: Optional[float]):
            s = _AppsiHighs()
            if time_limit_s is not None:
                s.config.time_limit = float(time_limit_s)
            return s

        return pyo, _factory
    except Exception:
        pass

    # Fall back to classic SolverFactory("appsi_highs") / ("highs")
    try:
        import pyomo.environ as pyo

        for name in ("appsi_highs", "highs"):
            try:
                s = pyo.SolverFactory(name)
                if s is not None and s.available(exception_flag=False):

                    def _factory(time_limit_s: Optional[float], _name=name):
                        sv = pyo.SolverFactory(_name)
                        if time_limit_s is not None:
                            try:
                                sv.options["time_limit"] = float(time_limit_s)
                            except Exception:
                                pass
                        return sv

                    return pyo, _factory
            except Exception:
                continue
    except Exception:
        return None, None

    return None, None


class PyomoPlugin(OptimizationPlugin):
    """MILP optimization plugin powered by Pyomo + HiGHS."""

    SUPPORTED_TYPES = frozenset({"battery_dispatch", "deterministic"})

    def __init__(self) -> None:
        super().__init__(name="pyomo_highs", version="0.1.0")
        self._pyo, self._solver_factory = _try_import_pyomo()

    # ------------------------------------------------------------------ ABC
    def is_available(self) -> bool:
        return self._pyo is not None and self._solver_factory is not None

    def validate_problem(self, problem: OptimizationProblem) -> bool:
        ptype = problem.metadata.get("type", "")
        if ptype not in self.SUPPORTED_TYPES:
            return False
        params = problem.parameters or {}
        missing = [k for k in REQUIRED_KEYS if k not in params]
        if missing:
            self.logger.debug(f"Pyomo plugin missing keys: {missing}")
            return False
        prices: List[float] = list(params["prices"])
        if len(prices) == 0:
            return False
        return True

    def solve(
        self,
        problem: OptimizationProblem,
        timeout_ms: Optional[int] = None,
    ) -> OptimizationResult:
        start = time.time()
        if not self.is_available():
            return OptimizationResult(
                status=OptimizationStatus.FAILED,
                objective_value=float("inf"),
                solution={},
                solve_time=time.time() - start,
                metadata={"error": "pyomo or HiGHS not available"},
            )
        if not self.validate_problem(problem):
            return OptimizationResult(
                status=OptimizationStatus.FAILED,
                objective_value=float("inf"),
                solution={},
                solve_time=time.time() - start,
                metadata={"error": "problem failed validation"},
            )

        pyo = self._pyo
        params = dict(problem.parameters)
        try:
            model = build_battery_dispatch_model(params)
        except Exception as e:
            return OptimizationResult(
                status=OptimizationStatus.FAILED,
                objective_value=float("inf"),
                solution={},
                solve_time=time.time() - start,
                metadata={"error": f"model build failed: {e}"},
            )

        time_limit_s = (timeout_ms / 1000.0) if timeout_ms else None
        try:
            solver = self._solver_factory(time_limit_s)
        except Exception as e:
            return OptimizationResult(
                status=OptimizationStatus.FAILED,
                objective_value=float("inf"),
                solution={},
                solve_time=time.time() - start,
                metadata={"error": f"solver init failed: {e}"},
            )

        # Two solver flavours have different result shapes; handle both.
        try:
            results = solver.solve(model)
        except Exception as e:
            return OptimizationResult(
                status=OptimizationStatus.FAILED,
                objective_value=float("inf"),
                solution={},
                solve_time=time.time() - start,
                metadata={"error": f"solve failed: {e}"},
            )

        status, gap = self._extract_status(results)
        if status != OptimizationStatus.SUCCESS:
            return OptimizationResult(
                status=status,
                objective_value=float("inf"),
                solution={},
                solve_time=time.time() - start,
                metadata={"solver_status": str(results)},
            )

        T = len(params["prices"])
        p_chg = [float(pyo.value(model.p_charge[t])) for t in range(T)]
        p_dis = [float(pyo.value(model.p_discharge[t])) for t in range(T)]
        soc = [float(pyo.value(model.soc[t])) for t in range(T)]
        obj = float(pyo.value(model.cost))

        solution: Dict[str, Any] = {
            "p_charge": p_chg,
            "p_discharge": p_dis,
            "p_net": [c - d for c, d in zip(p_chg, p_dis)],
            "soc": soc,
            "method": self.name,
        }
        return OptimizationResult(
            status=OptimizationStatus.SUCCESS,
            objective_value=obj,
            solution=solution,
            solve_time=time.time() - start,
            gap=gap,
            metadata={"horizon": T, "solver": "highs"},
            solver_info={"name": self.name, "version": self.version},
        )

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _extract_status(results: Any):
        """Map solver result objects to OptimizationStatus."""
        # appsi result: has termination_condition enum
        tc = getattr(results, "termination_condition", None)
        if tc is not None:
            tc_str = str(tc).lower()
            if "optimal" in tc_str:
                return OptimizationStatus.SUCCESS, 0.0
            if "infeasible" in tc_str:
                return OptimizationStatus.INFEASIBLE, 0.0
            if "unbounded" in tc_str:
                return OptimizationStatus.UNBOUNDED, 0.0
            if "timelimit" in tc_str or "maxtime" in tc_str:
                return OptimizationStatus.TIMEOUT, 0.0
            return OptimizationStatus.FAILED, 0.0

        # Classic pyomo SolverResults
        try:
            solver_status = str(results.solver.status).lower()
            term = str(results.solver.termination_condition).lower()
            if "ok" in solver_status and "optimal" in term:
                return OptimizationStatus.SUCCESS, 0.0
            if "infeasible" in term:
                return OptimizationStatus.INFEASIBLE, 0.0
            if "maxtime" in term or "timelimit" in term:
                return OptimizationStatus.TIMEOUT, 0.0
        except Exception:
            pass
        return OptimizationStatus.FAILED, 0.0


# ---------------------------------------------------------------------------
# Rule-based fallback for the same problem type
# ---------------------------------------------------------------------------
class SimpleBatteryDispatchRules(RuleBasedOptimizer):
    """Greedy threshold rule for battery arbitrage.

    Computes price thresholds (median-split) and charges below the lower
    threshold / discharges above the upper threshold, respecting power and SOC
    bounds. Acts as the fallback for ``battery_dispatch`` problems and as the
    baseline for fair comparisons against the MILP plugin.
    """

    def __init__(self) -> None:
        super().__init__("simple_battery_dispatch_rules")

    def solve(self, problem: OptimizationProblem) -> OptimizationResult:
        start = time.time()
        params = problem.parameters or {}
        try:
            for k in REQUIRED_KEYS:
                if k not in params:
                    raise ValueError(f"missing param {k}")
            prices = list(params["prices"])
            T = len(prices)
            cap = float(params["battery_capacity_kwh"])
            p_chg_max = float(params["max_charge_kw"])
            p_dis_max = float(params["max_discharge_kw"])
            soc_min = float(params["soc_min"]) * cap
            soc_max = float(params["soc_max"]) * cap
            eta_c = float(params["eta_charge"])
            eta_d = float(params["eta_discharge"])
            dt = float(params["dt_hours"])
            soc = float(params["soc_init"]) * cap
            terminal = float(params.get("terminal_soc", params["soc_init"])) * cap

            sorted_p = sorted(prices)
            lo = sorted_p[max(0, T // 3 - 1)]
            hi = sorted_p[min(T - 1, 2 * T // 3)]

            p_chg = [0.0] * T
            p_dis = [0.0] * T
            soc_traj = [0.0] * T

            for t in range(T):
                price = prices[t]
                # Reserve enough remaining time to recharge to terminal SOC if needed
                remaining_steps = T - 1 - t
                max_recoverable = remaining_steps * eta_c * p_chg_max * dt
                must_save_for_terminal = max(0.0, terminal - soc - max_recoverable)

                if price <= lo and soc < soc_max:
                    # charge as much as possible
                    headroom_kwh = soc_max - soc
                    p = min(p_chg_max, headroom_kwh / (eta_c * dt))
                    if p > 0:
                        p_chg[t] = p
                        soc += eta_c * p * dt
                elif price >= hi and soc > soc_min and must_save_for_terminal <= 0:
                    available_kwh = soc - max(soc_min, terminal if t == T - 1 else soc_min)
                    p = min(p_dis_max, available_kwh * eta_d / dt)
                    if p > 0:
                        p_dis[t] = p
                        soc -= p * dt / eta_d
                soc_traj[t] = soc

            cost = sum(prices[t] * (p_chg[t] - p_dis[t]) * dt for t in range(T))
            return OptimizationResult(
                status=OptimizationStatus.SUCCESS,
                objective_value=cost,
                solution={
                    "p_charge": p_chg,
                    "p_discharge": p_dis,
                    "p_net": [c - d for c, d in zip(p_chg, p_dis)],
                    "soc": soc_traj,
                    "method": self.name,
                },
                solve_time=time.time() - start,
                metadata={"method": self.name, "lo_threshold": lo, "hi_threshold": hi},
            )
        except Exception as e:
            return OptimizationResult(
                status=OptimizationStatus.FAILED,
                objective_value=float("inf"),
                solution={},
                solve_time=time.time() - start,
                metadata={"error": str(e)},
            )
