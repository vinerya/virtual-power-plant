"""
Stochastic extensive-form battery dispatch plugin (Pyomo + HiGHS) with CVaR.

Registered for ``problem_type == 'stochastic_dispatch'``. Expects
``problem.parameters`` to carry the deterministic dispatch params plus a
``scenarios`` list (each entry a dict or :class:`Scenario` with ``probability``
and ``prices``) and optional ``cvar_alpha`` / ``cvar_lambda``.
"""

from __future__ import annotations

import time
from typing import Any

from ..base import (
    OptimizationPlugin,
    OptimizationProblem,
    OptimizationResult,
    OptimizationStatus,
)
from ..formulations.stochastic import build_stochastic_dispatch_model
from .pyomo_plugin import PyomoPlugin, _try_import_pyomo


class StochasticCVaRPlugin(OptimizationPlugin):
    """Two-stage stochastic dispatch with CVaR risk term, solved as MILP."""

    SUPPORTED_TYPES = frozenset({"stochastic_dispatch"})

    def __init__(self) -> None:
        super().__init__(name="pyomo_highs_stochastic_cvar", version="0.1.0")
        self._pyo, self._solver_factory = _try_import_pyomo()

    def is_available(self) -> bool:
        return self._pyo is not None and self._solver_factory is not None

    def validate_problem(self, problem: OptimizationProblem) -> bool:
        ptype = problem.metadata.get("type", "")
        if ptype not in self.SUPPORTED_TYPES:
            return False
        params = problem.parameters or {}
        scenarios = params.get("scenarios")
        if not scenarios:
            return False
        # Quick sanity: each has probability + prices (directly or via .data).
        for sc in scenarios:
            prob = getattr(sc, "probability", None)
            data = getattr(sc, "data", None)
            if prob is None and isinstance(sc, dict):
                prob = sc.get("probability")
                data = sc.get("data") or sc
            prices = None
            if isinstance(data, dict):
                prices = data.get("prices")
            if prices is None and isinstance(sc, dict):
                prices = sc.get("prices")
            if prob is None or prices is None:
                return False
        return True

    def solve(
        self,
        problem: OptimizationProblem,
        timeout_ms: int | None = None,
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
        scenarios = list(params.pop("scenarios"))

        try:
            model = build_stochastic_dispatch_model(params, scenarios)
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

        status, gap = PyomoPlugin._extract_status(results)
        if status != OptimizationStatus.SUCCESS:
            return OptimizationResult(
                status=status,
                objective_value=float("inf"),
                solution={},
                solve_time=time.time() - start,
                metadata={"solver_status": str(results)},
            )

        # Extract per-scenario trajectories.
        S = len(scenarios)
        T = len(model.T)
        per_scenario: list[dict[str, Any]] = []
        scenario_costs: list[float] = []
        probs: list[float] = []
        for s in range(S):
            p_chg = [float(pyo.value(model.p_charge[t, s])) for t in range(T)]
            p_dis = [float(pyo.value(model.p_discharge[t, s])) for t in range(T)]
            soc = [float(pyo.value(model.soc[t, s])) for t in range(T)]
            cost_s = float(pyo.value(model.cost_s[s]))
            pi_s = float(pyo.value(model.pi[s]))
            per_scenario.append(
                {
                    "p_charge": p_chg,
                    "p_discharge": p_dis,
                    "soc": soc,
                    "cost": cost_s,
                    "probability": pi_s,
                }
            )
            scenario_costs.append(cost_s)
            probs.append(pi_s)

        expected_cost = float(pyo.value(model.expected_cost))
        cvar = float(pyo.value(model.cvar))
        eta = float(pyo.value(model.eta))
        alpha = float(pyo.value(model.alpha))
        lam = float(pyo.value(model.lam))
        # Stage-1 (here-and-now) decisions: identical across scenarios.
        stage1 = {
            "p_charge_0": float(pyo.value(model.p_charge[0, 0])),
            "p_discharge_0": float(pyo.value(model.p_discharge[0, 0])),
            "is_charging_0": float(pyo.value(model.is_charging[0, 0])),
        }

        # Value-at-risk at level alpha equals the optimal eta (Rockafellar–Uryasev).
        value_at_risk = eta

        solution: dict[str, Any] = {
            "scenarios": per_scenario,
            "expected_cost": expected_cost,
            "cvar": cvar,
            "value_at_risk": value_at_risk,
            "stage1": stage1,
            "scenario_costs": scenario_costs,
            "probabilities": probs,
            "alpha": alpha,
            "lambda": lam,
            "method": self.name,
        }

        return OptimizationResult(
            status=OptimizationStatus.SUCCESS,
            objective_value=float(pyo.value(model.total_cost)),
            solution=solution,
            solve_time=time.time() - start,
            gap=gap,
            metadata={"horizon": T, "scenarios": S, "solver": "highs"},
            solver_info={"name": self.name, "version": self.version},
        )
