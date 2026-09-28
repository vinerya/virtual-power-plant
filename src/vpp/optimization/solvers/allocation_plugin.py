"""
Single-interval power allocation: LP plugin (Pyomo + HiGHS) and rule fallback.

Registered for ``problem.metadata['type'] == 'power_allocation'``. The problem
parameters follow :mod:`vpp.optimization.formulations.power_allocation`.

Both solvers return the same solution shape::

    {
        "allocations": {resource_id: kW (export-positive)},
        "delivered_kw": float,
        "shortfall_kw": float,   # target - delivered
        "method": str,
    }
"""
from __future__ import annotations

import time
from typing import Any

from ..base import (
    OptimizationPlugin,
    OptimizationProblem,
    OptimizationResult,
    OptimizationStatus,
    RuleBasedOptimizer,
)
from ..formulations.power_allocation import (
    build_power_allocation_model,
    validate_allocation_params,
)
from .pyomo_plugin import PyomoPlugin, _try_import_pyomo

_TOL = 1e-6


def _energy_cost(resources: list[dict[str, Any]], alloc: dict[str, float], dt: float) -> float:
    """Linear (segment-free) cost of an allocation, for apples-to-apples reporting."""
    total = 0.0
    for r in resources:
        p = alloc.get(str(r["id"]), 0.0)
        if p >= 0:
            total += float(r.get("cost_up", 0.0)) * p * dt
        else:
            total += float(r.get("cost_down", 0.0)) * (-p) * dt
    return total


class PowerAllocationPlugin(OptimizationPlugin):
    """LP allocation of a site power target across heterogeneous resources."""

    SUPPORTED_TYPES = frozenset({"power_allocation"})

    def __init__(self) -> None:
        super().__init__(name="pyomo_highs_allocation", version="0.1.0")
        self._pyo, self._solver_factory = _try_import_pyomo()

    def is_available(self) -> bool:
        return self._pyo is not None and self._solver_factory is not None

    def validate_problem(self, problem: OptimizationProblem) -> bool:
        if problem.metadata.get("type", "") not in self.SUPPORTED_TYPES:
            return False
        try:
            validate_allocation_params(problem.parameters or {})
        except (ValueError, KeyError, TypeError):
            return False
        return True

    def _fail(self, start: float, error: str) -> OptimizationResult:
        return OptimizationResult(
            status=OptimizationStatus.FAILED,
            objective_value=float("inf"),
            solution={},
            solve_time=time.time() - start,
            metadata={"error": error},
        )

    def solve(
        self,
        problem: OptimizationProblem,
        timeout_ms: int | None = None,
    ) -> OptimizationResult:
        start = time.time()
        if not self.is_available():
            return self._fail(start, "pyomo or HiGHS not available")
        if not self.validate_problem(problem):
            return self._fail(start, "problem failed validation")
        params = dict(problem.parameters)
        try:
            model = build_power_allocation_model(params)
            solver = self._solver_factory((timeout_ms / 1000.0) if timeout_ms else None)
            results = solver.solve(model)
        except Exception as e:
            return self._fail(start, f"solve failed: {e}")

        status, gap = PyomoPlugin._extract_status(results)
        if status != OptimizationStatus.SUCCESS:
            return OptimizationResult(
                status=status,
                objective_value=float("inf"),
                solution={},
                solve_time=time.time() - start,
                metadata={"solver_status": str(results)},
            )

        pyo = self._pyo
        alloc = {r: float(pyo.value(model.p[r])) for r in model.R}
        # Clean solver noise so a 0 kW setpoint is reported as exactly 0.
        alloc = {r: (0.0 if abs(v) < _TOL else v) for r, v in alloc.items()}
        delivered = sum(alloc.values())
        target = float(params["target_kw"])
        dt = float(params.get("dt_hours", 1.0))
        return OptimizationResult(
            status=OptimizationStatus.SUCCESS,
            objective_value=_energy_cost(params["resources"], alloc, dt),
            solution={
                "allocations": alloc,
                "delivered_kw": delivered,
                "shortfall_kw": target - delivered,
                "method": self.name,
            },
            solve_time=time.time() - start,
            gap=gap,
            metadata={"solver": "highs", "resources": len(alloc)},
            solver_info={"name": self.name, "version": self.version},
        )


class ProportionalAllocationRules(RuleBasedOptimizer):
    """Headroom-proportional split — the rule-based fallback.

    Every resource contributes the same fraction of its headroom in the
    direction of the target (``hi_kw`` for export, ``-lo_kw`` for import),
    which is the classic proportional dispatch. Costs are ignored.
    """

    def __init__(self) -> None:
        super().__init__("proportional_allocation_rules")

    def solve(self, problem: OptimizationProblem) -> OptimizationResult:
        start = time.time()
        params = problem.parameters or {}
        try:
            validate_allocation_params(params)
        except (ValueError, KeyError, TypeError) as e:
            return OptimizationResult(
                status=OptimizationStatus.FAILED,
                objective_value=float("inf"),
                solution={},
                solve_time=time.time() - start,
                metadata={"error": str(e)},
            )
        resources = list(params["resources"])
        target = float(params["target_kw"])
        dt = float(params.get("dt_hours", 1.0))
        if target >= 0:
            headroom = {str(r["id"]): max(0.0, float(r["hi_kw"])) for r in resources}
        else:
            headroom = {str(r["id"]): min(0.0, float(r["lo_kw"])) for r in resources}
        total = sum(headroom.values())
        if total == 0:
            ratio = 0.0
        else:
            ratio = min(1.0, target / total)
        alloc = {rid: h * ratio for rid, h in headroom.items()}
        delivered = sum(alloc.values())
        return OptimizationResult(
            status=OptimizationStatus.SUCCESS,
            objective_value=_energy_cost(resources, alloc, dt),
            solution={
                "allocations": alloc,
                "delivered_kw": delivered,
                "shortfall_kw": target - delivered,
                "method": self.name,
            },
            solve_time=time.time() - start,
            metadata={"method": self.name},
        )


__all__ = ["PowerAllocationPlugin", "ProportionalAllocationRules"]
