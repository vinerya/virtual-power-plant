"""
ADMM consensus decomposition for fleet dispatch (Milestone 4).

For fleets with many batteries, the monolithic MILP from
:mod:`fleet_dispatch` becomes slow because the binary mutex variables couple
into a single MIP whose branching tree explodes with R. ADMM splits the
problem so each battery solves its own (small) MILP and a coordinator enforces
the feeder coupling.

Algorithm
---------
Average-form consensus ADMM. We let each battery's "local view" be
``p_r[t] = p_discharge[r,t] - p_charge[r,t]`` (export-positive kW) and
introduce a consensus variable ``z[t]`` representing the per-resource average
aggregate target. The site aggregate is then ``R * z[t]`` and is projected
onto the feeder window during the z-update.

    Subproblem r:  minimize cost_r + (rho/2) ||p_r - z + u_r||_2^2
    z-update    :  z = project_feeder_avg( mean_r(p_r + u_r) )
                   where the projection scales by R for the feeder caps.
    Dual update :  u_r <- u_r + (p_r - z)

Linearization choice
--------------------
HiGHS is a MILP solver — it cannot handle the quadratic penalty term directly.
We linearize ``||p_r - target||^2`` with an L1 surrogate: introduce auxiliary
non-negative deviation variables ``dev[t] >= ±(p_r[t] - target[t])`` and add
``rho * sum_t dev[t]`` to battery r's objective. L1 penalty is a known
ADMM variant ("L1 consensus" / Boyd et al. §6.4); it preserves convergence to
a feasible consensus when residuals are small but yields slightly slower
asymptotic convergence than L2 in the unconstrained case. Since our MILP is
discrete this is a reasonable trade-off and keeps the subproblems pure MILPs.

Feeder projection
-----------------
The z-update projects the aggregate target onto the feasible feeder window
``[-feeder_max_import_kw, feeder_max_export_kw]`` (export-positive). This is a
1-D box projection per timestep — O(T).

Returns dict
------------
    iterations           : int, count of ADMM iterations actually run.
    converged            : bool, whether tolerance was met.
    primal_residual      : float, final ||sum_r p_r - z||_inf.
    dual_residual        : float, final rho * ||z - z_prev||_inf.
    per_battery_solutions: dict[id -> {p_charge, p_discharge, soc, is_charging}]
    z_trajectory         : list[float], final aggregate target z (length T).
    objective            : float, sum of per-battery energy costs at consensus.
    aggregate            : list[float], realized sum_r (p_dis_r - p_chg_r).
    p_import / p_export  : projected feeder values.
"""

from __future__ import annotations

from typing import Any

import pyomo.environ as pyo

from ..solvers.pyomo_plugin import _try_import_pyomo
from .dispatch import build_battery_dispatch_model
from .fleet_dispatch import FleetBattery, FleetCoupling


def _make_l1_consensus_term(
    target: list[float],
    rho: float,
):
    """Closure: returns an objective_term hook that injects L1 ADMM penalty.

    The hook attaches auxiliary deviation vars ``m.admm_dev[t] >= 0`` and
    ``m.admm_dev_pos / m.admm_dev_neg`` constraints, then returns the linear
    term ``rho * sum_t admm_dev[t]`` to be added to the objective.
    """

    def _builder(model, params):
        T = len(target)
        if not hasattr(model, "admm_dev"):
            model.admm_dev = pyo.Var(model.T, domain=pyo.NonNegativeReals)

        # Constraints: dev[t] >= +(p_dis - p_chg - target[t])
        #              dev[t] >= -(p_dis - p_chg - target[t])
        def _pos(m, t):
            return m.admm_dev[t] >= (m.p_discharge[t] - m.p_charge[t]) - target[t]

        def _neg(m, t):
            return m.admm_dev[t] >= -((m.p_discharge[t] - m.p_charge[t]) - target[t])

        model.admm_dev_pos = pyo.Constraint(model.T, rule=_pos)
        model.admm_dev_neg = pyo.Constraint(model.T, rule=_neg)

    def _objective(model, params):
        return rho * sum(model.admm_dev[t] for t in model.T)

    return _builder, _objective


def _solve_subproblem(
    battery: FleetBattery,
    horizon_steps: int,
    dt_hours: float,
    prices: list[float],
    target: list[float],
    rho: float,
    solver_factory,
    pyo_mod,
    time_limit_s: float = 5.0,
) -> dict[str, list[float]]:
    """Solve battery r's local augmented Lagrangian subproblem."""
    params: dict[str, Any] = {
        "battery_capacity_kwh": battery.capacity_kwh,
        "max_charge_kw": battery.max_charge_kw,
        "max_discharge_kw": battery.max_discharge_kw,
        "soc_init": battery.soc_init,
        "soc_min": battery.soc_min,
        "soc_max": battery.soc_max,
        "eta_charge": battery.eta_charge,
        "eta_discharge": battery.eta_discharge,
        "prices": list(prices[:horizon_steps]),
        "dt_hours": dt_hours,
        "terminal_soc": (
            battery.terminal_soc if battery.terminal_soc is not None else battery.soc_init
        ),
    }

    builder, obj = _make_l1_consensus_term(target, rho)
    model = build_battery_dispatch_model(
        params,
        objective_terms=[obj],
        constraint_builders=[builder],
    )

    solver = solver_factory(time_limit_s)
    solver.solve(model)

    T = horizon_steps
    p_chg = [float(pyo_mod.value(model.p_charge[t])) for t in range(T)]
    p_dis = [float(pyo_mod.value(model.p_discharge[t])) for t in range(T)]
    soc = [float(pyo_mod.value(model.soc[t])) for t in range(T)]
    is_chg = [int(round(float(pyo_mod.value(model.is_charging[t])))) for t in range(T)]
    energy_cost = float(pyo_mod.value(model.energy_cost))

    return {
        "p_charge": p_chg,
        "p_discharge": p_dis,
        "soc": soc,
        "is_charging": is_chg,
        "energy_cost": energy_cost,
        "p_local": [d - c for c, d in zip(p_chg, p_dis)],
    }


def _project_feeder(value: float, coupling: FleetCoupling) -> float:
    """Project export-positive aggregate value onto feeder bounds."""
    upper = (
        coupling.feeder_max_export_kw
        if coupling.feeder_max_export_kw is not None
        else float("inf")
    )
    lower = (
        -coupling.feeder_max_import_kw
        if coupling.feeder_max_import_kw is not None
        else float("-inf")
    )
    if value > upper:
        return upper
    if value < lower:
        return lower
    return value


def admm_fleet_solve(
    batteries: list[FleetBattery],
    horizon_steps: int,
    dt_hours: float,
    prices: list[float],
    load_kw: list[float] | None = None,
    solar_kw: list[float] | None = None,
    coupling: FleetCoupling | None = None,
    rho: float = 1.0,
    max_iters: int = 50,
    tolerance: float = 1e-3,
    subproblem_time_limit_s: float = 5.0,
) -> dict[str, Any]:
    """Solve fleet dispatch via ADMM consensus decomposition."""
    if not batteries:
        raise ValueError("batteries must be non-empty")
    coupling = coupling or FleetCoupling()
    R = len(batteries)
    T = horizon_steps

    pyo_mod, solver_factory = _try_import_pyomo()
    if pyo_mod is None or solver_factory is None:
        raise RuntimeError("Pyomo + HiGHS required for ADMM fleet solve")

    prices = list(prices)
    if len(prices) < T:
        prices = prices + [prices[-1]] * (T - len(prices))
    prices = prices[:T]

    # State (average-form consensus ADMM)
    # z[t]: per-resource average target (kW per resource, export-positive).
    # u[r][t]: per-resource scaled dual.
    z = [0.0] * T
    u = [[0.0] * T for _ in range(R)]
    per_battery: dict[str, dict[str, list[float]]] = {}

    primal_res = float("inf")
    dual_res = float("inf")
    converged = False
    iterations = 0

    # Per-resource feeder scale: the site cap divided by R gives a per-resource
    # ceiling on the *average* target z.
    def _project_avg(value: float) -> float:
        upper = (
            coupling.feeder_max_export_kw / R
            if coupling.feeder_max_export_kw is not None
            else float("inf")
        )
        lower = (
            -coupling.feeder_max_import_kw / R
            if coupling.feeder_max_import_kw is not None
            else float("-inf")
        )
        if value > upper:
            return upper
        if value < lower:
            return lower
        return value

    for it in range(max_iters):
        iterations = it + 1
        z_prev = list(z)

        # ---- Subproblems ----
        # Each battery's L1 target = z - u_r (consensus form).
        per_battery_p_local: list[list[float]] = []
        for ri, b in enumerate(batteries):
            target = [z[t] - u[ri][t] for t in range(T)]
            sol = _solve_subproblem(
                b,
                T,
                dt_hours,
                prices,
                target,
                rho,
                solver_factory,
                pyo_mod,
                time_limit_s=subproblem_time_limit_s,
            )
            per_battery[b.id] = sol
            per_battery_p_local.append(sol["p_local"])

        # ---- z-update: average of (p_r + u_r), projected onto avg feeder window ----
        p_avg = [
            sum(per_battery_p_local[ri][t] + u[ri][t] for ri in range(R)) / R for t in range(T)
        ]
        z = [_project_avg(p_avg[t]) for t in range(T)]

        # ---- u-update (scaled dual): u_r <- u_r + (p_r - z) ----
        for ri in range(R):
            u[ri] = [u[ri][t] + (per_battery_p_local[ri][t] - z[t]) for t in range(T)]

        # ---- Residuals ----
        # Primal: per-resource consensus error
        primal_res = max(
            abs(per_battery_p_local[ri][t] - z[t]) for ri in range(R) for t in range(T)
        )
        dual_res = rho * max(abs(z[t] - z_prev[t]) for t in range(T))

        if primal_res <= tolerance and dual_res <= tolerance:
            converged = True
            break

    # Compute total energy cost (consensus)
    total_cost = sum(per_battery[b.id]["energy_cost"] for b in batteries)
    # Site aggregate: sum of per-battery locals (= R * z at consensus, but we
    # report the realised sum since L1 ADMM may stop with small residual).
    aggregate = [sum(per_battery[b.id]["p_local"][t] for b in batteries) for t in range(T)]
    # Project the realised aggregate onto feeder bounds for reporting.
    site_z = [_project_feeder(aggregate[t], coupling) for t in range(T)]
    p_export = [max(0.0, site_z[t]) for t in range(T)]
    p_import = [max(0.0, -site_z[t]) for t in range(T)]

    return {
        "iterations": iterations,
        "converged": converged,
        "primal_residual": primal_res,
        "dual_residual": dual_res,
        "per_battery_solutions": per_battery,
        "z_trajectory": [R * zt for zt in z],  # site-level
        "aggregate": aggregate,
        "p_import": p_import,
        "p_export": p_export,
        "objective": total_cost,
        "rho": rho,
    }


__all__ = ["admm_fleet_solve"]
