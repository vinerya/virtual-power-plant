"""
Deterministic battery dispatch MILP formulation (Milestone 1 + Milestone 2 hooks).

Single battery, finite horizon, perfect-foresight cost minimization against a
known price vector.

Stable Var/Param surface (M2 contract; downstream extensions may rely on these
names):

    Sets / Params:
        m.T          : RangeSet(0, T-1) - time index
        m.dt         : Param (mutable) - hours per step
        m.price[t]   : Param (mutable)
        m.cap, m.p_chg_max, m.p_dis_max, m.soc_min, m.soc_max
        m.eta_c, m.eta_d, m.soc_0, m.soc_terminal

    Vars:
        m.p_charge[t]    NonNegativeReals (kW)
        m.p_discharge[t] NonNegativeReals (kW)
        m.soc[t]         in [soc_min, soc_max] (kWh)
        m.is_charging[t] Binary

Hook API (M2):
    objective_terms     : list of callables (model, params) -> Pyomo Expression.
                          Each returned expression is added (summed) into the
                          objective alongside the base energy-cost term.
    constraint_builders : list of callables (model, params) -> None. Each
                          callable mutates the model in-place (adding Vars,
                          Constraints, Expressions, etc.) BEFORE the objective
                          is declared, so it may introduce auxiliary vars that
                          the objective hooks reference.

Order of construction:
    1. Validate params
    2. Build sets / params / vars / base constraints (mutex, SOC, terminal)
    3. Run constraint_builders (so they can add aux vars referenced later)
    4. Build base energy-cost expression
    5. Sum in objective_terms
    6. Declare m.cost Objective (minimize)

A negative objective therefore corresponds to net revenue (price arbitrage).
Prices are ``Param(mutable=True)`` so the model can be re-solved without rebuild.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pyomo.environ as pyo

REQUIRED_KEYS = (
    "battery_capacity_kwh",
    "max_charge_kw",
    "max_discharge_kw",
    "soc_init",
    "soc_min",
    "soc_max",
    "eta_charge",
    "eta_discharge",
    "prices",
    "dt_hours",
)

# Optional params (M2 tariff integration):
#   load        : sequence[float] length T, kW load (default zeros)
#   solar       : sequence[float] length T, kW PV (default zeros)
#   disable_base_energy_cost : bool, if True the base m.energy_cost expression
#                              is forced to 0 so callers can replace energy
#                              pricing with a tariff-based objective term
#                              (see vpp.tariffs.optimization.add_tariff_energy_term).

# Type alias for hook callables.
ObjectiveTerm = Callable[[pyo.ConcreteModel, dict[str, Any]], Any]
ConstraintBuilder = Callable[[pyo.ConcreteModel, dict[str, Any]], None]


def _validate(params: dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in params]
    if missing:
        raise ValueError(f"Missing required dispatch params: {missing}")
    prices: list[float] = list(params["prices"])
    if len(prices) == 0:
        raise ValueError("prices must be a non-empty sequence")
    for k in (
        "battery_capacity_kwh",
        "max_charge_kw",
        "max_discharge_kw",
        "dt_hours",
    ):
        if params[k] <= 0:
            raise ValueError(f"{k} must be > 0, got {params[k]}")
    for k in ("eta_charge", "eta_discharge"):
        v = params[k]
        if not (0 < v <= 1):
            raise ValueError(f"{k} must be in (0, 1], got {v}")
    smin, smax, s0 = params["soc_min"], params["soc_max"], params["soc_init"]
    if not (0 <= smin < smax <= 1):
        raise ValueError(f"soc_min/soc_max invalid: [{smin}, {smax}]")
    if not (smin <= s0 <= smax):
        raise ValueError(f"soc_init {s0} not in [soc_min, soc_max]")


def build_battery_dispatch_model(
    params: dict[str, Any],
    objective_terms: Sequence[ObjectiveTerm] | None = None,
    constraint_builders: Sequence[ConstraintBuilder] | None = None,
) -> pyo.ConcreteModel:
    """Build a Pyomo ConcreteModel for deterministic battery dispatch.

    Args:
        params: dict with the keys listed in :data:`REQUIRED_KEYS`. ``prices``
            is a sequence of length T defining the horizon. Costs are charged
            at ``price[t] * net_grid_draw[t] * dt_hours``.
        objective_terms: optional list of callables ``(model, params) -> Expression``;
            each returned expression is summed into the objective.
        constraint_builders: optional list of callables ``(model, params) -> None``
            that mutate the model in-place to add constraints/aux vars. Run
            BEFORE the objective is declared.

    Returns:
        A fully-built :class:`pyomo.environ.ConcreteModel` ready for solve.
    """
    _validate(params)

    prices = list(params["prices"])
    T = len(prices)
    cap = float(params["battery_capacity_kwh"])
    p_chg_max = float(params["max_charge_kw"])
    p_dis_max = float(params["max_discharge_kw"])
    soc_min = float(params["soc_min"]) * cap
    soc_max = float(params["soc_max"]) * cap
    soc_0 = float(params["soc_init"]) * cap
    eta_c = float(params["eta_charge"])
    eta_d = float(params["eta_discharge"])
    dt = float(params["dt_hours"])
    terminal_soc_frac = float(params.get("terminal_soc", params["soc_init"]))
    soc_terminal = terminal_soc_frac * cap

    m = pyo.ConcreteModel(name="battery_dispatch_m2")

    # Sets
    m.T = pyo.RangeSet(0, T - 1)

    # Mutable parameters (so callers can re-solve with updated prices)
    m.price = pyo.Param(m.T, initialize={t: prices[t] for t in range(T)}, mutable=True)
    m.dt = pyo.Param(initialize=dt, mutable=True)

    # Static config params
    m.cap = pyo.Param(initialize=cap)
    m.p_chg_max = pyo.Param(initialize=p_chg_max)
    m.p_dis_max = pyo.Param(initialize=p_dis_max)
    m.soc_min = pyo.Param(initialize=soc_min)
    m.soc_max = pyo.Param(initialize=soc_max)
    m.eta_c = pyo.Param(initialize=eta_c)
    m.eta_d = pyo.Param(initialize=eta_d)
    m.soc_0 = pyo.Param(initialize=soc_0)
    m.soc_terminal = pyo.Param(initialize=soc_terminal)

    # Optional load/solar parameters (default zero) for tariff-driven dispatch.
    load = list(params.get("load") or [0.0] * T)
    solar = list(params.get("solar") or [0.0] * T)
    if len(load) != T:
        raise ValueError(f"load length {len(load)} != horizon {T}")
    if len(solar) != T:
        raise ValueError(f"solar length {len(solar)} != horizon {T}")
    # Note: 'load' is a reserved attribute name on Pyomo Blocks; use load_kw.
    m.load_kw = pyo.Param(m.T, initialize={t: load[t] for t in range(T)}, mutable=True)
    m.solar_kw = pyo.Param(m.T, initialize={t: solar[t] for t in range(T)}, mutable=True)

    # Variables
    m.p_charge = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, p_chg_max))
    m.p_discharge = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, p_dis_max))
    m.soc = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(soc_min, soc_max))
    m.is_charging = pyo.Var(m.T, domain=pyo.Binary)

    # Mutual exclusion (big-M with M = max power)
    def _excl_chg(model, t):
        return model.p_charge[t] <= model.p_chg_max * model.is_charging[t]

    def _excl_dis(model, t):
        return model.p_discharge[t] <= model.p_dis_max * (1 - model.is_charging[t])

    m.excl_charge = pyo.Constraint(m.T, rule=_excl_chg)
    m.excl_discharge = pyo.Constraint(m.T, rule=_excl_dis)

    # SOC dynamics
    def _soc_dyn(model, t):
        if t == 0:
            prev = model.soc_0
        else:
            prev = model.soc[t - 1]
        return (
            model.soc[t]
            == prev
            + model.eta_c * model.p_charge[t] * model.dt
            - model.p_discharge[t] * model.dt / model.eta_d
        )

    m.soc_dynamics = pyo.Constraint(m.T, rule=_soc_dyn)

    # Terminal SOC
    def _terminal(model):
        return model.soc[T - 1] >= model.soc_terminal

    m.terminal_soc_con = pyo.Constraint(rule=_terminal)

    # ---- M2 hook: constraint_builders run before objective is declared ----
    if constraint_builders:
        for i, builder in enumerate(constraint_builders):
            try:
                builder(m, params)
            except Exception as e:
                raise RuntimeError(
                    f"constraint_builder #{i} ({getattr(builder, '__name__', builder)}) failed: {e}"
                ) from e

    # Base objective term: energy cost (net grid draw priced at price[t]).
    if params.get("disable_base_energy_cost", False):
        base_cost_expr = 0.0
    else:
        base_cost_expr = sum(m.price[t] * (m.p_charge[t] - m.p_discharge[t]) * m.dt for t in m.T)
    # Expose as a named Expression for downstream introspection / hooks.
    m.energy_cost = pyo.Expression(expr=base_cost_expr)

    extra_terms = []
    if objective_terms:
        for i, term in enumerate(objective_terms):
            try:
                term_expr = term(m, params)
            except Exception as exc:
                raise RuntimeError(
                    f"objective_term #{i} ({getattr(term, '__name__', term)}) failed: {exc}"
                ) from exc
            extra_terms.append(term_expr)

    if extra_terms:
        m.cost = pyo.Objective(expr=m.energy_cost + sum(extra_terms), sense=pyo.minimize)
    else:
        m.cost = pyo.Objective(expr=m.energy_cost, sense=pyo.minimize)

    return m
