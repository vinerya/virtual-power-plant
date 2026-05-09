"""
Deterministic battery dispatch MILP formulation (Milestone 1).

Single battery, finite horizon, perfect-foresight cost minimization against a
known price vector. Variables / constraints:

    p_charge[t]      >= 0    (kW drawn from grid into battery)
    p_discharge[t]   >= 0    (kW pushed to grid from battery)
    soc[t]           in [soc_min*cap, soc_max*cap]   (kWh)
    is_charging[t]   binary  (mutual-exclusion of charge/discharge)

    SOC dynamics:
        soc[t] = soc[t-1] + eta_charge*p_charge[t]*dt - p_discharge[t]*dt/eta_discharge
        soc[0] = soc_init * capacity                 (initial)
        soc[T-1] >= terminal_soc * capacity          (terminal SOC, defaults to soc_init)

    Big-M mutual exclusion:
        p_charge[t]    <= M_c * is_charging[t]
        p_discharge[t] <= M_d * (1 - is_charging[t])

    Objective (minimize):
        sum_t price[t] * (p_charge[t] - p_discharge[t]) * dt

A negative objective therefore corresponds to net revenue (price arbitrage).
Prices are ``Param(mutable=True)`` so the model can be rebuilt-free re-solved.
"""
from __future__ import annotations

from typing import Any, Dict, List

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


def _validate(params: Dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in params]
    if missing:
        raise ValueError(f"Missing required dispatch params: {missing}")
    prices: List[float] = list(params["prices"])
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


def build_battery_dispatch_model(params: Dict[str, Any]) -> pyo.ConcreteModel:
    """Build a Pyomo ConcreteModel for deterministic battery dispatch.

    Args:
        params: dict with the keys listed in :data:`REQUIRED_KEYS`. ``prices``
            is a sequence of length T defining the horizon. Costs are charged
            at ``price[t] * net_grid_draw[t] * dt_hours``.

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

    m = pyo.ConcreteModel(name="battery_dispatch_m1")

    # Sets
    m.T = pyo.RangeSet(0, T - 1)

    # Mutable parameters (so callers can re-solve with updated prices)
    m.price = pyo.Param(m.T, initialize={t: prices[t] for t in range(T)}, mutable=True)
    m.dt = pyo.Param(initialize=dt, mutable=True)

    # Static config params (kept as Param for transparency)
    m.cap = pyo.Param(initialize=cap)
    m.p_chg_max = pyo.Param(initialize=p_chg_max)
    m.p_dis_max = pyo.Param(initialize=p_dis_max)
    m.soc_min = pyo.Param(initialize=soc_min)
    m.soc_max = pyo.Param(initialize=soc_max)
    m.eta_c = pyo.Param(initialize=eta_c)
    m.eta_d = pyo.Param(initialize=eta_d)
    m.soc_0 = pyo.Param(initialize=soc_0)
    m.soc_terminal = pyo.Param(initialize=soc_terminal)

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

    # Objective: minimize energy cost (net grid draw priced at price[t])
    def _obj(model):
        return sum(
            model.price[t] * (model.p_charge[t] - model.p_discharge[t]) * model.dt
            for t in model.T
        )

    m.cost = pyo.Objective(rule=_obj, sense=pyo.minimize)

    return m
