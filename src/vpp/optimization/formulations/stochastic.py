"""
Stochastic battery dispatch (extensive form) with CVaR — Milestone 2.

We solve a two-stage MILP with scenario-indexed variables. Stage-1 (the
"here-and-now" decision at t=0) is shared across all scenarios; stage-2
(t=1..T-1) reacts to the realized scenario.

Variables (per scenario s):
    p_charge[t,s]     >= 0        (kW)
    p_discharge[t,s]  >= 0        (kW)
    soc[t,s]          in [smin, smax]  (kWh)
    is_charging[t,s]  binary

Per-scenario cost:
    cost[s] = sum_t price[t,s] * (p_charge[t,s] - p_discharge[t,s]) * dt

Non-anticipativity: p_charge[0,s], p_discharge[0,s], is_charging[0,s] are
identical across all scenarios. Encoded via explicit equality constraints to
the first scenario s=0 (cleaner than introducing extra "stage-1" vars).

Risk measure (Rockafellar-Uryasev linearization of CVaR_alpha):
    eta : free real
    z[s] >= cost[s] - eta,   z[s] >= 0
    CVaR_alpha = eta + (1 / (1 - alpha)) * sum_s pi[s] * z[s]

Objective:
    min  sum_s pi[s] * cost[s]   +   lambda * CVaR_alpha

When alpha == 0 the CVaR term collapses to the expected cost (eta and z self-
adjust so that CVaR == E[cost] is achievable). When alpha -> 1 the CVaR term
emphasizes worst-case scenarios.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyomo.environ as pyo

if TYPE_CHECKING:
    from collections.abc import Sequence

REQUIRED_DET_KEYS = (
    "battery_capacity_kwh",
    "max_charge_kw",
    "max_discharge_kw",
    "soc_init",
    "soc_min",
    "soc_max",
    "eta_charge",
    "eta_discharge",
    "dt_hours",
)


def _scenario_prices(scenario: Any) -> list[float]:
    """Extract a price vector from a Scenario or plain dict."""
    if hasattr(scenario, "data") and isinstance(scenario.data, dict):
        prices = scenario.data.get("prices")
    elif isinstance(scenario, dict):
        prices = scenario.get("prices") or scenario.get("data", {}).get("prices")
    else:
        prices = None
    if prices is None:
        raise ValueError("scenario missing 'prices'")
    return [float(p) for p in prices]


def _scenario_probability(scenario: Any) -> float:
    if hasattr(scenario, "probability"):
        return float(scenario.probability)
    if isinstance(scenario, dict):
        return float(scenario.get("probability", 0.0))
    raise ValueError("scenario has no probability")


def _validate(params: dict[str, Any], scenarios: Sequence[Any]) -> None:
    missing = [k for k in REQUIRED_DET_KEYS if k not in params]
    if missing:
        raise ValueError(f"Missing required dispatch params: {missing}")
    if len(scenarios) == 0:
        raise ValueError("scenarios must be non-empty")
    Ts = {len(_scenario_prices(s)) for s in scenarios}
    if len(Ts) != 1:
        raise ValueError(f"scenarios have inconsistent horizons: {Ts}")
    if next(iter(Ts)) == 0:
        raise ValueError("scenario prices must be non-empty")
    smin, smax, s0 = params["soc_min"], params["soc_max"], params["soc_init"]
    if not (0 <= smin < smax <= 1):
        raise ValueError(f"soc_min/soc_max invalid: [{smin}, {smax}]")
    if not (smin <= s0 <= smax):
        raise ValueError(f"soc_init {s0} not in [soc_min, soc_max]")
    alpha = float(params.get("cvar_alpha", 0.95))
    if not (0.0 <= alpha < 1.0):
        raise ValueError(f"cvar_alpha must be in [0, 1), got {alpha}")
    lam = float(params.get("cvar_lambda", 0.0))
    if lam < 0:
        raise ValueError(f"cvar_lambda must be >= 0, got {lam}")


def build_stochastic_dispatch_model(
    params: dict[str, Any],
    scenarios: Sequence[Any],
) -> pyo.ConcreteModel:
    """Build the extensive-form stochastic dispatch MILP with CVaR risk term.

    Args:
        params: dict with the deterministic dispatch keys, plus optional
            ``cvar_alpha`` (default 0.95) and ``cvar_lambda`` (default 0.0).
        scenarios: sequence of Scenario dataclass instances or dicts each
            providing ``probability`` and a ``prices`` vector (directly or
            nested under ``data``).

    Returns:
        Pyomo ConcreteModel ready for solve.
    """
    _validate(params, scenarios)

    prices_per_s = [_scenario_prices(s) for s in scenarios]
    probs = [_scenario_probability(s) for s in scenarios]
    # Normalize probabilities defensively.
    psum = sum(probs)
    if psum <= 0:
        raise ValueError("scenario probabilities sum to <= 0")
    probs = [p / psum for p in probs]

    S = len(scenarios)
    T = len(prices_per_s[0])

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
    alpha = float(params.get("cvar_alpha", 0.95))
    lam = float(params.get("cvar_lambda", 0.0))

    m = pyo.ConcreteModel(name="stochastic_dispatch_m2")

    # Sets
    m.T = pyo.RangeSet(0, T - 1)
    m.S = pyo.RangeSet(0, S - 1)

    # Params
    m.dt = pyo.Param(initialize=dt, mutable=True)
    m.price = pyo.Param(
        m.T,
        m.S,
        initialize={(t, s): prices_per_s[s][t] for s in range(S) for t in range(T)},
        mutable=True,
    )
    m.pi = pyo.Param(m.S, initialize={s: probs[s] for s in range(S)}, mutable=True)
    m.cap = pyo.Param(initialize=cap)
    m.p_chg_max = pyo.Param(initialize=p_chg_max)
    m.p_dis_max = pyo.Param(initialize=p_dis_max)
    m.soc_min = pyo.Param(initialize=soc_min)
    m.soc_max = pyo.Param(initialize=soc_max)
    m.eta_c = pyo.Param(initialize=eta_c)
    m.eta_d = pyo.Param(initialize=eta_d)
    m.soc_0 = pyo.Param(initialize=soc_0)
    m.soc_terminal = pyo.Param(initialize=soc_terminal)
    m.alpha = pyo.Param(initialize=alpha)
    m.lam = pyo.Param(initialize=lam)

    # Per-scenario decision variables
    m.p_charge = pyo.Var(m.T, m.S, domain=pyo.NonNegativeReals, bounds=(0, p_chg_max))
    m.p_discharge = pyo.Var(m.T, m.S, domain=pyo.NonNegativeReals, bounds=(0, p_dis_max))
    m.soc = pyo.Var(m.T, m.S, domain=pyo.NonNegativeReals, bounds=(soc_min, soc_max))
    m.is_charging = pyo.Var(m.T, m.S, domain=pyo.Binary)

    # Mutual exclusion
    def _excl_chg(model, t, s):
        return model.p_charge[t, s] <= model.p_chg_max * model.is_charging[t, s]

    def _excl_dis(model, t, s):
        return model.p_discharge[t, s] <= model.p_dis_max * (1 - model.is_charging[t, s])

    m.excl_charge = pyo.Constraint(m.T, m.S, rule=_excl_chg)
    m.excl_discharge = pyo.Constraint(m.T, m.S, rule=_excl_dis)

    # SOC dynamics (per scenario, sharing the same initial SOC)
    def _soc_dyn(model, t, s):
        if t == 0:
            prev = model.soc_0
        else:
            prev = model.soc[t - 1, s]
        return (
            model.soc[t, s]
            == prev
            + model.eta_c * model.p_charge[t, s] * model.dt
            - model.p_discharge[t, s] * model.dt / model.eta_d
        )

    m.soc_dynamics = pyo.Constraint(m.T, m.S, rule=_soc_dyn)

    # Terminal SOC (per scenario)
    def _terminal(model, s):
        return model.soc[T - 1, s] >= model.soc_terminal

    m.terminal_soc_con = pyo.Constraint(m.S, rule=_terminal)

    # Non-anticipativity at t=0: tie all scenarios' stage-1 decisions to s=0.
    if S >= 2:

        def _na_chg(model, s):
            if s == 0:
                return pyo.Constraint.Skip
            return model.p_charge[0, s] == model.p_charge[0, 0]

        def _na_dis(model, s):
            if s == 0:
                return pyo.Constraint.Skip
            return model.p_discharge[0, s] == model.p_discharge[0, 0]

        def _na_bin(model, s):
            if s == 0:
                return pyo.Constraint.Skip
            return model.is_charging[0, s] == model.is_charging[0, 0]

        m.na_charge = pyo.Constraint(m.S, rule=_na_chg)
        m.na_discharge = pyo.Constraint(m.S, rule=_na_dis)
        m.na_is_charging = pyo.Constraint(m.S, rule=_na_bin)

    # Per-scenario cost expression
    def _cost_s(model, s):
        return sum(
            model.price[t, s] * (model.p_charge[t, s] - model.p_discharge[t, s]) * model.dt
            for t in model.T
        )

    m.cost_s = pyo.Expression(m.S, rule=_cost_s)

    # Expected cost expression
    m.expected_cost = pyo.Expression(expr=sum(m.pi[s] * m.cost_s[s] for s in m.S))

    # CVaR via Rockafellar-Uryasev:
    #   CVaR = eta + (1 / (1 - alpha)) * sum_s pi_s * z_s
    #   z_s >= cost_s - eta,  z_s >= 0
    m.eta = pyo.Var(domain=pyo.Reals)
    m.z = pyo.Var(m.S, domain=pyo.NonNegativeReals)

    def _cvar_z(model, s):
        return model.z[s] >= model.cost_s[s] - model.eta

    m.cvar_z_con = pyo.Constraint(m.S, rule=_cvar_z)

    # CVaR expression. For alpha == 0 we collapse the (1/(1-alpha)) factor to 1
    # which yields E[z]; combined with z >= cost - eta and free eta, the
    # minimizer drives eta -> max(cost) and z_s -> 0, so CVaR == eta. To make
    # alpha == 0 truly equivalent to E[cost] minimization (per spec), we treat
    # alpha == 0 as "no risk weighting": CVaR_0 := E[cost].
    if alpha <= 0.0:
        m.cvar = pyo.Expression(expr=m.expected_cost)
    else:
        inv_q = 1.0 / (1.0 - alpha)
        m.cvar = pyo.Expression(expr=m.eta + inv_q * sum(m.pi[s] * m.z[s] for s in m.S))

    # Objective
    m.total_cost = pyo.Objective(expr=m.expected_cost + m.lam * m.cvar, sense=pyo.minimize)

    return m
