"""
Multi-resource fleet dispatch MILP formulation (Milestone 4).

Extends the M1/M2 single-battery formulation to a fleet of batteries sharing a
site-level interconnection (feeder). Per-resource SOC dynamics with mutex
charge/discharge are preserved; resources are coupled at the aggregate via:

    - Aggregate variable ``p_aggregate[t] = sum_r (p_discharge[r,t] - p_charge[r,t])``
      (grid-export-positive convention: positive means the site exports to grid).
    - Feeder import/export split with mutex (analogous to the M2 tariff pattern):
      ``p_export[t] - p_import[t] == p_aggregate[t]`` with ``p_export, p_import >= 0``
      and binary ``import_mode[t]`` enforcing mutex with big-M.
    - Optional feeder import/export caps.
    - Optional reserve (ancillary services) headroom: aggregate spare capacity
      at each binding timestep must meet ``reserve_capacity_kw``.

Stable Var/Param surface (M4 contract — fleet hooks may rely on these names):

    Sets / Params:
        m.R           : Set of resource ids (strings)
        m.T           : RangeSet(0, T-1)
        m.dt          : Param (mutable) - hours per step
        m.price[t]    : Param (mutable)
        m.cap[r], m.p_chg_max[r], m.p_dis_max[r], m.soc_min[r], m.soc_max[r]
        m.eta_c[r], m.eta_d[r], m.soc_0[r]
        m.feeder_max_import (or None), m.feeder_max_export (or None)
        m.reserve_capacity_kw, m.reserve_window[t] (Param 0/1, optional)

    Vars:
        m.p_charge[r,t]    NonNegativeReals
        m.p_discharge[r,t] NonNegativeReals
        m.soc[r,t]         in [soc_min*cap, soc_max*cap]
        m.is_charging[r,t] Binary
        m.p_aggregate[t]   Reals (free; sign per export convention)
        m.p_import[t]      NonNegativeReals
        m.p_export[t]      NonNegativeReals
        m.import_mode[t]   Binary

    Expressions:
        m.energy_cost      sum_t price[t] * p_import[t] * dt
                                       - price[t] * p_export[t] * dt
        m.load_kw[t], m.solar_kw[t]    Optional Params if supplied.

Hook API (M4):
    objective_terms     : list of (model, params) -> Pyomo expression. Summed
                          into the objective.
    constraint_builders : list of (model, params) -> None. Run BEFORE objective
                          declaration so they can introduce auxiliary vars
                          referenced by objective hooks.

The reserve formulation here is "available headroom" — the sum across resources
of the un-used charge/discharge power capacity must exceed
``reserve_capacity_kw`` at each binding timestep. This conflates up- and
down-regulation; M5 will refine to per-direction reserves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import pyomo.environ as pyo


@dataclass
class FleetBattery:
    id: str
    capacity_kwh: float
    max_charge_kw: float
    max_discharge_kw: float
    soc_init: float
    soc_min: float = 0.05
    soc_max: float = 0.95
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    terminal_soc: Optional[float] = None  # defaults to soc_init


@dataclass
class FleetCoupling:
    feeder_max_import_kw: Optional[float] = None
    feeder_max_export_kw: Optional[float] = None
    reserve_capacity_kw: float = 0.0
    reserve_window: Optional[List[bool]] = None


ObjectiveTerm = Callable[[pyo.ConcreteModel, Dict[str, Any]], Any]
ConstraintBuilder = Callable[[pyo.ConcreteModel, Dict[str, Any]], None]


def _validate_battery(b: FleetBattery) -> None:
    if b.capacity_kwh <= 0:
        raise ValueError(f"battery {b.id}: capacity must be > 0")
    if b.max_charge_kw <= 0 or b.max_discharge_kw <= 0:
        raise ValueError(f"battery {b.id}: max charge/discharge must be > 0")
    if not (0 <= b.soc_min < b.soc_max <= 1):
        raise ValueError(
            f"battery {b.id}: soc_min/soc_max invalid [{b.soc_min}, {b.soc_max}]"
        )
    if not (b.soc_min <= b.soc_init <= b.soc_max):
        raise ValueError(
            f"battery {b.id}: soc_init {b.soc_init} not in [{b.soc_min},{b.soc_max}]"
        )
    for v, n in ((b.eta_charge, "eta_charge"), (b.eta_discharge, "eta_discharge")):
        if not (0 < v <= 1):
            raise ValueError(f"battery {b.id}: {n} must be in (0,1], got {v}")


def build_fleet_dispatch_model(
    batteries: List[FleetBattery],
    horizon_steps: int,
    dt_hours: float,
    prices: List[float],
    load_kw: Optional[List[float]] = None,
    solar_kw: Optional[List[float]] = None,
    coupling: Optional[FleetCoupling] = None,
    objective_terms: Optional[Sequence[ObjectiveTerm]] = None,
    constraint_builders: Optional[Sequence[ConstraintBuilder]] = None,
) -> pyo.ConcreteModel:
    """Build a Pyomo ConcreteModel for multi-resource fleet dispatch."""
    if not batteries:
        raise ValueError("batteries must be non-empty")
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be > 0")
    if dt_hours <= 0:
        raise ValueError("dt_hours must be > 0")
    prices = list(prices)
    if len(prices) < horizon_steps:
        prices = prices + [prices[-1]] * (horizon_steps - len(prices))
    prices = prices[:horizon_steps]

    for b in batteries:
        _validate_battery(b)
    ids = [b.id for b in batteries]
    if len(set(ids)) != len(ids):
        raise ValueError("battery ids must be unique")

    coupling = coupling or FleetCoupling()
    T = horizon_steps

    m = pyo.ConcreteModel(name="fleet_dispatch_m4")

    m.R = pyo.Set(initialize=ids, ordered=True)
    m.T = pyo.RangeSet(0, T - 1)

    # Mutable params
    m.price = pyo.Param(m.T, initialize={t: prices[t] for t in range(T)}, mutable=True)
    m.dt = pyo.Param(initialize=dt_hours, mutable=True)

    # Per-resource params
    by_id = {b.id: b for b in batteries}
    m.cap = pyo.Param(m.R, initialize={r: by_id[r].capacity_kwh for r in ids})
    m.p_chg_max = pyo.Param(m.R, initialize={r: by_id[r].max_charge_kw for r in ids})
    m.p_dis_max = pyo.Param(m.R, initialize={r: by_id[r].max_discharge_kw for r in ids})
    m.soc_min = pyo.Param(
        m.R, initialize={r: by_id[r].soc_min * by_id[r].capacity_kwh for r in ids}
    )
    m.soc_max = pyo.Param(
        m.R, initialize={r: by_id[r].soc_max * by_id[r].capacity_kwh for r in ids}
    )
    m.eta_c = pyo.Param(m.R, initialize={r: by_id[r].eta_charge for r in ids})
    m.eta_d = pyo.Param(m.R, initialize={r: by_id[r].eta_discharge for r in ids})
    m.soc_0 = pyo.Param(
        m.R, initialize={r: by_id[r].soc_init * by_id[r].capacity_kwh for r in ids}
    )
    m.soc_terminal = pyo.Param(
        m.R,
        initialize={
            r: (by_id[r].terminal_soc if by_id[r].terminal_soc is not None
                else by_id[r].soc_init) * by_id[r].capacity_kwh
            for r in ids
        },
    )

    # Coupling caps stored as attributes (None-allowed); also exposed as
    # plain numeric attributes for hook introspection.
    m.feeder_max_import = coupling.feeder_max_import_kw
    m.feeder_max_export = coupling.feeder_max_export_kw
    m.reserve_capacity_kw = float(coupling.reserve_capacity_kw or 0.0)
    if coupling.reserve_window is not None:
        rw = list(coupling.reserve_window)
        if len(rw) < T:
            rw = rw + [False] * (T - len(rw))
        rw = rw[:T]
        m.reserve_window = pyo.Param(
            m.T, initialize={t: 1 if rw[t] else 0 for t in range(T)}
        )
    else:
        # Default: reserve binding at every timestep iff capacity > 0
        flag = 1 if m.reserve_capacity_kw > 0 else 0
        m.reserve_window = pyo.Param(m.T, initialize={t: flag for t in range(T)})

    # Optional load/solar surface
    if load_kw is not None:
        ld = list(load_kw)
        if len(ld) < T:
            ld = ld + [ld[-1]] * (T - len(ld))
        ld = ld[:T]
        m.load_kw = pyo.Param(
            m.T, initialize={t: float(ld[t]) for t in range(T)}, mutable=True
        )
    if solar_kw is not None:
        sl = list(solar_kw)
        if len(sl) < T:
            sl = sl + [sl[-1]] * (T - len(sl))
        sl = sl[:T]
        m.solar_kw = pyo.Param(
            m.T, initialize={t: float(sl[t]) for t in range(T)}, mutable=True
        )

    # Variables
    def _chg_bounds(model, r, t):
        return (0.0, float(pyo.value(model.p_chg_max[r])))

    def _dis_bounds(model, r, t):
        return (0.0, float(pyo.value(model.p_dis_max[r])))

    def _soc_bounds(model, r, t):
        return (float(pyo.value(model.soc_min[r])), float(pyo.value(model.soc_max[r])))

    m.p_charge = pyo.Var(m.R, m.T, domain=pyo.NonNegativeReals, bounds=_chg_bounds)
    m.p_discharge = pyo.Var(m.R, m.T, domain=pyo.NonNegativeReals, bounds=_dis_bounds)
    m.soc = pyo.Var(m.R, m.T, domain=pyo.NonNegativeReals, bounds=_soc_bounds)
    m.is_charging = pyo.Var(m.R, m.T, domain=pyo.Binary)

    # Aggregate / feeder vars
    # Big-M for import/export = sum of all max powers (an upper bound on |aggregate|).
    big_m = sum(b.max_discharge_kw + b.max_charge_kw for b in batteries) + 1.0
    if coupling.feeder_max_import_kw is not None:
        big_m_imp = float(coupling.feeder_max_import_kw)
    else:
        big_m_imp = big_m
    if coupling.feeder_max_export_kw is not None:
        big_m_exp = float(coupling.feeder_max_export_kw)
    else:
        big_m_exp = big_m

    m.p_aggregate = pyo.Var(m.T, domain=pyo.Reals)
    m.p_import = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, big_m_imp))
    m.p_export = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, big_m_exp))
    m.import_mode = pyo.Var(m.T, domain=pyo.Binary)

    # Per-resource mutex
    def _excl_chg(model, r, t):
        return model.p_charge[r, t] <= model.p_chg_max[r] * model.is_charging[r, t]

    def _excl_dis(model, r, t):
        return model.p_discharge[r, t] <= model.p_dis_max[r] * (1 - model.is_charging[r, t])

    m.excl_charge = pyo.Constraint(m.R, m.T, rule=_excl_chg)
    m.excl_discharge = pyo.Constraint(m.R, m.T, rule=_excl_dis)

    # Per-resource SOC dynamics
    def _soc_dyn(model, r, t):
        prev = model.soc_0[r] if t == 0 else model.soc[r, t - 1]
        return (
            model.soc[r, t]
            == prev
            + model.eta_c[r] * model.p_charge[r, t] * model.dt
            - model.p_discharge[r, t] * model.dt / model.eta_d[r]
        )

    m.soc_dynamics = pyo.Constraint(m.R, m.T, rule=_soc_dyn)

    # Per-resource terminal SOC
    def _terminal(model, r):
        return model.soc[r, T - 1] >= model.soc_terminal[r]

    m.terminal_soc_con = pyo.Constraint(m.R, rule=_terminal)

    # Aggregate definition (export-positive)
    def _agg(model, t):
        return model.p_aggregate[t] == sum(
            model.p_discharge[r, t] - model.p_charge[r, t] for r in model.R
        )

    m.aggregate_def = pyo.Constraint(m.T, rule=_agg)

    # Import/export decomposition with mutex
    def _imp_exp_balance(model, t):
        return model.p_export[t] - model.p_import[t] == model.p_aggregate[t]

    m.imp_exp_balance = pyo.Constraint(m.T, rule=_imp_exp_balance)

    def _imp_mutex(model, t):
        return model.p_import[t] <= big_m_imp * model.import_mode[t]

    def _exp_mutex(model, t):
        return model.p_export[t] <= big_m_exp * (1 - model.import_mode[t])

    m.import_mutex = pyo.Constraint(m.T, rule=_imp_mutex)
    m.export_mutex = pyo.Constraint(m.T, rule=_exp_mutex)

    # Feeder caps
    if coupling.feeder_max_import_kw is not None:
        cap_imp = float(coupling.feeder_max_import_kw)

        def _imp_cap(model, t):
            return model.p_import[t] <= cap_imp

        m.feeder_import_cap = pyo.Constraint(m.T, rule=_imp_cap)

    if coupling.feeder_max_export_kw is not None:
        cap_exp = float(coupling.feeder_max_export_kw)

        def _exp_cap(model, t):
            return model.p_export[t] <= cap_exp

        m.feeder_export_cap = pyo.Constraint(m.T, rule=_exp_cap)

    # Reserve headroom (available headroom formulation)
    if m.reserve_capacity_kw > 0:
        rcap = m.reserve_capacity_kw

        def _reserve(model, t):
            if int(pyo.value(model.reserve_window[t])) == 0:
                return pyo.Constraint.Skip
            headroom = sum(
                (model.p_chg_max[r] - model.p_charge[r, t])
                + (model.p_dis_max[r] - model.p_discharge[r, t])
                for r in model.R
            )
            return headroom >= rcap

        m.reserve_headroom = pyo.Constraint(m.T, rule=_reserve)

    # ---- M4 hook: constraint_builders run before objective is declared ----
    params_view: Dict[str, Any] = {
        "batteries": batteries,
        "horizon_steps": T,
        "dt_hours": dt_hours,
        "prices": prices,
        "load_kw": load_kw,
        "solar_kw": solar_kw,
        "coupling": coupling,
    }
    if constraint_builders:
        for i, builder in enumerate(constraint_builders):
            try:
                builder(m, params_view)
            except Exception as e:
                raise RuntimeError(
                    f"constraint_builder #{i} ({getattr(builder, '__name__', builder)}) failed: {e}"
                ) from e

    # Base energy cost: pay for imports, get paid for exports at price[t].
    base_cost_expr = sum(
        m.price[t] * (m.p_import[t] - m.p_export[t]) * m.dt for t in m.T
    )
    m.energy_cost = pyo.Expression(expr=base_cost_expr)

    extra_terms = []
    if objective_terms:
        for i, term in enumerate(objective_terms):
            try:
                e = term(m, params_view)
            except Exception as exc:
                raise RuntimeError(
                    f"objective_term #{i} ({getattr(term, '__name__', term)}) failed: {exc}"
                ) from exc
            extra_terms.append(e)

    if extra_terms:
        m.cost = pyo.Objective(
            expr=m.energy_cost + sum(extra_terms), sense=pyo.minimize
        )
    else:
        m.cost = pyo.Objective(expr=m.energy_cost, sense=pyo.minimize)

    return m


__all__ = [
    "FleetBattery",
    "FleetCoupling",
    "build_fleet_dispatch_model",
]
