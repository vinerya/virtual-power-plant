"""Tariff -> optimizer adapter (Milestone 2).

Bridges the URDB-shaped :class:`vpp.tariffs.Tariff` to the Pyomo dispatch model
defined in :mod:`vpp.optimization.formulations.dispatch`.

Design summary
--------------
The base dispatch model in M1 prices net grid draw at a flat ``m.price[t]``
vector. Real utility tariffs are richer: TOU energy + tiered blocks +
demand charges + fixed/min components, plus an asymmetric export
compensation regime (NEM 2.0 retail-rate vs. NEM 3.0 avoided cost).

This module supplies:

  * :func:`tariff_to_opt_params` — walks the horizon and emits per-step
    buy/sell prices and a demand-charge mask from a :class:`Tariff`.
  * :func:`add_tariff_constraints` — constraint builder hook that adds
    ``p_import[t]``, ``p_export[t]`` aux Vars with a big-M mutex and the
    grid balance ``p_import - p_export = p_charge - p_discharge + load -
    solar``.
  * :func:`add_tariff_energy_term` — objective term for
    ``Σ buy[t]·p_import[t]·dt - sell[t]·p_export[t]·dt``.
  * :func:`add_demand_charge_terms` — adds a single ``peak_demand_kw``
    aux Var + window constraints + ratchet floor; appends
    ``rate · peak_demand_kw + fixed_$`` to the objective.
  * :func:`load_nem3_avoided_cost_2024` — representative California
    avoided-cost vector hardcoded from the CPUC 2024 ACC table.

IMPORTANT — pair the two tariff hooks
-------------------------------------
``add_tariff_constraints`` and ``add_tariff_energy_term`` MUST be passed
together: the term references the aux Vars (``p_import``, ``p_export``)
the builder creates. The convenience helper :func:`build_tariff_hooks`
returns both as a ready-to-pass tuple.

Avoiding double-counting
------------------------
The base dispatch model still computes ``m.energy_cost`` from
``m.price[t]``. Callers using the tariff term should set
``params['disable_base_energy_cost'] = True`` (preferred) — see the
optional flag added to ``build_battery_dispatch_model`` — or pass an
all-zero ``prices`` vector of length T.

MILP cost note
--------------
:func:`add_demand_charge_terms` introduces one aux scalar
(``peak_demand_kw``) and ``|window|`` linear inequalities. This is what
makes demand charges cheap to add but tightens the LP relaxation
considerably; expect solver time to grow with horizon length.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from .components import (
    DemandCharge,
    FixedCharge,
    MinimumBill,
    TieredEnergyRate,
    TimeOfUseRate,
)
from .tariff import Tariff


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------


@dataclass
class TariffOptParams:
    """Tariff parameters resolved against an optimization horizon."""

    energy_buy_per_kwh: List[float]   # length T
    energy_sell_per_kwh: List[float]  # length T
    demand_charge_per_kw: Optional[float] = None
    demand_charge_window: Optional[List[bool]] = None  # length T mask
    demand_ratchet_floor_kw: float = 0.0
    fixed_charges: float = 0.0
    min_bill: float = 0.0
    # Echoed metadata, useful for diagnostics:
    interval_hours: float = 0.25
    horizon_steps: int = 0


# ---------------------------------------------------------------------------
# Tariff -> per-timestep parameter resolution
# ---------------------------------------------------------------------------


def _classify_tou_rate(tou: TimeOfUseRate, dt_local: datetime) -> float:
    """Return the $/kWh rate at dt_local from a TimeOfUseRate, or 0.0 if no match."""
    label = tou._classify(dt_local)
    if label is None:
        return 0.0
    for sch in tou.periods[label]:
        if dt_local.month in sch.season_mask:
            return float(sch.rate)
    return 0.0


def _demand_in_window(dc: DemandCharge, dt_local: datetime) -> bool:
    return dc._in_window(dt_local)


def tariff_to_opt_params(
    tariff: Tariff,
    horizon_start: datetime,
    horizon_hours: int,
    interval_minutes: int = 15,
    nem: Literal["none", "nem2", "nem3"] = "nem2",
    nem3_avoided_cost: Optional[List[float]] = None,
    prior_demand_max_kw: float = 0.0,
    tz: timezone = timezone.utc,
) -> TariffOptParams:
    """Project a :class:`Tariff` onto a discrete optimization horizon.

    Walks each timestep and resolves the active TOU rate (or tiered rate's
    lowest tier — see note) plus the demand-charge window mask.

    Parameters
    ----------
    tariff : Tariff
        Composed tariff with energy + demand + fixed components.
    horizon_start : datetime
        First interval-start (timezone-aware preferred).
    horizon_hours : int
        Horizon length in hours.
    interval_minutes : int
        Discretization step (typical 15 or 60).
    nem : {'none','nem2','nem3'}
        Export compensation regime.
        * ``nem2`` : exports earn the same per-kWh price as imports.
        * ``nem3`` : exports earn ``nem3_avoided_cost[t mod len]``
          (must be supplied).
        * ``none`` : exports earn $0/kWh.
    nem3_avoided_cost : list[float] or None
        Hourly $/kWh avoided cost vector. If shorter than horizon, repeats.
    prior_demand_max_kw : float
        Highest billed demand in the prior 11 months. Combined with the
        first ratchet-bearing :class:`DemandCharge` in the tariff (if any)
        to compute ``demand_ratchet_floor_kw = ratchet_pct * prior``.

    Tier handling (note)
    --------------------
    Tier prices depend on cumulative kWh in a billing period and so are
    not state-decoupled in the optimization horizon. M2 uses the LOWEST
    (tier-0) rate — appropriate for short-horizon dispatch where the
    customer is unlikely to cross a tier within the day. For
    full-month optimization with high-baseline customers a tier-aware
    formulation (piecewise SOS2 over cumulative import) is required;
    that is M3+ work.
    """
    if horizon_start.tzinfo is None:
        horizon_start = horizon_start.replace(tzinfo=timezone.utc)
    dt_h = interval_minutes / 60.0
    T = int(round(horizon_hours / dt_h))
    step = timedelta(minutes=interval_minutes)

    # Pull components out of the tariff.
    tou_components: List[TimeOfUseRate] = [
        c for c in tariff.components if isinstance(c, TimeOfUseRate)
    ]
    tier_components: List[TieredEnergyRate] = [
        c for c in tariff.components if isinstance(c, TieredEnergyRate)
    ]
    demand_components: List[DemandCharge] = [
        c for c in tariff.components if isinstance(c, DemandCharge)
    ]
    fixed_components: List[FixedCharge] = [
        c for c in tariff.components if isinstance(c, FixedCharge)
    ]
    min_components: List[MinimumBill] = [
        c for c in tariff.components if isinstance(c, MinimumBill)
    ]

    # Prefer TOU; fall back to tier-0 of the first TieredEnergyRate; else 0.
    tier0_rate: Optional[float] = None
    if tier_components:
        tiers = tier_components[0].tiers
        if tiers:
            tier0_rate = float(tiers[0][1])

    energy_buy: List[float] = []
    for t in range(T):
        ts = (horizon_start + t * step).astimezone(tz)
        rate = 0.0
        for tou in tou_components:
            r = _classify_tou_rate(tou, ts)
            if r > rate:
                rate = r
        if rate == 0.0 and tier0_rate is not None:
            rate = tier0_rate
        energy_buy.append(rate)

    # Sell prices.
    if nem == "nem2":
        energy_sell = list(energy_buy)
    elif nem == "nem3":
        if not nem3_avoided_cost:
            raise ValueError("nem='nem3' requires a nem3_avoided_cost vector")
        ac = list(nem3_avoided_cost)
        energy_sell = []
        for t in range(T):
            # Map step index to hour index in the source vector.
            hour_idx = int((horizon_start + t * step).astimezone(tz).hour)
            # Allow a multi-day vector by also indexing day-of-horizon hours.
            full_h_idx = int(t * dt_h) if len(ac) >= T * dt_h else hour_idx
            energy_sell.append(float(ac[full_h_idx % len(ac)]))
    else:  # "none"
        energy_sell = [0.0] * T

    # Demand charge: choose first DemandCharge component, build window mask.
    demand_rate: Optional[float] = None
    demand_window: Optional[List[bool]] = None
    ratchet_floor = 0.0
    if demand_components:
        dc = demand_components[0]
        demand_rate = float(dc.rate)
        demand_window = [
            _demand_in_window(dc, (horizon_start + t * step).astimezone(tz))
            for t in range(T)
        ]
        if dc.ratchet_pct > 0 and prior_demand_max_kw > 0:
            ratchet_floor = float(dc.ratchet_pct) * float(prior_demand_max_kw)

    # Fixed charges amortized across the horizon.
    horizon_days = horizon_hours / 24.0
    fixed_total = 0.0
    for fc in fixed_components:
        if fc.frequency == "daily":
            fixed_total += fc.amount * horizon_days
        else:  # monthly
            fixed_total += fc.amount * (horizon_days / 30.4375)

    min_bill = min_components[0].amount if min_components else 0.0

    return TariffOptParams(
        energy_buy_per_kwh=energy_buy,
        energy_sell_per_kwh=energy_sell,
        demand_charge_per_kw=demand_rate,
        demand_charge_window=demand_window,
        demand_ratchet_floor_kw=ratchet_floor,
        fixed_charges=fixed_total,
        min_bill=min_bill,
        interval_hours=dt_h,
        horizon_steps=T,
    )


# ---------------------------------------------------------------------------
# Optimizer hooks
# ---------------------------------------------------------------------------


def add_tariff_constraints(opt_params: TariffOptParams) -> Callable:
    """Constraint-builder hook: adds ``p_import``, ``p_export`` and the
    grid balance + import/export mutex.

    The grid balance is

        p_import[t] - p_export[t]
            == (p_charge[t] - p_discharge[t]) + load[t] - solar[t]

    Both p_import and p_export are nonneg; mutex is enforced via a big-M
    binary so the LP cannot synthesize fictitious simultaneous import +
    export to harvest sell - buy spreads at hours when sell > buy.
    """
    import pyomo.environ as pyo

    T = opt_params.horizon_steps
    # Big-M: bound by max charge + load magnitude. We compute it inside the
    # builder once we have access to the model's params.

    def _builder(model, params: Dict[str, Any]) -> None:
        load_max = max(list(params.get("load") or [0.0]) + [0.0])
        solar_max = max(list(params.get("solar") or [0.0]) + [0.0])
        big_m = float(
            params["max_charge_kw"]
            + params["max_discharge_kw"]
            + load_max
            + solar_max
            + 1.0
        )
        model.p_import = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, big_m))
        model.p_export = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, big_m))
        model.is_importing = pyo.Var(model.T, domain=pyo.Binary)

        def _balance(mm, t):
            return (
                mm.p_import[t] - mm.p_export[t]
                == mm.p_charge[t] - mm.p_discharge[t] + mm.load_kw[t] - mm.solar_kw[t]
            )

        model.tariff_balance = pyo.Constraint(model.T, rule=_balance)

        def _mutex_imp(mm, t):
            return mm.p_import[t] <= big_m * mm.is_importing[t]

        def _mutex_exp(mm, t):
            return mm.p_export[t] <= big_m * (1 - mm.is_importing[t])

        model.tariff_mutex_imp = pyo.Constraint(model.T, rule=_mutex_imp)
        model.tariff_mutex_exp = pyo.Constraint(model.T, rule=_mutex_exp)

    _builder.__name__ = "add_tariff_constraints"
    return _builder


def add_tariff_energy_term(opt_params: TariffOptParams) -> Callable:
    """Objective-term hook: tariff energy cost = Σ_t (buy·p_import - sell·p_export)·dt.

    Must be paired with :func:`add_tariff_constraints`. Caller should set
    ``params['disable_base_energy_cost'] = True`` to suppress the M1
    flat-price energy term.
    """
    import pyomo.environ as pyo

    buy = list(opt_params.energy_buy_per_kwh)
    sell = list(opt_params.energy_sell_per_kwh)

    def _term(model, params):
        T_idx = list(model.T)
        if len(buy) != len(T_idx):
            raise ValueError(
                f"tariff buy vector length {len(buy)} != horizon {len(T_idx)}"
            )
        return sum(
            (buy[t] * model.p_import[t] - sell[t] * model.p_export[t]) * model.dt
            for t in T_idx
        )

    _term.__name__ = "add_tariff_energy_term"
    return _term


def add_demand_charge_terms(
    opt_params: TariffOptParams,
) -> Tuple[Callable, Callable]:
    """Return (objective_term, constraint_builder) for non-coincident demand.

    Variable structure
    ------------------
    * Adds one nonneg scalar ``peak_demand_kw``.
    * For each ``t`` with ``demand_charge_window[t] == True``:
      ``peak_demand_kw >= p_import[t]``.
    * Optionally pins ``peak_demand_kw >= demand_ratchet_floor_kw``.

    Objective contribution is ``rate · peak_demand_kw + fixed_charges``.
    The objective is a no-op (returns 0) when no demand rate is configured
    so this hook can be safely passed in all cases.

    MILP impact
    -----------
    One scalar Var + ``|window|`` linear inequalities. No new binaries.
    The big-M-style coupling to ``p_import[t]`` (which itself depends on
    a binary mutex from :func:`add_tariff_constraints`) is what makes
    demand-aware dispatch noticeably slower than energy-only dispatch:
    the LP relaxation gets very tight only after branching on the
    import/export disjunction.
    """
    import pyomo.environ as pyo

    rate = opt_params.demand_charge_per_kw
    window = opt_params.demand_charge_window
    floor_kw = float(opt_params.demand_ratchet_floor_kw)
    fixed_dollars = float(opt_params.fixed_charges)

    def _builder(model, _params):
        model.peak_demand_kw = pyo.Var(domain=pyo.NonNegativeReals)
        if rate is None or not window:
            # Still register the var so introspection works; just leave
            # unconstrained (will collapse to 0 in objective minimization).
            return

        T_idx = list(model.T)
        if len(window) != len(T_idx):
            raise ValueError(
                f"demand_charge_window length {len(window)} != horizon {len(T_idx)}"
            )

        def _peak(mm, t):
            if not window[t]:
                return pyo.Constraint.Skip
            return mm.peak_demand_kw >= mm.p_import[t]

        model.demand_peak_con = pyo.Constraint(model.T, rule=_peak)

        if floor_kw > 0:
            model.demand_ratchet_con = pyo.Constraint(
                expr=model.peak_demand_kw >= floor_kw
            )

    _builder.__name__ = "add_demand_charge_constraints"

    def _term(model, _params):
        if rate is None:
            return fixed_dollars
        return rate * model.peak_demand_kw + fixed_dollars

    _term.__name__ = "add_demand_charge_term"
    return _term, _builder


def build_tariff_hooks(
    opt_params: TariffOptParams,
) -> Tuple[List[Callable], List[Callable]]:
    """Convenience: build (objective_terms, constraint_builders) ready to pass
    to :func:`vpp.optimization.formulations.dispatch.build_battery_dispatch_model`.

    Includes tariff energy term, tariff balance/mutex builder, demand charge
    term + builder. Caller still needs to set
    ``params['disable_base_energy_cost'] = True`` to avoid double-counting.
    """
    energy_term = add_tariff_energy_term(opt_params)
    tariff_builder = add_tariff_constraints(opt_params)
    demand_term, demand_builder = add_demand_charge_terms(opt_params)
    return [energy_term, demand_term], [tariff_builder, demand_builder]


# ---------------------------------------------------------------------------
# NEM 3.0 avoided-cost reference vector
# ---------------------------------------------------------------------------


def load_nem3_avoided_cost_2024() -> List[float]:
    """Return a representative 24-hour $/kWh avoided-cost vector for NEM 3.0.

    Values are illustrative weekday averages drawn from the CPUC 2024 Avoided
    Cost Calculator (ACC) outputs for residential customers in PG&E
    territory, summer month. They are intentionally lower than retail rates
    (NEM 3.0 reduced the export-compensation rate by roughly 75% vs. NEM 2.0),
    with the highest values during the 4-9 PM ramp.

    Source: California Public Utilities Commission, 2024 Avoided Cost
    Calculator update, available at
    https://www.cpuc.ca.gov/industries-and-topics/electrical-energy/demand-side-management/energy-efficiency/idsm
    (see also the E3 ACC documentation,
    https://www.ethree.com/tools/avoided-cost-calculator-acc/).

    For M2 this is a single representative profile; production deployments
    should pull the monthly published vector from the live ACC data feed.
    """
    # 24 hourly $/kWh values, midnight-first.
    return [
        0.045, 0.040, 0.038, 0.038, 0.040, 0.045,  # 0-5
        0.052, 0.060, 0.058, 0.052, 0.045, 0.040,  # 6-11
        0.038, 0.038, 0.040, 0.045, 0.060, 0.110,  # 12-17  (ramp begins)
        0.180, 0.220, 0.180, 0.110, 0.070, 0.055,  # 18-23  (peak 7-8 PM)
    ]
