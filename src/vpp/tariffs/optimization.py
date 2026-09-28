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

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

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

    energy_buy_per_kwh: list[float]  # length T
    energy_sell_per_kwh: list[float]  # length T
    demand_charge_per_kw: float | None = None
    demand_charge_window: list[bool] | None = None  # length T mask
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
    nem3_avoided_cost: list[float] | None = None,
    prior_demand_max_kw: float = 0.0,
    tz: timezone = timezone.utc,
    live_price_overrides: list[Any] | None = None,
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
        * ``nem2`` : exports earn the same per-kWh price as imports, unless
          the tariff's TOU schedule defines an explicit URDB ``sell`` rate
          for that period, in which case that rate is used instead.
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
    tou_components: list[TimeOfUseRate] = [
        c for c in tariff.components if isinstance(c, TimeOfUseRate)
    ]
    tier_components: list[TieredEnergyRate] = [
        c for c in tariff.components if isinstance(c, TieredEnergyRate)
    ]
    demand_components: list[DemandCharge] = [
        c for c in tariff.components if isinstance(c, DemandCharge)
    ]
    fixed_components: list[FixedCharge] = [
        c for c in tariff.components if isinstance(c, FixedCharge)
    ]
    min_components: list[MinimumBill] = [
        c for c in tariff.components if isinstance(c, MinimumBill)
    ]

    # Prefer TOU; fall back to tier-0 of the first TieredEnergyRate; else 0.
    tier0_rate: float | None = None
    if tier_components:
        tiers = tier_components[0].tiers
        if tiers:
            tier0_rate = float(tiers[0][1])

    energy_buy: list[float] = []
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

    # ---- Live-price overrides (M4) -------------------------------------
    # Precedence: live > TOU > tier-0 fallback > 0.0.
    # A live override matches a horizon step when its timestamp falls within
    # the half-open interval [step_start, step_start + interval).
    live_override_steps: set = set()
    if live_price_overrides:
        for t in range(T):
            step_start = (horizon_start + t * step).astimezone(tz)
            step_end = step_start + step
            for pp in live_price_overrides:
                pp_ts = pp.timestamp
                if pp_ts.tzinfo is None:
                    pp_ts = pp_ts.replace(tzinfo=timezone.utc)
                pp_ts = pp_ts.astimezone(tz)
                if step_start <= pp_ts < step_end:
                    energy_buy[t] = float(pp.price_per_kwh)
                    live_override_steps.add(t)
                    break

    # Sell prices.
    if nem == "nem2":
        # Same-as-buy is the default -- and the only option for a step with
        # no TOU component, or one a live override already claimed (which
        # supersedes the static schedule for both buy and sell). When a TOU
        # period defines an explicit URDB `sell` rate that differs from
        # `rate`, prefer it: real NEM 2.0 per-period export compensation,
        # not a uniform buy-price proxy.
        energy_sell = list(energy_buy)
        for t in range(T):
            if t in live_override_steps:
                continue
            ts = (horizon_start + t * step).astimezone(tz)
            for tou in tou_components:
                r = tou.export_rate(ts)
                if r is not None:
                    energy_sell[t] = r
                    break
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
    demand_rate: float | None = None
    demand_window: list[bool] | None = None
    ratchet_floor = 0.0
    if demand_components:
        dc = demand_components[0]
        demand_rate = float(dc.rate)
        demand_window = [
            _demand_in_window(dc, (horizon_start + t * step).astimezone(tz)) for t in range(T)
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

    def _builder(model, params: dict[str, Any]) -> None:
        load_max = max(list(params.get("load") or [0.0]) + [0.0])
        solar_max = max(list(params.get("solar") or [0.0]) + [0.0])
        big_m = float(
            params["max_charge_kw"] + params["max_discharge_kw"] + load_max + solar_max + 1.0
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

    buy = list(opt_params.energy_buy_per_kwh)
    sell = list(opt_params.energy_sell_per_kwh)

    def _term(model, params):
        T_idx = list(model.T)
        if len(buy) != len(T_idx):
            raise ValueError(f"tariff buy vector length {len(buy)} != horizon {len(T_idx)}")
        return sum(
            (buy[t] * model.p_import[t] - sell[t] * model.p_export[t]) * model.dt for t in T_idx
        )

    _term.__name__ = "add_tariff_energy_term"
    return _term


def add_demand_charge_terms(
    opt_params: TariffOptParams,
) -> tuple[Callable, Callable]:
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
            raise ValueError(f"demand_charge_window length {len(window)} != horizon {len(T_idx)}")

        def _peak(mm, t):
            if not window[t]:
                return pyo.Constraint.Skip
            return mm.peak_demand_kw >= mm.p_import[t]

        model.demand_peak_con = pyo.Constraint(model.T, rule=_peak)

        if floor_kw > 0:
            model.demand_ratchet_con = pyo.Constraint(expr=model.peak_demand_kw >= floor_kw)

    _builder.__name__ = "add_demand_charge_constraints"

    def _term(model, _params):
        if rate is None:
            return fixed_dollars
        return rate * model.peak_demand_kw + fixed_dollars

    _term.__name__ = "add_demand_charge_term"
    return _term, _builder


def add_tiered_energy_term(
    tier_thresholds_kwh: list[float],
    tier_rates_per_kwh: list[float],
    use_native_sos2: bool = False,
) -> tuple[Callable, Callable]:
    """SOS2 piecewise-linear tier-aware energy cost.

    Replaces the M2 "lowest-tier" approximation. The piecewise-linear total
    energy cost ``C(cum_import)`` is encoded as a convex combination of
    breakpoint values selected by SOS2-constrained ``lambda`` weights.

    Mechanics
    ---------
    Let breakpoints be ``b_0 = 0 < b_1 < ... < b_K`` where ``b_1 .. b_{K-1}``
    are the supplied tier thresholds (the upper bound of each finite tier)
    and ``b_K`` is a finite cap derived from the horizon (since the last
    tier's threshold is conventionally ``+inf``). The cumulative cost
    ``c_k = sum_{i=1..k} (b_i - b_{i-1}) * rate_i`` is precomputed.

    For each step the model carries:

    * ``cum_import_kwh`` — scalar ``= sum_t p_import[t] * dt``.
    * ``lambda_b[k]`` ∈ [0, 1] for each breakpoint, with
      ``sum_k lambda_b[k] == 1``.
    * ``cum_import_kwh == sum_k b_k * lambda_b[k]``.
    * ``SOSConstraint(sos=2)`` over ``lambda_b`` so at most two adjacent
      ``lambda_b[k]`` may be nonzero — that is exactly the convex
      combination of two adjacent breakpoints, which traces the
      piecewise-linear curve.

    Energy contribution to the objective:
        ``sum_k c_k * lambda_b[k]``

    Horizon assumption
    ------------------
    The cumulative variable spans the WHOLE optimization horizon — so the
    SOS2 path is correct only when the horizon covers (approximately) one
    URDB billing month. For sub-month horizons callers should fall back to
    a fixed-tier rate via :func:`tariff_to_opt_params`.

    Returns
    -------
    (objective_term, constraint_builder)
    """
    import pyomo.environ as pyo

    if not tier_thresholds_kwh or not tier_rates_per_kwh:
        raise ValueError("tier_thresholds_kwh and tier_rates_per_kwh must be non-empty")
    if len(tier_thresholds_kwh) != len(tier_rates_per_kwh):
        raise ValueError("tier_thresholds_kwh / tier_rates_per_kwh length mismatch")

    # Build finite breakpoints. The last threshold is conventionally +inf;
    # replace with a generous cap = 2 * (next-to-last threshold or 1000).
    finite_thresholds: list[float] = []
    for th in tier_thresholds_kwh:
        if th == float("inf"):
            break
        finite_thresholds.append(float(th))

    # Determine cap for the last (open) tier. Heuristic: 2x the highest
    # finite threshold; if none, default to 10_000 kWh.
    if finite_thresholds:
        cap = max(2.0 * finite_thresholds[-1], finite_thresholds[-1] + 1000.0)
    else:
        cap = 10_000.0

    # Breakpoints: b_0 = 0, then each finite threshold, then cap.
    breakpoints: list[float] = [0.0] + list(finite_thresholds)
    if breakpoints[-1] < cap:
        breakpoints.append(cap)

    # Costs at each breakpoint via cumulative integral.
    costs: list[float] = [0.0]
    for k in range(1, len(breakpoints)):
        rate_k = float(tier_rates_per_kwh[min(k - 1, len(tier_rates_per_kwh) - 1)])
        costs.append(costs[-1] + (breakpoints[k] - breakpoints[k - 1]) * rate_k)

    K = len(breakpoints)

    def _builder(model, params: dict[str, Any]) -> None:
        # cum_import = sum_t p_import[t] * dt
        model.cum_import_kwh = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.0, breakpoints[-1]))
        model.tier_lambda = pyo.Var(range(K), domain=pyo.NonNegativeReals, bounds=(0.0, 1.0))

        def _cum(mm):
            return mm.cum_import_kwh == sum(mm.p_import[t] * mm.dt for t in mm.T)

        model.tier_cum_def = pyo.Constraint(rule=_cum)

        def _convex(mm):
            return sum(mm.tier_lambda[k] for k in range(K)) == 1.0

        model.tier_convex = pyo.Constraint(rule=_convex)

        def _bp(mm):
            return mm.cum_import_kwh == sum(breakpoints[k] * mm.tier_lambda[k] for k in range(K))

        model.tier_breakpoint = pyo.Constraint(rule=_bp)

        # SOS2 — at most two adjacent lambdas nonzero.
        # Two equivalent encodings are available; pick one based on solver
        # capability:
        #   * Native pyomo.SOSConstraint(sos=2) (Gurobi/CPLEX/SCIP).
        #   * Binary-segment "adjacent indicator" reformulation that
        #     enforces the same SOS2 property purely with MILP variables —
        #     this works with HiGHS, CBC, GLPK, etc.
        # Default is the binary encoding (broadly compatible). Pass
        # ``use_native_sos2=True`` for SOS-aware solvers.
        if use_native_sos2:
            model.tier_sos2 = pyo.SOSConstraint(
                var=model.tier_lambda,
                index=list(range(K)),
                sos=2,
            )
        # Binary z[s] = 1 iff cum_import lies in segment s = [b_s, b_{s+1}].
        n_seg = K - 1
        if n_seg > 0:
            model.tier_seg = pyo.Var(range(n_seg), domain=pyo.Binary)

            def _one_seg(mm):
                return sum(mm.tier_seg[s] for s in range(n_seg)) == 1

            model.tier_one_seg = pyo.Constraint(rule=_one_seg)

            def _lambda_bound(mm, k):
                # lambda[k] is allowed only if segment k-1 or k is active.
                left = mm.tier_seg[k - 1] if k - 1 >= 0 else 0
                right = mm.tier_seg[k] if k < n_seg else 0
                return mm.tier_lambda[k] <= left + right

            model.tier_lambda_bound = pyo.Constraint(range(K), rule=_lambda_bound)

    _builder.__name__ = "add_tiered_energy_constraints"

    def _term(model, _params):
        return sum(costs[k] * model.tier_lambda[k] for k in range(K))

    _term.__name__ = "add_tiered_energy_term"
    # Stash breakpoint metadata so callers/tests can introspect.
    _term.breakpoints = breakpoints  # type: ignore[attr-defined]
    _term.costs = costs  # type: ignore[attr-defined]
    return _term, _builder


def build_tariff_hooks(
    opt_params: TariffOptParams,
    tariff: Tariff | None = None,
) -> tuple[list[Callable], list[Callable]]:
    """Convenience: build (objective_terms, constraint_builders) ready to pass
    to :func:`vpp.optimization.formulations.dispatch.build_battery_dispatch_model`.

    Includes tariff energy term, tariff balance/mutex builder, demand charge
    term + builder. Caller still needs to set
    ``params['disable_base_energy_cost'] = True`` to avoid double-counting.

    SOS2 tier hook (M3)
    -------------------
    If ``tariff`` is supplied AND it carries a :class:`TieredEnergyRate`,
    a tier-aware SOS2 piecewise-linear term is appended (replacing the
    tier-0 fallback in :func:`tariff_to_opt_params`). The SOS2 path is
    intended for horizons that span (approximately) one billing month;
    for shorter horizons the fixed tier-0 rate inside ``opt_params`` is
    a reasonable approximation.
    """
    energy_term = add_tariff_energy_term(opt_params)
    tariff_builder = add_tariff_constraints(opt_params)
    demand_term, demand_builder = add_demand_charge_terms(opt_params)
    obj_terms: list[Callable] = [energy_term, demand_term]
    builders: list[Callable] = [tariff_builder, demand_builder]

    if tariff is not None:
        tier_components = [c for c in tariff.components if isinstance(c, TieredEnergyRate)]
        tou_present = any(isinstance(c, TimeOfUseRate) for c in tariff.components)
        if tier_components and not tou_present:
            # The standard energy_term currently prices p_import at the
            # tier-0 fallback. Replace it with the SOS2 piecewise term and
            # rebuild the standard term against export-only (sell side).
            tiers = tier_components[0].tiers
            thresholds = [t for t, _ in tiers]
            rates = [r for _, r in tiers]
            tier_term, tier_builder = add_tiered_energy_term(thresholds, rates)
            # Replace the buy-side energy term with an export-only one.
            zeroed = TariffOptParams(
                energy_buy_per_kwh=[0.0] * opt_params.horizon_steps,
                energy_sell_per_kwh=list(opt_params.energy_sell_per_kwh),
                demand_charge_per_kw=opt_params.demand_charge_per_kw,
                demand_charge_window=opt_params.demand_charge_window,
                demand_ratchet_floor_kw=opt_params.demand_ratchet_floor_kw,
                fixed_charges=0.0,  # carried by demand_term already
                min_bill=opt_params.min_bill,
                interval_hours=opt_params.interval_hours,
                horizon_steps=opt_params.horizon_steps,
            )
            obj_terms[0] = add_tariff_energy_term(zeroed)
            obj_terms.append(tier_term)
            builders.append(tier_builder)

    return obj_terms, builders


# ---------------------------------------------------------------------------
# NEM 3.0 avoided-cost reference vector
# ---------------------------------------------------------------------------


def load_nem3_avoided_cost_2024() -> list[float]:
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
        0.045,
        0.040,
        0.038,
        0.038,
        0.040,
        0.045,  # 0-5
        0.052,
        0.060,
        0.058,
        0.052,
        0.045,
        0.040,  # 6-11
        0.038,
        0.038,
        0.040,
        0.045,
        0.060,
        0.110,  # 12-17  (ramp begins)
        0.180,
        0.220,
        0.180,
        0.110,
        0.070,
        0.055,  # 18-23  (peak 7-8 PM)
    ]
