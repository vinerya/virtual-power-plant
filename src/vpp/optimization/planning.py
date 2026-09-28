"""
Fleet-level planning services behind the optimization REST API.

This module is deliberately free of FastAPI / SQLAlchemy: it takes plain
:class:`FleetAsset` descriptions (built from the database by the API layer)
and drives the existing solver stack:

* :func:`allocate_power` - single-interval power target split, solved by the
  ``power_allocation`` LP plugin with a proportional rule fallback.
* :func:`plan_schedule` - horizon schedule via :mod:`vpp.optimization.mpc`
  (single battery: :class:`MPCController` with tariff + degradation hooks;
  fleet: :class:`MultiResourceMPCController`, monolithic MILP or ADMM).
* :func:`run_closed_loop_backtest` - receding-horizon MPC replay through
  :func:`vpp.optimization.backtest.run_backtest`, compared against the
  perfect-foresight optimum and the rule baseline.
* :func:`explain_schedule` - counterfactual explainer for any persisted run
  that recorded a per-step schedule and price vector.

Sign convention for schedules: ``charge``/``discharge`` are non-negative kW;
``power = charge - discharge`` is the net battery draw (positive = charging),
matching the ``p_net`` convention of the dispatch formulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from .base import OptimizationProblem

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# Default marginal $/kWh used to nudge batteries toward balanced SOC in
# single-interval allocation: discharging a nearly-empty battery (or charging
# a nearly-full one) costs up to this much extra per kWh.
DEFAULT_SOC_BALANCE_WEIGHT = 0.02
DEFAULT_REPLACEMENT_COST_PER_KWH = 250.0
BATTERY_TYPES = frozenset({"battery"})
# Mirrors MultiResourceMPCConfig.admm_threshold: fleets larger than this are
# solved with ADMM instead of the monolithic MILP.
ADMM_FLEET_THRESHOLD = 10
RENEWABLE_TYPES = frozenset({"solar", "wind_turbine", "wind"})


# ---------------------------------------------------------------------------
# Asset description
# ---------------------------------------------------------------------------


@dataclass
class FleetAsset:
    """Optimizer-facing view of one persisted resource."""

    id: str
    name: str
    resource_type: str
    rated_power_kw: float
    online: bool = True
    # Battery fields
    capacity_kwh: float | None = None  # nominal (nameplate) energy
    capacity_source: str = "unknown"
    soc: float | None = None  # fraction 0..1
    soc_source: str = "unknown"
    soh: float = 1.0
    chemistry: str | None = None
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    soc_min: float = 0.05
    soc_max: float = 0.95
    # Non-battery availability (kW) and where it came from.
    available_kw: float | None = None
    availability_basis: str = "nameplate"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_battery(self) -> bool:
        return self.resource_type in BATTERY_TYPES

    @property
    def effective_capacity_kwh(self) -> float:
        """Usable pack capacity after capacity fade (nominal * SOH)."""
        cap = float(self.capacity_kwh or 0.0)
        return max(1e-6, cap * max(0.0, min(1.0, self.soh)))

    def soc_clamped(self) -> float:
        soc = 0.5 if self.soc is None else float(self.soc)
        return max(self.soc_min, min(self.soc_max, soc))

    def battery_params(self) -> dict[str, Any]:
        """Parameters for the M1 single-battery dispatch formulation."""
        return {
            "battery_capacity_kwh": self.effective_capacity_kwh,
            "max_charge_kw": float(self.rated_power_kw),
            "max_discharge_kw": float(self.rated_power_kw),
            "soc_init": self.soc_clamped(),
            "soc_min": self.soc_min,
            "soc_max": self.soc_max,
            "eta_charge": self.eta_charge,
            "eta_discharge": self.eta_discharge,
        }

    def summary(self) -> dict[str, Any]:
        """Compact, JSON-safe description persisted with each run."""
        out: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "resource_type": self.resource_type,
            "rated_power_kw": self.rated_power_kw,
        }
        if self.is_battery:
            out.update(
                {
                    "capacity_kwh": self.effective_capacity_kwh,
                    "nominal_capacity_kwh": self.capacity_kwh,
                    "capacity_source": self.capacity_source,
                    "max_charge_kw": float(self.rated_power_kw),
                    "max_discharge_kw": float(self.rated_power_kw),
                    "soc_init": self.soc_clamped(),
                    "soc_source": self.soc_source,
                    "soc_min": self.soc_min,
                    "soc_max": self.soc_max,
                    "soh": self.soh,
                    "chemistry": self.chemistry,
                    "eta_charge": self.eta_charge,
                    "eta_discharge": self.eta_discharge,
                }
            )
        else:
            out.update(
                {
                    "available_kw": self.available_kw,
                    "availability_basis": self.availability_basis,
                }
            )
        return out


# ---------------------------------------------------------------------------
# Degradation-aware wear cost
# ---------------------------------------------------------------------------


def soh_adjusted_wear_cost(
    asset: FleetAsset,
    replacement_cost_per_kwh: float = DEFAULT_REPLACEMENT_COST_PER_KWH,
    eol_capacity_fraction: float = 0.8,
):
    """Return a :class:`~vpp.degradation.optimization.WearCost` scaled by SOH.

    :meth:`WearCost.from_preset` amortises the pack replacement cost over the
    *full* rated cycle life of a new pack. A pack at state-of-health ``s`` has
    (assuming linear fade to the end-of-life threshold ``eol``) only
    ``(s - eol) / (1 - eol)`` of its cycle life left, and moves only ``s`` of
    its nameplate energy per cycle, so every kWh of throughput consumes a
    proportionally larger share of the remaining asset value::

        cost_per_kwh(s) = cost_per_kwh(new) / (remaining_life_fraction * s)

    The remaining-life fraction is floored at 5 % so packs at or past EOL get
    a large-but-finite wear cost instead of infinity.
    """
    from vpp.degradation import LFP_PRESET, NMC_PRESET
    from vpp.degradation.optimization import WearCost

    preset = NMC_PRESET if (asset.chemistry or "").lower() == "nmc" else LFP_PRESET
    nominal = float(asset.capacity_kwh or 0.0) or 1.0
    base = WearCost.from_preset(
        preset,
        capacity_kwh=nominal,
        replacement_cost_dollars=float(replacement_cost_per_kwh) * nominal,
        eol_capacity_fraction=eol_capacity_fraction,
    )
    soh = max(1e-3, min(1.0, float(asset.soh)))
    remaining = (soh - eol_capacity_fraction) / (1.0 - eol_capacity_fraction)
    remaining = max(0.05, min(1.0, remaining))
    per_kwh_scale = 1.0 / (remaining * soh)
    per_cycle_scale = 1.0 / remaining
    return WearCost(
        throughput_cost_per_kwh=base.throughput_cost_per_kwh * per_kwh_scale,
        cycle_cost_curve={d: c * per_cycle_scale for d, c in base.cycle_cost_curve.items()},
        capacity_kwh=nominal,
        replacement_cost_dollars=base.replacement_cost_dollars,
    )


# ---------------------------------------------------------------------------
# Single-interval allocation
# ---------------------------------------------------------------------------


def _constraint_for(asset: FleetAsset, constraints: dict[str, Any]) -> dict[str, Any]:
    c = constraints.get(asset.id)
    if c is None:
        c = constraints.get(asset.name)
    return dict(c) if isinstance(c, dict) else {}


def build_allocation_problem(
    assets: Sequence[FleetAsset],
    target_kw: float,
    *,
    dt_hours: float = 0.25,
    constraints: dict[str, Any] | None = None,
    wear_cost: bool = True,
    replacement_cost_per_kwh: float = DEFAULT_REPLACEMENT_COST_PER_KWH,
    soc_balance_weight: float = DEFAULT_SOC_BALANCE_WEIGHT,
) -> tuple[OptimizationProblem, dict[str, dict[str, Any]]]:
    """Translate assets into a ``power_allocation`` problem.

    Returns ``(problem, bounds_by_id)``; ``bounds_by_id`` carries the derived
    ``lo_kw``/``hi_kw``/cost per resource for reporting.

    Per-resource ``constraints`` (keyed by resource id or name) accept:
    ``exclude`` (bool), ``max_kw`` (cap on export), ``min_kw`` (floor on
    output; only values <= 0 are honoured, i.e. it limits absorption) and
    ``cost_per_kwh`` (overrides the derived marginal cost).
    """
    constraints = constraints or {}
    rows: list[dict[str, Any]] = []
    info: dict[str, dict[str, Any]] = {}
    for a in assets:
        c = _constraint_for(a, constraints)
        if c.get("exclude"):
            continue
        rated = max(0.0, float(a.rated_power_kw))
        if a.is_battery:
            cap = a.effective_capacity_kwh
            soc = a.soc_clamped()
            dis_energy_kw = max(0.0, soc - a.soc_min) * cap * a.eta_discharge / dt_hours
            chg_energy_kw = max(0.0, a.soc_max - soc) * cap / (a.eta_charge * dt_hours)
            hi = min(rated, dis_energy_kw)
            lo = -min(rated, chg_energy_kw)
            wear = (
                soh_adjusted_wear_cost(a, replacement_cost_per_kwh).throughput_cost_per_kwh
                if wear_cost
                else 0.0
            )
            cost_up = wear + soc_balance_weight * (1.0 - soc)
            cost_down = wear + soc_balance_weight * soc
        elif a.resource_type in RENEWABLE_TYPES:
            avail = rated if a.available_kw is None else max(0.0, min(rated, a.available_kw))
            hi, lo = avail, 0.0
            cost_up = cost_down = 0.0
        else:
            hi, lo = rated, 0.0
            cost_up = cost_down = 0.0
        if "max_kw" in c and c["max_kw"] is not None:
            hi = max(0.0, min(hi, float(c["max_kw"])))
        if "min_kw" in c and c["min_kw"] is not None:
            lo = min(0.0, max(lo, float(c["min_kw"])))
        if "cost_per_kwh" in c and c["cost_per_kwh"] is not None:
            cost_up = cost_down = float(c["cost_per_kwh"])
        row = {"id": a.id, "lo_kw": lo, "hi_kw": hi, "cost_up": cost_up, "cost_down": cost_down}
        rows.append(row)
        info[a.id] = row
    problem = OptimizationProblem(
        variables={},
        objectives=[{"type": "minimize_cost"}],
        constraints=[],
        parameters={"resources": rows, "target_kw": float(target_kw), "dt_hours": dt_hours},
        time_horizon=1,
        time_step=dt_hours,
        metadata={"type": "power_allocation"},
    )
    return problem, info


def allocate_power(
    assets: Sequence[FleetAsset],
    target_kw: float,
    *,
    dt_hours: float = 0.25,
    constraints: dict[str, Any] | None = None,
    timeout_ms: int = 5000,
    force_fallback: bool = False,
    wear_cost: bool = True,
    replacement_cost_per_kwh: float = DEFAULT_REPLACEMENT_COST_PER_KWH,
) -> dict[str, Any]:
    """Split ``target_kw`` across ``assets``; see :func:`build_allocation_problem`."""
    from . import solve_with_fallback

    problem, info = build_allocation_problem(
        assets,
        target_kw,
        dt_hours=dt_hours,
        constraints=constraints,
        wear_cost=wear_cost,
        replacement_cost_per_kwh=replacement_cost_per_kwh,
    )
    if not info:
        return {
            "status": "failed",
            "method": "none",
            "fallback_used": False,
            "allocations": {},
            "bounds": {},
            "delivered_kw": 0.0,
            "shortfall_kw": float(target_kw),
            "objective_value": 0.0,
            "message": "No online, dispatchable resources available",
        }
    result = solve_with_fallback(problem, timeout_ms=timeout_ms, force_fallback=force_fallback)
    sol = result.solution or {}
    status = result.status.value if hasattr(result.status, "value") else str(result.status)
    return {
        "status": status,
        "method": sol.get("method", "none"),
        "fallback_used": bool(result.fallback_used),
        "allocations": dict(sol.get("allocations") or {}),
        "bounds": info,
        "delivered_kw": float(sol.get("delivered_kw", 0.0)),
        "shortfall_kw": float(sol.get("shortfall_kw", target_kw)),
        "objective_value": (
            float(result.objective_value) if math.isfinite(result.objective_value) else 0.0
        ),
        "message": str((result.metadata or {}).get("error", "")),
    }


# ---------------------------------------------------------------------------
# Horizon schedule
# ---------------------------------------------------------------------------


def _fleet_wear_term(costs: dict[str, float]) -> Callable:
    """Throughput wear cost for the fleet formulation (per-resource $/kWh)."""

    def _term(model, _params):
        return sum(
            costs.get(r, 0.0) * (model.p_charge[r, t] + model.p_discharge[r, t]) * model.dt
            for r in model.R
            for t in model.T
        )

    _term.__name__ = "fleet_wear_cost_throughput"
    return _term


def _energy_cost(
    prices: Sequence[float], charge: Sequence[float], discharge: Sequence[float], dt: float
) -> float:
    return float(sum(p * (c - d) * dt for p, c, d in zip(prices, charge, discharge, strict=False)))


def terminal_energy_value(prices: Sequence[float]) -> float:
    """Value (currency/kWh) credited to energy left in storage at the end of
    a window: the window's mean price. Used to compare policies that finish
    at different states of charge on an equal footing."""
    prices = list(prices)
    return float(sum(prices) / len(prices)) if prices else 0.0


def _rules_plan(params: dict[str, Any]) -> dict[str, list[float]]:
    from .solvers.pyomo_plugin import SimpleBatteryDispatchRules

    problem = OptimizationProblem(
        variables={},
        objectives=[],
        constraints=[],
        parameters=params,
        metadata={"type": "battery_dispatch"},
    )
    res = SimpleBatteryDispatchRules().solve(problem)
    sol = res.solution or {}
    T = len(params["prices"])
    cap = float(params["battery_capacity_kwh"])
    return {
        "p_charge": list(sol.get("p_charge") or [0.0] * T),
        "p_discharge": list(sol.get("p_discharge") or [0.0] * T),
        "soc": [s / cap for s in (sol.get("soc") or [params["soc_init"] * cap] * T)],
    }


def _round_list(xs: Sequence[float], nd: int = 4) -> list[float]:
    return [round(float(x), nd) for x in xs]


def plan_schedule(
    batteries: Sequence[FleetAsset],
    *,
    prices: Sequence[float],
    interval_minutes: int,
    load_kw: Sequence[float] | None = None,
    solar_kw: Sequence[float] | None = None,
    tariff_opt: Any = None,
    tariff: Any = None,
    degradation_aware: bool = False,
    replacement_cost_per_kwh: float = DEFAULT_REPLACEMENT_COST_PER_KWH,
    feeder_max_import_kw: float | None = None,
    feeder_max_export_kw: float | None = None,
    timeout_ms: int = 10_000,
    force_fallback: bool = False,
    terminal_soc_policy: str = "value",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Optimise a charge/discharge schedule for ``batteries`` over ``prices``.

    ``terminal_soc_policy`` decides what happens at the end of the horizon:
    ``"hold"`` requires every battery to finish at (or above) its starting
    SOC; ``"value"`` (default) leaves the terminal SOC free but credits
    stored energy at :func:`terminal_energy_value` — the same valuation the
    explainer and backtest use, so the plan is optimal for the metric it is
    later scored on. Large fleets routed to ADMM always use ``"hold"`` since
    the ADMM path takes no extra objective terms.

    ``prices`` are per-step energy prices (currency/kWh). When ``tariff_opt``
    (a :class:`~vpp.tariffs.optimization.TariffOptParams`) is supplied, the
    single-battery path replaces the flat energy cost with the full tariff
    hooks (TOU import/export prices, demand charge, fixed charges); ``prices``
    should then be the tariff's buy-price vector, used for fallbacks and
    reporting. The fleet formulation prices net exchange at ``prices`` only.

    Returns a JSON-safe dict with aggregate ``charge``/``discharge``/``power``
    vectors, a per-resource breakdown, the solve method and notes about any
    modelling simplification applied.
    """
    from .formulations.fleet_dispatch import FleetBattery, FleetCoupling
    from .mpc import (
        MPCConfig,
        MPCController,
        MPCStep,
        MultiResourceMPCConfig,
        MultiResourceMPCController,
        MultiResourceMPCStep,
    )

    if not batteries:
        raise ValueError("at least one battery is required")
    prices = [float(p) for p in prices]
    H = len(prices)
    if H == 0:
        raise ValueError("prices must be non-empty")
    dt = interval_minutes / 60.0
    now = now or datetime.now(timezone.utc)
    notes: list[str] = []
    if terminal_soc_policy not in ("value", "hold"):
        raise ValueError(f"unknown terminal_soc_policy {terminal_soc_policy!r}")
    if len(batteries) > ADMM_FLEET_THRESHOLD and terminal_soc_policy == "value":
        terminal_soc_policy = "hold"
        notes.append("large fleets are solved with ADMM, which uses terminal_soc_policy='hold'")
    value = terminal_energy_value(prices)
    wear_costs: dict[str, float] = {}
    if degradation_aware:
        for b in batteries:
            wear_costs[b.id] = soh_adjusted_wear_cost(
                b, replacement_cost_per_kwh
            ).throughput_cost_per_kwh

    per_resource: dict[str, dict[str, list[float]]] = {}
    method = "milp_highs"
    fallback_used = False
    fallback_reason: str | None = None
    objective: float | None = None
    solver_meta: dict[str, Any] = {}

    if len(batteries) == 1:
        b = batteries[0]
        bp = b.battery_params()
        obj_terms: list[Callable] = []
        builders: list[Callable] = []
        if load_kw is not None:
            bp["load"] = [float(x) for x in load_kw]
        if solar_kw is not None:
            bp["solar"] = [float(x) for x in solar_kw]
        if tariff_opt is not None:
            from vpp.tariffs.optimization import build_tariff_hooks

            t_terms, t_builders = build_tariff_hooks(tariff_opt, tariff)
            obj_terms += t_terms
            builders += t_builders
            bp["disable_base_energy_cost"] = True
        if degradation_aware:
            from vpp.degradation.optimization import add_dod_constraints, add_wear_cost_term

            wear = soh_adjusted_wear_cost(b, replacement_cost_per_kwh)
            # Rainflow-consistent cycle-depth cost (matches the SOH persisted
            # by the telemetry DegradationUpdater).
            builders.append(add_dod_constraints(wear))
            obj_terms.append(add_wear_cost_term(wear, mode="dod_pwl"))
        if terminal_soc_policy == "value":
            bp["terminal_soc"] = bp["soc_min"]

            def _terminal_credit(m, _params):
                return -value * (m.soc[H - 1] - m.soc_0)

            obj_terms.append(_terminal_credit)
        if force_fallback:
            params = dict(bp, prices=prices, dt_hours=dt)
            params.setdefault("terminal_soc", bp["soc_init"])
            plan = _rules_plan(params)
            method, fallback_used, fallback_reason = "rule_based_threshold", True, "forced"
        else:
            ctl = MPCController(
                MPCConfig(
                    horizon_steps=H,
                    interval_minutes=interval_minutes,
                    warm_start=False,
                    solver_timeout_ms=timeout_ms,
                ),
                bp,
            )
            decision = ctl.step(
                MPCStep(
                    timestamp=now,
                    soc_init=bp["soc_init"],
                    forecast={"prices": prices},
                    additional_objective_terms=obj_terms,
                    additional_constraint_builders=builders,
                )
            )
            fp = decision.full_horizon_plan
            cap = bp["battery_capacity_kwh"]
            if decision.fallback_used:
                method, fallback_used = "rule_based_threshold", True
                fallback_reason = str(fp.get("fallback_reason", ""))
                plan = {
                    "p_charge": list(fp.get("p_charge") or [0.0] * H),
                    "p_discharge": list(fp.get("p_discharge") or [0.0] * H),
                    "soc": [s / cap for s in (fp.get("soc") or [bp["soc_init"] * cap] * H)],
                }
            else:
                plan = {
                    "p_charge": list(fp["p_charge"]),
                    "p_discharge": list(fp["p_discharge"]),
                    "soc": [s / cap for s in fp["soc"]],
                }
                objective = float(decision.expected_cost_remaining)
                solver_meta["solver_iterations"] = decision.solver_iterations
        per_resource[b.id] = plan
    else:
        if tariff_opt is not None:
            notes.append(
                "fleet schedules price net exchange at the tariff buy price; "
                "export compensation and demand charges are only modelled for "
                "single-battery schedules"
            )
        fleet = [
            FleetBattery(
                id=b.id,
                capacity_kwh=b.effective_capacity_kwh,
                max_charge_kw=float(b.rated_power_kw),
                max_discharge_kw=float(b.rated_power_kw),
                soc_init=b.soc_clamped(),
                soc_min=b.soc_min,
                soc_max=b.soc_max,
                eta_charge=b.eta_charge,
                eta_discharge=b.eta_discharge,
                terminal_soc=b.soc_min if terminal_soc_policy == "value" else None,
            )
            for b in batteries
        ]
        fleet_terms: list[Callable] = []
        if wear_costs:
            fleet_terms.append(_fleet_wear_term(wear_costs))
        if terminal_soc_policy == "value":

            def _fleet_terminal_credit(m, _params):
                return -value * sum(m.soc[r, H - 1] - m.soc_0[r] for r in m.R)

            fleet_terms.append(_fleet_terminal_credit)
        decision = None
        if not force_fallback:
            ctl = MultiResourceMPCController(
                MultiResourceMPCConfig(
                    horizon_steps=H,
                    interval_minutes=interval_minutes,
                    solver_timeout_ms=timeout_ms,
                ),
                fleet,
                FleetCoupling(
                    feeder_max_import_kw=feeder_max_import_kw,
                    feeder_max_export_kw=feeder_max_export_kw,
                ),
            )
            forecast: dict[str, list[float]] = {"prices": prices}
            if load_kw is not None:
                forecast["load_kw"] = [float(x) for x in load_kw]
            if solar_kw is not None:
                forecast["solar_kw"] = [float(x) for x in solar_kw]
            decision = ctl.step(
                MultiResourceMPCStep(
                    timestamp=now,
                    forecast=forecast,
                    additional_objective_terms=fleet_terms,
                )
            )
            if decision.method == "admm" and wear_costs:
                notes.append("ADMM path does not include the wear-cost term")
        if decision is not None and not decision.fallback_used:
            method = "milp_highs" if decision.method == "monolithic" else "admm_highs"
            objective = float(decision.expected_cost_remaining)
            for rid, p in decision.metadata.get("plan", {}).items():
                cap = next(b.effective_capacity_kwh for b in batteries if b.id == rid)
                per_resource[rid] = {
                    "p_charge": list(p["p_charge"]),
                    "p_discharge": list(p["p_discharge"]),
                    "soc": [s / cap for s in p.get("soc", [])],
                }
            for k in ("iterations", "converged", "primal_residual", "dual_residual"):
                if k in decision.metadata:
                    solver_meta[k] = decision.metadata[k]
        else:
            method, fallback_used = "rule_based_threshold", True
            fallback_reason = (
                "forced" if decision is None else str(decision.metadata.get("fallback_reason", ""))
            )
            if feeder_max_import_kw is not None or feeder_max_export_kw is not None:
                notes.append("rule fallback does not enforce feeder limits")
            for b in batteries:
                params = dict(
                    b.battery_params(),
                    prices=prices,
                    dt_hours=dt,
                    terminal_soc=b.soc_min if terminal_soc_policy == "value" else b.soc_clamped(),
                )
                per_resource[b.id] = _rules_plan(params)

    charge = [sum(p["p_charge"][t] for p in per_resource.values()) for t in range(H)]
    discharge = [sum(p["p_discharge"][t] for p in per_resource.values()) for t in range(H)]
    energy_cost = _energy_cost(prices, charge, discharge, dt)
    wear_total = sum(
        wear_costs.get(rid, 0.0)
        * sum(c + d for c, d in zip(p["p_charge"], p["p_discharge"], strict=False))
        * dt
        for rid, p in per_resource.items()
    )
    return {
        "method": method,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "objective": objective if objective is not None else energy_cost + wear_total,
        "energy_cost": energy_cost,
        "wear_cost": wear_total,
        "charge": _round_list(charge),
        "discharge": _round_list(discharge),
        "power": _round_list([c - d for c, d in zip(charge, discharge, strict=False)]),
        "per_resource": {
            rid: {
                "charge": _round_list(p["p_charge"]),
                "discharge": _round_list(p["p_discharge"]),
                "soc": _round_list(p["soc"]),
            }
            for rid, p in per_resource.items()
        },
        "wear_cost_per_kwh": wear_costs,
        "terminal_soc_policy": terminal_soc_policy,
        "terminal_energy_value_per_kwh": value,
        "notes": notes,
        "solver_meta": solver_meta,
    }


# ---------------------------------------------------------------------------
# Closed-loop backtest
# ---------------------------------------------------------------------------


def make_forecast_fn(
    prices: Sequence[float],
    load: Sequence[float],
    solar: Sequence[float],
    *,
    start: datetime,
    interval_minutes: int,
    mode: str = "perfect",
    noise_sigma: float = 0.1,
    seed: int | None = None,
) -> Callable[[datetime, int], dict[str, list[float]]]:
    """Build the ``forecast_fn`` consumed by :func:`run_backtest`.

    * ``perfect``: the true future prices.
    * ``persistence``: day-ahead persistence, ``forecast[t] = actual[t - 1 day]``
      (falls back to the most recent observed price before a full day of
      history exists). Uses no future information.
    * ``noisy``: true prices times a seeded log-normal error (sigma
      ``noise_sigma``), redrawn every tick.
    """
    import numpy as np

    prices = [float(p) for p in prices]
    N = len(prices)
    steps_per_day = max(1, round(24 * 60 / interval_minutes))
    rng = np.random.default_rng(seed)

    def _idx(now: datetime) -> int:
        return round((now - start).total_seconds() / 60.0 / interval_minutes)

    def _window(series: Sequence[float], k: int, H: int) -> list[float]:
        if not series:
            return []
        out = list(series[k : k + H])
        while len(out) < H:
            out.append(out[-1] if out else float(series[-1]))
        return out

    def _fn(now: datetime, H: int) -> dict[str, list[float]]:
        k = _idx(now)
        if mode == "perfect":
            fc = _window(prices, k, H)
        elif mode == "persistence":
            # Most recent same-time-of-day observation at or before step k
            # (the current interval's price is assumed known, as in a
            # real-time market); before a day of history exists, hold the
            # latest observed price.
            last = prices[min(k, N - 1)]
            fc = []
            for j in range(H):
                src = k + j - math.ceil(j / steps_per_day) * steps_per_day
                fc.append(prices[src] if 0 <= src < N else last)
        elif mode == "noisy":
            base = _window(prices, k, H)
            mult = rng.lognormal(mean=0.0, sigma=noise_sigma, size=H)
            fc = [float(p * m) for p, m in zip(base, mult, strict=False)]
        else:
            raise ValueError(f"unknown forecast mode {mode!r}")
        return {
            "prices": fc,
            "load_kw": _window(load, k, H),
            "solar_kw": _window(solar, k, H),
        }

    return _fn


def _terminal_value_controller_cls():
    from .mpc import MPCController

    class TerminalValueMPCController(MPCController):
        """MPCController whose plans credit end-of-horizon stored energy at
        the mean forecast price instead of pinning the terminal SOC."""

        def step(self, mpc_step):
            H = self.config.horizon_steps
            fc = list(mpc_step.forecast.get("prices") or [])
            window = (fc + [fc[-1]] * H)[:H] if fc else []
            v = terminal_energy_value(window)
            last = H - 1

            def _credit(m, _params):
                return -v * (m.soc[last] - m.soc_0)

            mpc_step.additional_objective_terms = [
                *list(mpc_step.additional_objective_terms),
                _credit,
            ]
            return super().step(mpc_step)

    return TerminalValueMPCController


def _TerminalValueMPCController(config, battery_params):
    return _terminal_value_controller_cls()(config, battery_params)


def run_closed_loop_backtest(
    battery: FleetAsset,
    *,
    prices: Sequence[float],
    interval_minutes: int,
    horizon_steps: int,
    load_kw: Sequence[float] | None = None,
    solar_kw: Sequence[float] | None = None,
    forecast_mode: str = "perfect",
    noise_sigma: float = 0.1,
    seed: int | None = None,
    solver_timeout_ms: int = 2_000,
    start: datetime | None = None,
    compare_offline: bool = True,
    terminal_soc_policy: str = "value",
) -> dict[str, Any]:
    """Replay a receding-horizon MPC over ``prices`` and score it.

    The realized cost is compared with (a) doing nothing (cost 0 — costs
    count only the battery's net exchange), (b) the rule-based threshold
    dispatcher run on the true prices, and (c) when ``compare_offline`` the
    perfect-foresight offline MILP optimum over the whole window, giving the
    MPC's regret.

    Policies end the window with different amounts of stored energy (the rule
    dispatcher tends to drain the pack), so raw costs are not comparable.
    ``*_adjusted`` costs credit the change in stored energy at
    :func:`terminal_energy_value` (the window's mean price). The offline
    benchmark optimises exactly that adjusted cost, so it is a true lower
    bound and ``regret >= 0`` up to solver tolerance.

    ``terminal_soc_policy`` configures each MPC tick: ``"hold"`` makes every
    plan return to the current SOC at the horizon end (the MPCController
    default); ``"value"`` frees it and credits stored energy at the mean
    forecast price of the horizon.
    """
    from .backtest import BacktestConfig, run_backtest
    from .mpc import MPCConfig, MPCController, MPCStep

    prices = [float(p) for p in prices]
    N = len(prices)
    load = [float(x) for x in (load_kw or [0.0] * N)]
    solar = [float(x) for x in (solar_kw or [0.0] * N)]
    start = start or datetime(2024, 1, 1, tzinfo=timezone.utc)
    dt = interval_minutes / 60.0
    bp = battery.battery_params()

    if terminal_soc_policy not in ("value", "hold"):
        raise ValueError(f"unknown terminal_soc_policy {terminal_soc_policy!r}")
    mpc_cfg = MPCConfig(
        horizon_steps=horizon_steps,
        interval_minutes=interval_minutes,
        warm_start=True,
        solver_timeout_ms=solver_timeout_ms,
    )
    if terminal_soc_policy == "value":
        ctl: MPCController = _TerminalValueMPCController(
            mpc_cfg, dict(bp, terminal_soc=bp["soc_min"])
        )
    else:
        ctl = MPCController(mpc_cfg, bp)
    fn = make_forecast_fn(
        prices,
        load,
        solar,
        start=start,
        interval_minutes=interval_minutes,
        mode=forecast_mode,
        noise_sigma=noise_sigma,
        seed=seed,
    )
    res = run_backtest(
        ctl,
        BacktestConfig(
            start=start,
            end=start + timedelta(minutes=interval_minutes * N),
            interval_minutes=interval_minutes,
        ),
        prices,
        load,
        solar,
        fn,
    )
    cap = bp["battery_capacity_kwh"]
    e_init = bp["soc_init"] * cap
    value = terminal_energy_value(prices)
    charge = [d["p_charge_kw"] for d in res.realized_dispatch]
    discharge = [d["p_discharge_kw"] for d in res.realized_dispatch]
    realized = float(res.realized_cost)
    e_final = res.soc_trajectory[-1] if res.soc_trajectory else e_init
    realized_adj = realized - value * (e_final - e_init)

    full_params = dict(bp, prices=prices, dt_hours=dt, terminal_soc=bp["soc_init"])
    rules = _rules_plan(full_params)
    rules_cost = _energy_cost(prices, rules["p_charge"], rules["p_discharge"], dt)
    rules_adj = rules_cost - value * (
        (rules["soc"][-1] if rules["soc"] else bp["soc_init"]) * cap - e_init
    )

    offline_adj: float | None = None
    offline_status = "skipped"
    if compare_offline:
        # Free terminal SOC, with stored energy credited at the same value
        # used to adjust the other policies: the result is the exact
        # perfect-foresight lower bound on the adjusted cost.
        ctl_off = MPCController(
            MPCConfig(
                horizon_steps=N,
                interval_minutes=interval_minutes,
                warm_start=False,
                solver_timeout_ms=max(solver_timeout_ms, 5_000),
                fallback_on_failure=False,
            ),
            dict(bp, terminal_soc=bp["soc_min"]),
        )

        def _terminal_credit(m, _params):
            return -value * (m.soc[N - 1] - m.soc_0)

        dec = ctl_off.step(
            MPCStep(
                timestamp=start,
                soc_init=bp["soc_init"],
                forecast={"prices": prices},
                additional_objective_terms=[_terminal_credit],
            )
        )
        fp = dec.full_horizon_plan
        if "p_charge" in fp and not dec.fallback_used:
            offline_adj = _energy_cost(prices, fp["p_charge"], fp["p_discharge"], dt) - value * (
                fp["soc"][-1] - e_init
            )
            offline_status = "success"
        else:
            offline_status = str(fp.get("status", "failed"))

    return {
        "ticks": len(res.realized_dispatch),
        "realized_cost": realized,
        "realized_cost_adjusted": realized_adj,
        "no_action_cost": 0.0,
        "rules_cost": rules_cost,
        "rules_cost_adjusted": rules_adj,
        "perfect_foresight_cost_adjusted": offline_adj,
        "perfect_foresight_status": offline_status,
        "regret": (realized_adj - offline_adj) if offline_adj is not None else None,
        "terminal_energy_value_per_kwh": value,
        "terminal_soc_policy": terminal_soc_policy,
        "final_soc": e_final / cap,
        "fallback_count": res.fallback_count,
        "cumulative_solve_time_ms": res.cumulative_solve_time_ms,
        "cumulative_solver_iterations": res.cumulative_solver_iterations,
        "wall_time_s": res.wall_time_s,
        "charge": _round_list(charge),
        "discharge": _round_list(discharge),
        "power": _round_list([c - d for c, d in zip(charge, discharge, strict=False)]),
        "soc": _round_list([s / cap for s in res.soc_trajectory]),
        "realized_price": _round_list(prices[: len(charge)]),
    }


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------

_BINDING_TOL = 1e-3
_MAX_BINDING = 25


def _numeric_list(v: Any) -> list[float] | None:
    if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
        return [float(x) for x in v]
    return None


def explain_schedule(inputs: dict[str, Any], solution: dict[str, Any]) -> dict[str, Any] | None:
    """Counterfactual explanation for a persisted schedule run.

    Needs ``inputs['prices']`` (per step), ``inputs['interval_minutes']`` and
    ``solution['charge']``/``solution['discharge']``. Battery descriptions in
    ``inputs['batteries']`` enable the ``price_naive`` counterfactual (the
    rule-based threshold dispatcher on the same prices) and the binding-
    constraint scan. Returns ``None`` when the run carries no schedule.
    """
    prices = _numeric_list(inputs.get("prices"))
    charge = _numeric_list(solution.get("charge"))
    discharge = _numeric_list(solution.get("discharge"))
    if prices is None or charge is None or discharge is None:
        return None
    T = min(len(prices), len(charge), len(discharge))
    prices, charge, discharge = prices[:T], charge[:T], discharge[:T]
    dt = float(inputs.get("interval_minutes", 60)) / 60.0
    value = terminal_energy_value(prices)
    batteries = [b for b in (inputs.get("batteries") or []) if isinstance(b, dict)]

    def _run(
        name: str, chg: Sequence[float], dis: Sequence[float], delta_kwh: float = 0.0
    ) -> dict[str, Any]:
        energy = _energy_cost(prices, chg, dis, dt)
        return {
            "name": name,
            # Energy cost with the change in stored energy credited at the
            # window's mean price, so runs ending at different SOCs compare
            # fairly. ``energy_cost`` is the raw sum of the per-step costs.
            "total_cost": round(energy - value * delta_kwh, 6),
            "energy_cost": round(energy, 6),
            "stored_energy_delta_kwh": round(delta_kwh, 4),
            "per_step": [
                {
                    "step": t,
                    "charge": round(chg[t], 4),
                    "discharge": round(dis[t], 4),
                    "power": round(chg[t] - dis[t], 4),
                    "price": prices[t],
                    "cost": round(prices[t] * (chg[t] - dis[t]) * dt, 6),
                }
                for t in range(T)
            ],
        }

    actual = _run(
        "actual",
        charge,
        discharge,
        _stored_energy_delta(batteries, solution, charge, discharge, dt),
    )
    counterfactuals = [_run("no_action", [0.0] * T, [0.0] * T)]

    binding: list[dict[str, Any]] = []
    if batteries:
        naive_c = [0.0] * T
        naive_d = [0.0] * T
        naive_delta = 0.0
        for b in batteries:
            try:
                params = {
                    "battery_capacity_kwh": float(b["capacity_kwh"]),
                    "max_charge_kw": float(b["max_charge_kw"]),
                    "max_discharge_kw": float(b["max_discharge_kw"]),
                    "soc_init": float(b["soc_init"]),
                    "soc_min": float(b["soc_min"]),
                    "soc_max": float(b["soc_max"]),
                    "eta_charge": float(b["eta_charge"]),
                    "eta_discharge": float(b["eta_discharge"]),
                    "prices": prices,
                    "dt_hours": dt,
                    "terminal_soc": float(b["soc_init"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
            plan = _rules_plan(params)
            naive_c = [a + x for a, x in zip(naive_c, plan["p_charge"], strict=False)]
            naive_d = [a + x for a, x in zip(naive_d, plan["p_discharge"], strict=False)]
            if plan["soc"]:
                naive_delta += (plan["soc"][-1] - params["soc_init"]) * params[
                    "battery_capacity_kwh"
                ]
        counterfactuals.append(_run("price_naive", naive_c, naive_d, naive_delta))
        binding = _binding_constraints(batteries, solution, T)

    savings = counterfactuals[0]["total_cost"] - actual["total_cost"]
    return {
        "actual": actual,
        "counterfactuals": counterfactuals,
        "binding_constraints": binding,
        "rationale": _rationale(prices, charge, discharge, dt, savings, counterfactuals, value),
    }


def _stored_energy_delta(
    batteries: list[dict[str, Any]],
    solution: dict[str, Any],
    charge: Sequence[float],
    discharge: Sequence[float],
    dt: float,
) -> float:
    """Change in stored energy (kWh) over a recorded schedule.

    Uses the per-resource SOC trajectories when recorded; otherwise
    integrates the aggregate charge/discharge with the first battery's
    efficiencies (0.95 when unknown).
    """
    per_resource = solution.get("per_resource") or {}
    total = 0.0
    found = False
    for b in batteries:
        plan = per_resource.get(str(b.get("id"))) or {}
        soc = _numeric_list(plan.get("soc"))
        if soc is None and len(batteries) == 1:
            soc = _numeric_list(solution.get("soc"))
        try:
            if soc is not None:
                total += (soc[-1] - float(b["soc_init"])) * float(b["capacity_kwh"])
                found = True
        except (KeyError, TypeError, ValueError):
            continue
    if found:
        return total
    eta_c = float(batteries[0].get("eta_charge", 0.95)) if batteries else 0.95
    eta_d = float(batteries[0].get("eta_discharge", 0.95)) if batteries else 0.95
    return sum(eta_c * c - d / eta_d for c, d in zip(charge, discharge, strict=False)) * dt


def _binding_constraints(
    batteries: list[dict[str, Any]], solution: dict[str, Any], T: int
) -> list[dict[str, Any]]:
    per_resource = solution.get("per_resource") or {}
    if not per_resource and len(batteries) == 1:
        per_resource = {
            str(batteries[0].get("id", "battery")): {
                "charge": solution.get("charge"),
                "discharge": solution.get("discharge"),
                "soc": solution.get("soc"),
            }
        }
    out: list[dict[str, Any]] = []
    multi = len(batteries) > 1
    for b in batteries:
        rid = str(b.get("id"))
        plan = per_resource.get(rid) or {}
        label = f" ({b.get('name', rid)})" if multi else ""
        soc = _numeric_list(plan.get("soc")) or []
        chg = _numeric_list(plan.get("charge")) or []
        dis = _numeric_list(plan.get("discharge")) or []
        smin, smax = float(b.get("soc_min", 0.0)), float(b.get("soc_max", 1.0))
        pmax_c = float(b.get("max_charge_kw", 0.0))
        pmax_d = float(b.get("max_discharge_kw", 0.0))
        for t in range(min(T, len(soc))):
            if smax - soc[t] <= _BINDING_TOL:
                out.append(
                    {
                        "name": "soc_upper",
                        "step": t,
                        "slack": max(0.0, smax - soc[t]),
                        "description": f"Battery{label} reached its upper SOC limit ({smax:.0%}) at t={t}",
                    }
                )
            elif soc[t] - smin <= _BINDING_TOL:
                out.append(
                    {
                        "name": "soc_lower",
                        "step": t,
                        "slack": max(0.0, soc[t] - smin),
                        "description": f"Battery{label} reached its lower SOC limit ({smin:.0%}) at t={t}",
                    }
                )
        for t in range(min(T, len(chg))):
            if pmax_c > 0 and pmax_c - chg[t] <= _BINDING_TOL * max(1.0, pmax_c):
                out.append(
                    {
                        "name": "charge_power_max",
                        "step": t,
                        "slack": max(0.0, pmax_c - chg[t]),
                        "description": f"Battery{label} charging at its power limit ({pmax_c:g} kW) at t={t}",
                    }
                )
        for t in range(min(T, len(dis))):
            if pmax_d > 0 and pmax_d - dis[t] <= _BINDING_TOL * max(1.0, pmax_d):
                out.append(
                    {
                        "name": "discharge_power_max",
                        "step": t,
                        "slack": max(0.0, pmax_d - dis[t]),
                        "description": f"Battery{label} discharging at its power limit ({pmax_d:g} kW) at t={t}",
                    }
                )
    out.sort(key=lambda c: (c["step"], c["name"]))
    return out[:_MAX_BINDING]


def _rationale(
    prices: list[float],
    charge: list[float],
    discharge: list[float],
    dt: float,
    savings: float,
    counterfactuals: list[dict[str, Any]],
    value: float,
) -> str:
    e_in = sum(charge) * dt
    e_out = sum(discharge) * dt
    if e_in < 1e-6 and e_out < 1e-6:
        return "The optimizer kept the batteries idle: no price spread over the horizon was large enough to cover losses and wear."
    parts = []
    if e_in > 1e-6:
        avg_in = sum(p * c for p, c in zip(prices, charge, strict=False)) * dt / e_in
        parts.append(f"charged {e_in:.1f} kWh at an average price of {avg_in:.4g}/kWh")
    if e_out > 1e-6:
        avg_out = sum(p * d for p, d in zip(prices, discharge, strict=False)) * dt / e_out
        parts.append(f"discharged {e_out:.1f} kWh at an average price of {avg_out:.4g}/kWh")
    text = "The schedule " + " and ".join(parts) + "."
    if savings >= 0:
        text += f" Compared with leaving the batteries idle it saves {savings:.2f}."
    else:
        text += (
            f" Compared with leaving the batteries idle it costs {-savings:.2f} more"
            " (e.g. because of tariff or wear-cost terms not reflected in the energy price)."
        )
    naive = next((c for c in counterfactuals if c["name"] == "price_naive"), None)
    if naive is not None:
        actual_cost = counterfactuals[0]["total_cost"] - savings
        text += (
            f" The rule-based threshold dispatcher on the same prices would cost"
            f" {naive['total_cost']:.2f} vs. {actual_cost:.2f} for this schedule."
        )
    text += (
        f" Costs credit any change in stored energy at the mean price ({value:.4g}/kWh)"
        " so schedules ending at different states of charge compare fairly."
    )
    return text


__all__ = [
    "FleetAsset",
    "allocate_power",
    "build_allocation_problem",
    "explain_schedule",
    "make_forecast_fn",
    "plan_schedule",
    "run_closed_loop_backtest",
    "soh_adjusted_wear_cost",
]
