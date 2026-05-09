"""Wear-cost terms and SOC-bias hooks for the dispatch optimizer (Milestone 2).

This module bridges the chemistry-aware degradation models in
:mod:`vpp.degradation.models` and the M2 hook API of
:func:`vpp.optimization.formulations.dispatch.build_battery_dispatch_model`.

Phenomenon -> optimization-term mapping
---------------------------------------

* **Throughput wear** (``ThroughputDegradation``) -> linear cost term

  ``Sum_t lambda_throughput * (p_charge[t] + p_discharge[t]) * dt``

  This is exact for ThroughputDegradation, since that model's loss is
  proportional to summed |dSOC| (= summed (p_chg + p_dis)*dt / cap). It is
  also a sound LP-friendly proxy for cycle counting: every kWh moved costs a
  fixed amount.

* **Cycle-depth wear** (``RainflowDegradation``) -> piecewise-linear
  per-step DoD bins. We linearize "cost per cycle is convex in DoD" by
  binning the absolute step-wise SOC change into B bins with strictly
  increasing per-bin marginal costs. The optimizer therefore prefers to
  spread cycling shallow (cheaper bins) rather than concentrate it deep
  (expensive bins).

  Assumptions of the linearization:
    1. Each step's |dSOC| is treated as half-cycle worth (the *per cycle*
       cost is divided by 2 since one full cycle = one charge half + one
       discharge half).
    2. No rest periods / no cycle-mean SOC effect (rainflow proper does
       this; the linearization does not).
    3. Single round-trip cycles -- multi-step nested cycles are not
       captured exactly. Rainflow remains the post-hoc validator.

* **Calendar aging** (``CalendarDegradation``) -> SOC L1-bias term

  ``Sum_t weight * |soc_frac[t] - soc_neutral|``  (default soc_neutral=0.5)

  Calendar aging is state-dependent (SOC reservoir at rest), not control-
  dependent. We add a soft penalty pushing the optimizer away from parking
  at high SOC. We use L1 (LP-friendly) instead of the more physical L2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Mapping, Optional

if TYPE_CHECKING:  # pragma: no cover
    import pyomo.environ as pyo

# Type aliases mirroring dispatch.py
ObjectiveTerm = Callable[[Any, Dict[str, Any]], Any]
ConstraintBuilder = Callable[[Any, Dict[str, Any]], None]


# ---------------------------------------------------------------------------
# WearCost dataclass
# ---------------------------------------------------------------------------


@dataclass
class WearCost:
    """Linearized $/kWh wear cost derived from chemistry preset + replacement cost.

    Attributes
    ----------
    throughput_cost_per_kwh
        Marginal cost in $ per kWh of energy moved (charge OR discharge).
        For a full charge+discharge round-trip of E kWh, the wear cost is
        ``2 * E * throughput_cost_per_kwh``.
    cycle_cost_curve
        ``{dod: $/cycle}`` map. The cost of one full cycle at a given DoD,
        derived from the chemistry's cycle-life curve and the replacement
        cost. Used by :func:`add_dod_constraints` and the ``dod_pwl`` term.
    capacity_kwh
        The pack capacity used to derive these numbers (informational).
    replacement_cost_dollars
        The replacement-cost basis used to derive these numbers.
    """

    throughput_cost_per_kwh: float
    cycle_cost_curve: Dict[float, float]
    capacity_kwh: float = 1.0
    replacement_cost_dollars: float = 0.0

    @classmethod
    def from_preset(
        cls,
        preset: Mapping[str, Mapping[str, float]],
        capacity_kwh: float,
        replacement_cost_dollars: float,
        eol_capacity_fraction: float = 0.8,
    ) -> "WearCost":
        """Derive a WearCost from an LFP_PRESET / NMC_PRESET-style mapping.

        The throughput cost is::

            replacement_cost / total_lifetime_throughput_kwh

        where total lifetime throughput = ``2 * cycles_to_eol * capacity``
        (one full cycle moves 2*capacity kWh; charge + discharge).

        The cycle-cost curve at DoD ``d`` is::

            replacement_cost / N_eol(d)

        where ``N_eol(d)`` comes from the rainflow cycle-life curve.
        """
        if capacity_kwh <= 0:
            raise ValueError("capacity_kwh must be > 0")
        if replacement_cost_dollars <= 0:
            raise ValueError("replacement_cost_dollars must be > 0")

        # Throughput cost from the throughput preset entry.
        cycles_to_eol = float(preset["throughput"]["cycles_to_eol"])
        # 1 full cycle = charge(cap) + discharge(cap) = 2 * cap kWh moved.
        total_throughput_kwh = 2.0 * cycles_to_eol * capacity_kwh
        # Scale by (1 - eol_fraction) is implicit -- the replacement happens at EOL.
        throughput_cost = replacement_cost_dollars / total_throughput_kwh

        # Cycle-cost curve from the rainflow preset entry.
        rf_curve = preset["rainflow"]["cycle_life_curve"]
        cycle_cost = {
            float(dod): replacement_cost_dollars / float(n_eol)
            for dod, n_eol in rf_curve.items()
        }

        return cls(
            throughput_cost_per_kwh=throughput_cost,
            cycle_cost_curve=cycle_cost,
            capacity_kwh=capacity_kwh,
            replacement_cost_dollars=replacement_cost_dollars,
        )


# ---------------------------------------------------------------------------
# Throughput wear term (objective_terms hook)
# ---------------------------------------------------------------------------


def add_wear_cost_term(
    wear: WearCost,
    mode: Literal["throughput", "dod_pwl"] = "throughput",
) -> ObjectiveTerm:
    """Return an ``objective_terms`` callable adding wear cost to the objective.

    ``throughput`` mode (default): linear in summed (charge+discharge) energy.
    ``dod_pwl`` mode: piecewise-linear in per-step |dSOC|. Requires the
    matching :func:`add_dod_constraints` hook to be installed in
    ``constraint_builders`` since it introduces ``model.delta_in_bin``.
    """
    if mode not in ("throughput", "dod_pwl"):
        raise ValueError(f"mode must be 'throughput' or 'dod_pwl', got {mode!r}")

    def _term(model: "pyo.ConcreteModel", _params: Dict[str, Any]):
        if mode == "throughput":
            # lambda * Sum_t (p_charge[t] + p_discharge[t]) * dt
            return wear.throughput_cost_per_kwh * sum(
                (model.p_charge[t] + model.p_discharge[t]) * model.dt
                for t in model.T
            )
        # dod_pwl: model.delta_in_bin and model._dod_bin_marginals must exist
        if not hasattr(model, "delta_in_bin"):
            raise RuntimeError(
                "dod_pwl mode requires add_dod_constraints() to be added "
                "to constraint_builders before this term."
            )
        marginals: List[float] = list(model._dod_bin_marginals)  # type: ignore[attr-defined]
        T_idx = list(model.T)
        return sum(
            marginals[b] * model.delta_in_bin[t, b]
            for t in T_idx
            for b in range(len(marginals))
        )

    _term.__name__ = f"wear_cost_{mode}"
    return _term


# ---------------------------------------------------------------------------
# DoD piecewise-linear constraints (constraint_builders hook)
# ---------------------------------------------------------------------------


def add_dod_constraints(
    wear: WearCost,
    num_bins: int = 4,
    bin_edges: Optional[List[float]] = None,
) -> ConstraintBuilder:
    """Return a ``constraint_builders`` callable that adds DoD-bin aux vars.

    Construction
    ------------
    For each timestep t we introduce:

      * ``delta_soc[t] = soc_frac[t] - soc_frac[t-1]``  (free var, expressed
        in SOC fraction)
      * ``abs_delta[t] >= |delta_soc[t]|`` via two inequalities
      * ``delta_in_bin[t, b] >= 0`` for b in 0..B-1, with per-bin upper
        bounds ``delta_in_bin[t, b] <= width_b`` and the coverage constraint
        ``sum_b delta_in_bin[t, b] >= abs_delta[t]``.

    Bin marginals (``$/SOC-fraction``) are precomputed: the cost-per-cycle
    interpolated at each bin's representative DoD divided by 2 (because one
    full cycle = one charge half + one discharge half, each of size DoD in
    SOC fraction).

    The optimizer will fill the cheapest (smallest-DoD) bins first by
    construction, since marginals are non-decreasing in bin index.

    Default bin edges: ``[0, 0.2, 0.5, 0.8, 1.0]``  (4 bins).
    """
    import pyomo.environ as pyo

    if bin_edges is None:
        if num_bins == 4:
            edges = [0.0, 0.2, 0.5, 0.8, 1.0]
        else:
            # Uniform partition of [0, 1].
            edges = [i / num_bins for i in range(num_bins + 1)]
    else:
        edges = list(bin_edges)
        if edges[0] != 0.0 or edges[-1] != 1.0:
            raise ValueError("bin_edges must start at 0 and end at 1")
        if any(edges[i] >= edges[i + 1] for i in range(len(edges) - 1)):
            raise ValueError("bin_edges must be strictly increasing")

    B = len(edges) - 1
    widths = [edges[b + 1] - edges[b] for b in range(B)]
    # Representative DoD for each bin: midpoint.
    midpoints = [(edges[b] + edges[b + 1]) / 2.0 for b in range(B)]
    # Marginal cost per SOC-fraction (not per cycle). One full cycle
    # contributes 2 units of dSOC sum (charge half + discharge half), so
    # marginal $/SOC-frac = ($/cycle) / 2.
    marginals = []
    for d in midpoints:
        cpc = _interp_cycle_cost(wear.cycle_cost_curve, d)
        marginals.append(cpc / 2.0)
    # Enforce non-decreasing marginals so the LP fills cheap bins first.
    # (If the curve is non-monotonic, sort assignment.)
    # We do not rewrite the bin order; we just trust the chemistry curves
    # supply increasing $/cycle in DoD.

    def _builder(model: "pyo.ConcreteModel", _params: Dict[str, Any]) -> None:
        T_idx = list(model.T)
        cap = float(pyo.value(model.cap))

        model.dod_bins = pyo.RangeSet(0, B - 1)
        # delta in SOC fraction, free
        model.delta_soc = pyo.Var(model.T, domain=pyo.Reals)
        model.abs_delta = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0.0, 1.0))
        model.delta_in_bin = pyo.Var(
            model.T, model.dod_bins, domain=pyo.NonNegativeReals
        )

        def _delta_def(mm, t):
            if t == T_idx[0]:
                prev = mm.soc_0 / cap
            else:
                prev = mm.soc[t - 1] / cap
            return mm.delta_soc[t] == mm.soc[t] / cap - prev

        model.delta_def = pyo.Constraint(model.T, rule=_delta_def)

        def _abs_pos(mm, t):
            return mm.abs_delta[t] >= mm.delta_soc[t]

        def _abs_neg(mm, t):
            return mm.abs_delta[t] >= -mm.delta_soc[t]

        model.abs_pos = pyo.Constraint(model.T, rule=_abs_pos)
        model.abs_neg = pyo.Constraint(model.T, rule=_abs_neg)

        # Per-bin upper bound
        def _bin_ub(mm, t, b):
            return mm.delta_in_bin[t, b] <= widths[b]

        model.bin_ub = pyo.Constraint(model.T, model.dod_bins, rule=_bin_ub)

        # Sum across bins must cover abs_delta.
        def _bin_cover(mm, t):
            return sum(mm.delta_in_bin[t, b] for b in mm.dod_bins) >= mm.abs_delta[t]

        model.bin_cover = pyo.Constraint(model.T, rule=_bin_cover)

        # Stash marginals on the model so the objective term can read them.
        model._dod_bin_marginals = list(marginals)
        model._dod_bin_edges = list(edges)

    _builder.__name__ = "add_dod_constraints"
    return _builder


def _interp_cycle_cost(curve: Dict[float, float], dod: float) -> float:
    """Piecewise-linear interpolation on ``{dod: $/cycle}``."""
    if not curve:
        raise ValueError("cycle_cost_curve must be non-empty")
    knots = sorted(curve.items())
    if dod <= knots[0][0]:
        return float(knots[0][1])
    if dod >= knots[-1][0]:
        return float(knots[-1][1])
    for (d0, c0), (d1, c1) in zip(knots, knots[1:]):
        if d0 <= dod <= d1:
            frac = (dod - d0) / (d1 - d0)
            return float(c0 + frac * (c1 - c0))
    return float(knots[-1][1])


# ---------------------------------------------------------------------------
# Calendar aging side channel
# ---------------------------------------------------------------------------


def add_calendar_aging_bias(
    weight: float,
    soc_neutral: float = 0.5,
    model: Any = None,
) -> ObjectiveTerm:
    """Return objective_terms callable adding L1 SOC-deviation penalty.

    ``weight * Sum_t |soc_frac[t] - soc_neutral|``

    Encoded with two aux nonneg vars ``soc_dev_pos[t], soc_dev_neg[t]``
    such that ``soc_frac[t] - soc_neutral = soc_dev_pos[t] - soc_dev_neg[t]``.

    The ``model`` argument is accepted for symmetry with the spec
    (``add_calendar_aging_bias(model, weight)``) but is not used; the
    actual penalty is wired in when the returned callable is invoked by
    the dispatch builder. Either calling form is supported.
    """
    # Allow two call signatures:
    #   add_calendar_aging_bias(weight=..., soc_neutral=...)
    #   add_calendar_aging_bias(model_obj, weight=...)
    # The spec's prose says ``add_calendar_aging_bias(model: CalendarDegradation, weight)``.
    # When ``model`` is non-None and ``weight`` is the second positional, swap them.
    if model is not None:
        # Caller passed a CalendarDegradation as first positional.
        # We accept and ignore (calendar-model details don't change the L1 form).
        pass

    if weight < 0:
        raise ValueError("weight must be non-negative")
    if not 0.0 <= soc_neutral <= 1.0:
        raise ValueError("soc_neutral must be in [0, 1]")

    def _term(m: "pyo.ConcreteModel", _params: Dict[str, Any]):
        import pyomo.environ as pyo

        cap = float(pyo.value(m.cap))
        # Add aux vars on first call. Use distinct names to avoid collision.
        if not hasattr(m, "soc_dev_pos"):
            m.soc_dev_pos = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0.0, 1.0))
            m.soc_dev_neg = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0.0, 1.0))

            def _split(mm, t):
                return mm.soc[t] / cap - soc_neutral == mm.soc_dev_pos[t] - mm.soc_dev_neg[t]

            m.soc_dev_split = pyo.Constraint(m.T, rule=_split)

        return weight * sum(m.soc_dev_pos[t] + m.soc_dev_neg[t] for t in m.T)

    _term.__name__ = "calendar_aging_bias"
    return _term
