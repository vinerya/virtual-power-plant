"""
Single-interval fleet power allocation LP.

Splits a site-level power target (export-positive kW) across a set of
heterogeneous resources for one dispatch interval. It is the formulation
behind ``POST /api/v1/optimization/dispatch``.

Model
-----
For each resource ``r`` with feasible output range ``[lo_r, hi_r]`` (kW,
export-positive; ``lo_r < 0`` means the resource can absorb power, e.g. a
battery charging) the output is split into a positive and a negative part,
each of which is further split into ``K`` equal-width segments::

    p_r = sum_k pos[r, k] - sum_k neg[r, k]
    0 <= pos[r, k] <= hi_r / K,   0 <= neg[r, k] <= -lo_r / K

Segment ``k`` carries marginal cost::

    c_r + loading_penalty * (2k + 1) / K

i.e. a piecewise-linear approximation of ``c_r * |p| + loading_penalty *
p^2 / p_max``. The quadratic part is a first-order stand-in for resistive
(I^2 R) conversion losses, which grow with the square of current. Because
the marginal cost of each segment is non-decreasing the LP fills segments
in order without needing binaries, and resources with equal marginal cost
end up loaded roughly proportionally to their headroom instead of the
solver picking an arbitrary vertex.

Resources with identical marginal costs are additionally coupled to equal
per-segment utilisation, so exact ties are shared in proportion to headroom
rather than resolved at an arbitrary LP vertex.

The balance constraint uses two penalised slack variables so an infeasible
target still yields the closest achievable allocation (and reports the
shortfall) instead of an ``infeasible`` status::

    sum_r p_r + short_up - short_down == target

Resource-level parameters (``resources`` list of dicts):

    id           : str, unique
    lo_kw        : float <= 0   (most negative output, i.e. max absorption)
    hi_kw        : float >= 0   (max output)
    cost_up      : float, marginal $/kWh when exporting (p > 0)
    cost_down    : float, marginal $/kWh when absorbing (p < 0)
"""
from __future__ import annotations

from typing import Any

import pyomo.environ as pyo

DEFAULT_SEGMENTS = 4


def validate_allocation_params(params: dict[str, Any]) -> None:
    """Raise ``ValueError`` if ``params`` is not a valid allocation problem."""
    resources: list[dict[str, Any]] = list(params.get("resources") or [])
    if not resources:
        raise ValueError("resources must be non-empty")
    ids = [str(r["id"]) for r in resources]
    if len(set(ids)) != len(ids):
        raise ValueError("resource ids must be unique")
    for r in resources:
        lo, hi = float(r["lo_kw"]), float(r["hi_kw"])
        if lo > 0 or hi < 0:
            raise ValueError(f"resource {r['id']}: need lo_kw <= 0 <= hi_kw, got [{lo}, {hi}]")
    if "target_kw" not in params:
        raise ValueError("target_kw is required")
    if float(params.get("dt_hours", 1.0)) <= 0:
        raise ValueError("dt_hours must be > 0")


def build_power_allocation_model(params: dict[str, Any]) -> pyo.ConcreteModel:
    """Build the single-interval allocation LP described in the module docstring."""
    validate_allocation_params(params)
    resources = list(params["resources"])
    target = float(params["target_kw"])
    K = int(params.get("segments", DEFAULT_SEGMENTS))
    if K < 1:
        raise ValueError("segments must be >= 1")
    loading_penalty = float(params.get("loading_penalty_per_kwh", 0.001))
    dt = float(params.get("dt_hours", 1.0))

    ids = [str(r["id"]) for r in resources]
    by_id = {str(r["id"]): r for r in resources}
    max_cost = max(
        [abs(float(r.get("cost_up", 0.0))) for r in resources]
        + [abs(float(r.get("cost_down", 0.0))) for r in resources]
        + [loading_penalty]
    )
    shortfall_penalty = float(params.get("shortfall_penalty_per_kwh", 1000.0 * (max_cost + 1.0)))

    m = pyo.ConcreteModel(name="power_allocation")
    m.R = pyo.Set(initialize=ids, ordered=True)
    m.K = pyo.RangeSet(0, K - 1)

    def _pos_bounds(_m, r, _k):
        return (0.0, max(0.0, float(by_id[r]["hi_kw"])) / K)

    def _neg_bounds(_m, r, _k):
        return (0.0, max(0.0, -float(by_id[r]["lo_kw"])) / K)

    m.pos = pyo.Var(m.R, m.K, domain=pyo.NonNegativeReals, bounds=_pos_bounds)
    m.neg = pyo.Var(m.R, m.K, domain=pyo.NonNegativeReals, bounds=_neg_bounds)
    m.short_up = pyo.Var(domain=pyo.NonNegativeReals)
    m.short_down = pyo.Var(domain=pyo.NonNegativeReals)

    # Resources with identical marginal costs form a class. The LP is
    # indifferent between them, so without a tie-break the solver would pick
    # an arbitrary vertex (e.g. 15/25 kW instead of 20/20 kW). Couple each
    # class to equal per-segment utilisation, i.e. exact headroom-proportional
    # sharing among equals.
    classes: dict[tuple, list[str]] = {}
    for r in ids:
        key = (
            round(float(by_id[r].get("cost_up", 0.0)), 12),
            round(float(by_id[r].get("cost_down", 0.0)), 12),
        )
        classes.setdefault(key, []).append(r)
    pairs_pos: list[tuple] = []
    pairs_neg: list[tuple] = []
    for members in classes.values():
        up = [r for r in members if float(by_id[r]["hi_kw"]) > 0]
        dn = [r for r in members if float(by_id[r]["lo_kw"]) < 0]
        pairs_pos += [(up[0], r) for r in up[1:]]
        pairs_neg += [(dn[0], r) for r in dn[1:]]

    def _share_pos(mm, i, k):
        ref, r = pairs_pos[i]
        return mm.pos[r, k] * float(by_id[ref]["hi_kw"]) == mm.pos[ref, k] * float(by_id[r]["hi_kw"])

    def _share_neg(mm, i, k):
        ref, r = pairs_neg[i]
        return mm.neg[r, k] * float(by_id[ref]["lo_kw"]) == mm.neg[ref, k] * float(by_id[r]["lo_kw"])

    m.share_pos = pyo.Constraint(range(len(pairs_pos)), m.K, rule=_share_pos)
    m.share_neg = pyo.Constraint(range(len(pairs_neg)), m.K, rule=_share_neg)

    m.p = pyo.Expression(
        m.R, rule=lambda mm, r: sum(mm.pos[r, k] for k in mm.K) - sum(mm.neg[r, k] for k in mm.K)
    )
    m.balance = pyo.Constraint(
        expr=sum(m.p[r] for r in m.R) + m.short_up - m.short_down == target
    )

    def _seg_cost(base: float, k: int) -> float:
        return base + loading_penalty * (2 * k + 1) / K

    m.cost = pyo.Objective(
        expr=dt * (
            sum(
                _seg_cost(float(by_id[r].get("cost_up", 0.0)), k) * m.pos[r, k]
                + _seg_cost(float(by_id[r].get("cost_down", 0.0)), k) * m.neg[r, k]
                for r in m.R
                for k in m.K
            )
            + shortfall_penalty * (m.short_up + m.short_down)
        ),
        sense=pyo.minimize,
    )
    return m


__all__ = ["DEFAULT_SEGMENTS", "build_power_allocation_model", "validate_allocation_params"]
