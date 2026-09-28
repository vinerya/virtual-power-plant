"""Presentational view of a URDB tariff for UIs.

The stored representation of a tariff is its URDB JSON; the billing engine
parses that into rate components. Consoles need something flatter: a list of
human-readable components and a month x hour rate grid for the TOU schedule.
:func:`describe_tariff` derives both *from the same parsed components the
engine bills with*, so what the UI shows is what the simulator charges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .components import (
    AdderRate,
    DemandCharge,
    FixedCharge,
    MinimumBill,
    TaxRate,
    TieredEnergyRate,
    TimeOfUseRate,
    TOUSchedule,
)
from .nem import nem_config_from_urdb
from .urdb import load_urdb_json

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
WEEKDAYS = (True, True, True, True, True, False, False)


@dataclass
class ComponentView:
    name: str
    kind: str
    unit: str
    rate: float | None = None
    rates: list[float] | None = None
    tiers: list[dict[str, float | None]] | None = None
    sell_rate: float | None = None
    schedule: list[str] = field(default_factory=list)
    detail: str | None = None


@dataclass
class TariffView:
    sector: str | None
    source: str | None
    description: str | None
    components: list[ComponentView]
    tou_heatmap: list[list[float]] | None
    tou_heatmap_weekend: list[list[float]] | None
    is_tou: bool
    nem_regime: str
    nem_source: str
    parse_error: str | None = None


def _month_ranges(months: frozenset[int] | set[int]) -> str:
    ms = sorted(months)
    if len(ms) == 12:
        return "all year"
    runs: list[tuple[int, int]] = []
    for m in ms:
        if runs and m == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], m)
        else:
            runs.append((m, m))
    # Merge a Dec->Jan wrap (e.g. Oct-Dec + Jan-May => Oct-May).
    if len(runs) > 1 and runs[0][0] == 1 and runs[-1][1] == 12:
        runs = [(runs[-1][0], runs[0][1]), *runs[1:-1]]
    return ", ".join(
        MONTHS[a - 1] if a == b else f"{MONTHS[a - 1]}-{MONTHS[b - 1]}" for a, b in runs
    )


def _schedule_lines(schedules: list[TOUSchedule]) -> list[str]:
    # (months, hours) -> set of day kinds
    grouped: dict[tuple[frozenset[int], tuple[int, int]], set[str]] = {}
    for s in schedules:
        kind = "weekdays" if s.weekday_mask == WEEKDAYS else "weekends/holidays"
        grouped.setdefault((s.season_mask, s.hour_range), set()).add(kind)
    lines = []
    for (months, (h0, h1)), kinds in sorted(
        grouped.items(), key=lambda kv: (min(kv[0][0]), kv[0][1])
    ):
        days = "every day" if len(kinds) == 2 else next(iter(kinds))
        lines.append(f"{_month_ranges(months)}, {days} {h0:02d}:00-{h1 % 24 or 24:02d}:00")
    return lines


def _tiers(tiers: list[tuple[float, float]]) -> list[dict[str, float | None]]:
    return [{"max_kwh": None if t == float("inf") else t, "rate": r} for t, r in tiers]


def _components(tariff) -> list[ComponentView]:
    out: list[ComponentView] = []
    for comp in tariff.components:
        if isinstance(comp, TimeOfUseRate):
            for label, schedules in sorted(comp.periods.items()):
                first = schedules[0] if schedules else None
                tiers = comp.period_tiers.get(label)
                out.append(
                    ComponentView(
                        name=f"Energy — {label.replace('_', ' ')}",
                        kind="energy",
                        unit="$/kWh",
                        rate=first.rate if first else None,
                        rates=[r for _t, r in tiers] if tiers else None,
                        tiers=_tiers(tiers) if tiers else None,
                        sell_rate=first.sell_rate if first else None,
                        schedule=_schedule_lines(schedules),
                    )
                )
        elif isinstance(comp, TieredEnergyRate):
            out.append(
                ComponentView(
                    name="Energy (tiered)",
                    kind="tier",
                    unit="$/kWh",
                    rates=[r for _t, r in comp.tiers],
                    tiers=_tiers(comp.tiers),
                    detail="Inclining block on total monthly kWh",
                )
            )
        elif isinstance(comp, DemandCharge):
            detail = {
                "monthly_max": "Highest interval kW in the billing period",
                "on_peak_max": (
                    f"Highest kW {comp.on_peak_hours[0]:02d}:00-"
                    f"{comp.on_peak_hours[1]:02d}:00"
                    + (" on weekdays" if comp.on_peak_weekdays_only else "")
                ),
                "coincident": "Highest kW in coincident-peak hours",
            }[comp.window]
            if comp.ratchet_pct > 0:
                detail += f"; {comp.ratchet_pct:.0%} ratchet"
            out.append(
                ComponentView(
                    name=comp.label or "Demand",
                    kind="demand",
                    unit="$/kW",
                    rate=comp.rate,
                    detail=detail,
                )
            )
        elif isinstance(comp, FixedCharge):
            out.append(
                ComponentView(
                    name=comp.label,
                    kind="fixed",
                    unit="$/day" if comp.frequency == "daily" else "$/month",
                    rate=comp.amount,
                )
            )
        elif isinstance(comp, MinimumBill):
            out.append(
                ComponentView(
                    name="Minimum bill",
                    kind="minimum",
                    unit="$/month",
                    rate=comp.amount,
                    detail="Bill is topped up to this amount",
                )
            )
        elif isinstance(comp, AdderRate):
            out.append(
                ComponentView(
                    name=comp.name,
                    kind="adder",
                    unit="$/kWh" if comp.basis == "per_kwh" else "fraction",
                    rate=comp.rate,
                    detail=f"Applied to {comp.applies_to}"
                    if comp.basis == "percent"
                    else "Per imported kWh",
                )
            )
        elif isinstance(comp, TaxRate):
            out.append(
                ComponentView(
                    name=comp.name,
                    kind="tax",
                    unit="fraction",
                    rate=comp.rate,
                    detail=f"Applied to {comp.applies_to}"
                    + (f" ({comp.jurisdiction})" if comp.jurisdiction else ""),
                )
            )
    return out


def _heatmap(data: dict[str, Any], key: str) -> list[list[float]] | None:
    struct = data.get("energyratestructure")
    sched = data.get(key)
    if not struct or not sched:
        return None
    try:
        rates = [
            float(p[0].get("rate", 0.0)) + float(p[0].get("adj", 0.0)) if p else 0.0
            for p in struct
        ]
        grid = [[round(rates[int(sched[m][h])], 6) for h in range(24)] for m in range(12)]
    except (IndexError, TypeError, ValueError, AttributeError):
        return None
    return grid


def describe_tariff(data: dict[str, Any] | None) -> TariffView:
    data = data or {}
    nem = nem_config_from_urdb(data)
    parse_error = None
    components: list[ComponentView] = []
    try:
        components = _components(load_urdb_json(data))
    except Exception as exc:  # show the tariff, flag it as unbillable
        parse_error = str(exc) or exc.__class__.__name__
    wd = _heatmap(data, "energyweekdayschedule")
    we = _heatmap(data, "energyweekendschedule")
    distinct = {v for grid in (wd or [], we or []) for row in grid for v in row}
    return TariffView(
        sector=data.get("sector") or None,
        source=data.get("source") or data.get("uri") or None,
        description=data.get("description") or data.get("_comment") or None,
        components=components,
        tou_heatmap=wd,
        tou_heatmap_weekend=we,
        is_tou=len(distinct) > 1,
        nem_regime=nem.regime,
        nem_source=nem.source,
        parse_error=parse_error,
    )
