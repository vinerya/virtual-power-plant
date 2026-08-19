"""URDB JSON loader.

Reference: https://openei.org/services/doc/rest/util_rates/?version=8

Supported fields (M1):
    - energyratestructure        : list[ list[ {rate, max?, adj?, sell?} ] ]
    - energyweekdayschedule      : 12x24 matrix of period indices
    - energyweekendschedule      : 12x24 matrix of period indices
    - demandratestructure        : list[ list[ {rate, max?} ] ]
    - demandweekdayschedule      : 12x24 matrix
    - demandweekendschedule      : 12x24 matrix
    - flatdemandstructure        : list[ list[ {rate} ] ]
    - flatdemandmonths           : 12-array of period indices
    - fixedchargefirstmeter      : float
    - fixedchargeunits           : '$/month' | '$/day'
    - mincharge                  : float
    - utility, name, sector, startdate, source

Deferred (TODO):
    - tiered + TOU combined (we treat tiers only when schedules collapse to one period).
    - lookback-window fields, demandwindow.
    - non-USD currencies, taxes.
"""
from __future__ import annotations

import json
from pathlib import Path
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
from .tariff import Tariff


ALL_DAYS = (True,) * 7
WEEKDAYS = (True, True, True, True, True, False, False)
WEEKENDS = (False, False, False, False, False, True, True)
ALL_MONTHS = frozenset(range(1, 13))


def _hour_runs(hours: list[int]) -> list[tuple[int, int]]:
    """Compress a list of hour indices into (start, end_exclusive) ranges."""
    if not hours:
        return []
    hours = sorted(set(hours))
    runs: list[tuple[int, int]] = []
    start = prev = hours[0]
    for h in hours[1:]:
        if h == prev + 1:
            prev = h
        else:
            runs.append((start, prev + 1))
            start = prev = h
    runs.append((start, prev + 1))
    return runs


def _build_tou_schedules(
    rate_structure: list[list[dict[str, Any]]],
    weekday_sched: list[list[int]],
    weekend_sched: list[list[int]],
) -> tuple[dict[str, list[TOUSchedule]], bool]:
    """Walk the URDB schedule matrices and group into per-period TOUSchedule lists.

    Returns (periods_dict, is_tiered_only).
    is_tiered_only is True when there is exactly 1 period referenced and that
    period has multiple tiers — caller may fold into TieredEnergyRate.
    """
    n_periods = len(rate_structure)
    # period_idx -> { (mask_kind, month) : list[hour] }
    # mask_kind: 'wd' or 'we'
    bucket: dict[int, dict[tuple[str, int], list[int]]] = {
        p: {} for p in range(n_periods)
    }
    for month_idx in range(12):
        for hour in range(24):
            wd_p = weekday_sched[month_idx][hour]
            we_p = weekend_sched[month_idx][hour]
            bucket[wd_p].setdefault(("wd", month_idx + 1), []).append(hour)
            bucket[we_p].setdefault(("we", month_idx + 1), []).append(hour)

    periods: dict[str, list[TOUSchedule]] = {}
    referenced_periods = {p for p, b in bucket.items() if b}

    for p in referenced_periods:
        tiers = rate_structure[p]
        if not tiers:
            continue
        # Tier 0 rate (period rate); tiered handling done separately.
        rate = float(tiers[0].get("rate", 0.0)) + float(tiers[0].get("adj", 0.0))
        sell = tiers[0].get("sell")
        sell_rate = float(sell) if sell is not None else None
        label = f"period_{p}"
        scheds: list[TOUSchedule] = []
        # Group by (mask_kind, month) -> hour runs
        # Then merge contiguous months with identical hour-run sets.
        per_kind_month: dict[str, dict[int, list[tuple[int, int]]]] = {"wd": {}, "we": {}}
        for (kind, month), hours in bucket[p].items():
            per_kind_month[kind][month] = _hour_runs(hours)

        for kind, month_map in per_kind_month.items():
            mask = WEEKDAYS if kind == "wd" else WEEKENDS
            # Merge months that share the same hour-run signature.
            sig_to_months: dict[tuple, list[int]] = {}
            for month, runs in month_map.items():
                sig_to_months.setdefault(tuple(runs), []).append(month)
            for runs, months in sig_to_months.items():
                season = frozenset(months)
                for hr in runs:
                    scheds.append(
                        TOUSchedule(
                            weekday_mask=mask,
                            hour_range=hr,
                            season_mask=season,
                            rate=rate,
                            sell_rate=sell_rate,
                        )
                    )
        periods[label] = scheds

    is_tiered_only = (
        len(referenced_periods) == 1
        and len(rate_structure[next(iter(referenced_periods))]) > 1
    )
    return periods, is_tiered_only


def load_urdb_json(path_or_dict: str | Path | dict) -> Tariff:
    """Load a URDB-formatted tariff JSON into a :class:`Tariff`.

    Parameters
    ----------
    path_or_dict : str | Path | dict
        Path to URDB JSON or already-parsed dict.
    """
    if isinstance(path_or_dict, (str, Path)):
        with open(path_or_dict) as f:
            data = json.load(f)
    else:
        data = path_or_dict

    name = data.get("name", "URDB tariff")
    components: list = []

    # --- Energy ---
    e_struct = data.get("energyratestructure")
    e_wd = data.get("energyweekdayschedule")
    e_we = data.get("energyweekendschedule")
    if e_struct and e_wd and e_we:
        periods, tiered_only = _build_tou_schedules(e_struct, e_wd, e_we)
        if tiered_only:
            # Single period with multiple tiers => TieredEnergyRate
            tiers_in = e_struct[0]
            tiers: list[tuple[float, float]] = []
            for t in tiers_in:
                threshold = t.get("max")
                rate = float(t.get("rate", 0.0)) + float(t.get("adj", 0.0))
                tiers.append((float(threshold) if threshold is not None else float("inf"), rate))
            # ensure last tier has +inf threshold
            if tiers and tiers[-1][0] != float("inf"):
                tiers.append((float("inf"), tiers[-1][1]))
            components.append(TieredEnergyRate(tiers=tiers))
        elif periods:
            components.append(TimeOfUseRate(periods=periods))

    # --- TOU Demand ---
    d_struct = data.get("demandratestructure")
    d_wd = data.get("demandweekdayschedule")
    d_we = data.get("demandweekendschedule")
    if d_struct and d_wd and d_we:
        # Use the first non-empty period's tier-0 rate as on-peak; treat as
        # on_peak_max window inferred from weekday schedule.
        # Find weekday hours for period 1 (URDB convention: 0=off-peak, 1=on-peak).
        on_peak_hours: list[int] = []
        target_p = 1 if len(d_struct) > 1 else 0
        for h in range(24):
            if any(d_wd[m][h] == target_p for m in range(12)):
                on_peak_hours.append(h)
        if d_struct[target_p]:
            rate = float(d_struct[target_p][0].get("rate", 0.0))
            if on_peak_hours and rate > 0:
                runs = _hour_runs(on_peak_hours)
                # take widest run as the on-peak band
                start, end = max(runs, key=lambda r: r[1] - r[0])
                components.append(
                    DemandCharge(
                        rate=rate,
                        window="on_peak_max",
                        on_peak_hours=(start, end),
                        component_id=f"demand_p{target_p}",
                        label=f"Demand on-peak ({start:02d}-{end:02d})",
                    )
                )

    # --- Flat demand ---
    f_struct = data.get("flatdemandstructure")
    f_months = data.get("flatdemandmonths")
    if f_struct and f_months:
        # Single rate flat demand: take first period's tier-0 rate.
        rate = float(f_struct[0][0].get("rate", 0.0)) if f_struct[0] else 0.0
        if rate > 0:
            components.append(
                DemandCharge(
                    rate=rate,
                    window="monthly_max",
                    component_id="flat_demand",
                    label="Flat demand",
                )
            )

    # --- Fixed charge ---
    fixed = data.get("fixedchargefirstmeter")
    if fixed:
        units = data.get("fixedchargeunits", "$/month")
        freq = "daily" if "day" in units.lower() else "monthly"
        components.append(FixedCharge(amount=float(fixed), frequency=freq))

    # --- Minimum charge ---
    minc = data.get("mincharge")
    if minc:
        components.append(MinimumBill(amount=float(minc)))

    # --- Energy minimum (per-kWh floor) ---
    e_min = data.get("caenergyminratestructure")
    if e_min:
        # Map the first non-zero rate as a per-kWh adder. URDB shape varies;
        # accept either a list-of-list-of-{rate} or a flat list.
        try:
            if isinstance(e_min, list) and e_min and isinstance(e_min[0], list):
                rate = float(e_min[0][0].get("rate", 0.0))
            elif isinstance(e_min, list) and e_min:
                rate = float(e_min[0].get("rate", 0.0))
            else:
                rate = 0.0
        except (TypeError, ValueError, AttributeError, IndexError):
            rate = 0.0
        if rate > 0:
            components.append(
                AdderRate(
                    name="Energy minimum charge",
                    rate=rate,
                    basis="per_kwh",
                    applies_to="energy",
                )
            )

    # --- Taxes (URDB 'taxes' is a free-form list of {name, rate, jurisdiction?}) ---
    taxes = data.get("taxes")
    if isinstance(taxes, list):
        for t in taxes:
            try:
                rate = float(t.get("rate", 0.0))
            except (TypeError, ValueError):
                continue
            if rate <= 0:
                continue
            components.append(
                TaxRate(
                    name=str(t.get("name", "Tax")),
                    rate=rate,
                    jurisdiction=str(t.get("jurisdiction", "")),
                    applies_to=t.get("applies_to", "subtotal"),
                )
            )

    # --- Adders (custom 'adders' list in extended URDB JSON) ---
    adders = data.get("adders")
    if isinstance(adders, list):
        for a in adders:
            try:
                rate = float(a.get("rate", 0.0))
            except (TypeError, ValueError):
                continue
            if rate == 0:
                continue
            components.append(
                AdderRate(
                    name=str(a.get("name", "Adder")),
                    rate=rate,
                    basis=a.get("basis", "percent"),
                    applies_to=a.get("applies_to", "subtotal"),
                )
            )

    return Tariff(
        name=name,
        components=components,
        utility=data.get("utility", ""),
        sector=data.get("sector", "Residential"),
        source=data.get("source", ""),
        effective_date=str(data.get("startdate", "")),
    )
