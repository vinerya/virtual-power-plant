"""Bill simulation over arbitrary windows: billing cycles + NEM export credit.

:meth:`Tariff.bill` computes *one* bill: monthly fixed charges and minimum
bills are applied once and demand charges see a single peak. Billing a
90-day trace in one call would therefore under-charge. :func:`simulate_bill`
splits long windows into monthly billing cycles (anchored on the window's
start day in the simulation timezone, like a meter-read cycle), bills each
cycle, applies the NEM export credit per cycle, and aggregates.

Cycle modes: ``single`` (one bill for the whole window), ``monthly``
(always split), ``auto`` (split only when the window exceeds 31 days).
A trailing partial cycle is billed as a full cycle (fixed/minimum charges are
not prorated) -- the per-cycle breakdown makes that visible.
"""

from __future__ import annotations

import calendar
from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import datetime, timedelta, tzinfo
from itertools import pairwise
from typing import TYPE_CHECKING

from .components import BillingPeriod, BillLineItem
from .meter import MeterTrace
from .nem import ExportCredit, compute_export_credit, normalize_regime

if TYPE_CHECKING:
    from .tariff import Tariff

CYCLE_MODES = ("auto", "single", "monthly")


@dataclass
class CycleBill:
    start: datetime
    end: datetime
    line_items: list[BillLineItem]
    total: float
    credit: ExportCredit


@dataclass
class SimulationResult:
    tariff_name: str
    start: datetime
    end: datetime
    cycles: list[CycleBill]
    line_items: list[BillLineItem]
    total: float
    nem_regime: str
    export_kwh: float = 0.0
    export_credit: float = 0.0
    notes: list[str] = field(default_factory=list)


def _add_month(local: datetime, anchor_day: int) -> datetime:
    year = local.year + (local.month == 12)
    month = local.month % 12 + 1
    day = min(anchor_day, calendar.monthrange(year, month)[1])
    return local.replace(year=year, month=month, day=day)


def billing_cycles(
    start: datetime, end: datetime, tz: tzinfo, mode: str = "auto"
) -> list[tuple[datetime, datetime]]:
    if mode not in CYCLE_MODES:
        raise ValueError(f"billing cycle mode must be one of {CYCLE_MODES}")
    if mode == "single" or (mode == "auto" and end - start <= timedelta(days=31)):
        return [(start, end)]
    anchor = start.astimezone(tz).day
    cycles: list[tuple[datetime, datetime]] = []
    cur = start
    while cur < end:
        nxt = min(_add_month(cur.astimezone(tz), anchor), end)
        cycles.append((cur, nxt))
        cur = nxt
    return cycles


def _slice(trace: MeterTrace, start: datetime, end: datetime) -> MeterTrace:
    """Intervals of ``trace`` starting in ``[start, end)`` (bisect; trace is sorted)."""
    lo = bisect_left(trace.timestamps, start)
    hi = bisect_left(trace.timestamps, end)
    return MeterTrace(
        timestamps=trace.timestamps[lo:hi],
        import_kwh=trace.import_kwh[lo:hi],
        export_kwh=trace.export_kwh[lo:hi],
        interval_minutes=trace.interval_minutes,
        tz=trace.tz,
    )


def _merge(items: list[BillLineItem], n_cycles: int) -> list[BillLineItem]:
    merged: dict[tuple[str, str, str], BillLineItem] = {}
    rates: dict[tuple[str, str, str], set[float]] = {}
    for li in items:
        key = (li.kind, li.label, li.unit)
        rates.setdefault(key, set()).add(li.rate)
        if key not in merged:
            merged[key] = BillLineItem(
                kind=li.kind,
                label=li.label,
                quantity=li.quantity,
                unit=li.unit,
                rate=li.rate,
                amount=li.amount,
                meta=dict(li.meta),
            )
        else:
            m = merged[key]
            m.quantity = round(m.quantity + li.quantity, 4)
            m.amount = round(m.amount + li.amount, 4)
    for key, m in merged.items():
        if len(rates[key]) > 1:
            m.rate = round(m.amount / m.quantity, 6) if m.quantity else 0.0
        if n_cycles > 1 and m.kind == "demand":
            m.unit = "kW-mo"  # billed kW summed over cycles
    return list(merged.values())


def simulate_bill(
    tariff: Tariff,
    trace: MeterTrace,
    start: datetime,
    end: datetime,
    *,
    nem_regime: str = "none",
    avoided_cost: list[float] | tuple[float, ...] | None = None,
    cycle_mode: str = "auto",
) -> SimulationResult:
    """Bill ``trace`` against ``tariff`` over ``[start, end)``.

    Raises :class:`~vpp.tariffs.nem.NEMConfigError` when the NEM regime
    cannot be evaluated.
    """
    regime = normalize_regime(nem_regime)
    ts = trace.timestamps
    if any(b < a for a, b in pairwise(ts)):
        order = sorted(range(len(ts)), key=ts.__getitem__)
        trace = MeterTrace(
            timestamps=[ts[i] for i in order],
            import_kwh=[trace.import_kwh[i] for i in order],
            export_kwh=[trace.export_kwh[i] for i in order],
            interval_minutes=trace.interval_minutes,
            tz=trace.tz,
        )
    cycles: list[CycleBill] = []
    notes: list[str] = []
    for c_start, c_end in billing_cycles(start, end, trace.tz, cycle_mode):
        sub = _slice(trace, c_start, c_end)
        bill = tariff.bill(sub, BillingPeriod(start=c_start, end=c_end))
        credit = compute_export_credit(
            tariff,
            sub,
            c_start,
            c_end,
            regime=regime,
            avoided_cost=avoided_cost,
            bill=bill,
        )
        items = list(bill.line_items)
        total = bill.total
        if credit.line_item is not None:
            items.append(credit.line_item)
            total = round(total - credit.amount, 4)
        if credit.note and credit.note not in notes:
            notes.append(credit.note)
        cycles.append(CycleBill(c_start, c_end, items, total, credit))

    all_items = [li for c in cycles for li in c.line_items]
    return SimulationResult(
        tariff_name=tariff.name,
        start=start,
        end=end,
        cycles=cycles,
        line_items=_merge(all_items, len(cycles)) if len(cycles) > 1 else all_items,
        total=round(sum(c.total for c in cycles), 4),
        nem_regime=regime,
        export_kwh=round(sum(c.credit.export_kwh for c in cycles), 4),
        export_credit=round(sum(c.credit.amount for c in cycles), 4),
        notes=notes,
    )
