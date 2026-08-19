"""Rate components for tariff billing.

Each component implements ``compute(trace, period) -> list[BillLineItem]``.

URDB schema reference: https://openei.org/services/doc/rest/util_rates/?version=8
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Literal, Protocol

from .calendar import SeasonConfig, is_weekend_or_holiday
from .meter import MeterTrace


# ---------------------------------------------------------------------------
# Bill data structures
# ---------------------------------------------------------------------------


@dataclass
class BillLineItem:
    """One row of a bill: kind, label, quantity, unit, rate, amount."""

    kind: str  # 'energy' | 'demand' | 'fixed' | 'minimum' | 'tier' | 'adder' | 'tax' | 'credit'
    label: str
    quantity: float
    unit: str  # 'kWh' | 'kW' | 'day' | 'month' | '$'
    rate: float
    amount: float
    meta: dict = field(default_factory=dict)


@dataclass
class BillingPeriod:
    """Closed-open billing window [start, end). Both should be tz-aware."""

    start: datetime
    end: datetime
    season_config: SeasonConfig = field(default_factory=SeasonConfig)
    # Prior 11-month demand peaks per component-id, used for ratchet calc.
    prior_peaks_kw: dict[str, list[float]] = field(default_factory=dict)

    @property
    def days(self) -> int:
        return max(1, (self.end - self.start).days)

    @property
    def months(self) -> float:
        # Approximate; URDB monthly charges typically billed 1x per period.
        return self.days / 30.4375


# ---------------------------------------------------------------------------
# Component protocol
# ---------------------------------------------------------------------------


class Component(Protocol):
    """Protocol all rate components implement.

    Maps to URDB ``energyratestructure`` / ``demandratestructure`` /
    ``fixedchargefirstmeter`` / ``mincharge`` etc.
    """

    def compute(
        self, trace: MeterTrace, period: BillingPeriod
    ) -> list[BillLineItem]:  # pragma: no cover
        ...


# ---------------------------------------------------------------------------
# TOU energy
# ---------------------------------------------------------------------------


@dataclass
class TOUSchedule:
    """One scheduling row for a TOU period.

    weekday_mask: 7-bool tuple (Mon..Sun); True = applies that day (holidays
        treated as weekend per URDB convention).
    hour_range: (start_hour_inclusive, end_hour_exclusive). Wraps midnight if
        end <= start.
    season_mask: set of months in which this row applies.
    rate: $/kWh.
    priority: higher number wins on overlap (default 0).
    sell_rate: $/kWh export compensation, from URDB
        ``energyratestructure[..].sell``. ``None`` when the source tariff
        doesn't define a distinct export rate for this period -- see
        :meth:`TimeOfUseRate.export_rate` for the same-as-import fallback.
    """

    weekday_mask: tuple[bool, ...]
    hour_range: tuple[int, int]
    season_mask: frozenset[int]
    rate: float
    priority: int = 0
    sell_rate: float | None = None


@dataclass
class TimeOfUseRate:
    """Time-of-use energy rate.

    Maps to URDB ``energyratestructure`` + ``energyweekdayschedule`` /
    ``energyweekendschedule``. Period labels (e.g. 'peak', 'off-peak') are
    arbitrary keys carried into line items.
    """

    periods: dict[str, list[TOUSchedule]]

    def _classify(self, dt_local: datetime) -> str | None:
        """Return the highest-priority matching period label for `dt_local`."""
        is_we = is_weekend_or_holiday(dt_local)
        wd = dt_local.weekday()
        hour = dt_local.hour
        month = dt_local.month
        best: tuple[int, str] | None = None
        for label, schedules in self.periods.items():
            for sch in schedules:
                if month not in sch.season_mask:
                    continue
                # weekday_mask has 7 entries Mon..Sun; treat holidays as Sat (idx 5)
                day_idx = 5 if is_we and wd < 5 else wd
                if not sch.weekday_mask[day_idx]:
                    continue
                start_h, end_h = sch.hour_range
                if start_h < end_h:
                    in_window = start_h <= hour < end_h
                else:
                    in_window = hour >= start_h or hour < end_h
                if not in_window:
                    continue
                if best is None or sch.priority > best[0]:
                    best = (sch.priority, label)
        return best[1] if best else None

    def export_rate(self, dt_local: datetime) -> float | None:
        """Return the $/kWh export (sell) rate in effect at ``dt_local``.

        Falls back to the period's import ``rate`` when the source tariff
        doesn't define a distinct ``sell_rate`` for that period -- this
        matches NEM 2.0's definition ("exports earn the same per-kWh price
        as imports") used throughout this codebase as the default. Returns
        ``None`` only when ``dt_local`` doesn't match any scheduled period.
        """
        label = self._classify(dt_local)
        if label is None:
            return None
        for sch in self.periods[label]:
            if dt_local.month in sch.season_mask:
                return sch.sell_rate if sch.sell_rate is not None else sch.rate
        return None

    def compute(self, trace: MeterTrace, period: BillingPeriod) -> list[BillLineItem]:
        totals_kwh: dict[str, float] = {}
        rates_seen: dict[str, float] = {}
        for _i, dt_local, imp, _exp in trace.iter_with_local():
            ts_utc = dt_local.astimezone(timezone.utc)
            if not (period.start <= ts_utc < period.end):
                continue
            label = self._classify(dt_local)
            if label is None:
                continue
            totals_kwh[label] = totals_kwh.get(label, 0.0) + imp
            # Capture the rate used; if multiple schedules in a period share
            # a label, we record the first encountered (URDB rates are uniform per period).
            if label not in rates_seen:
                # find rate
                for sch in self.periods[label]:
                    if dt_local.month in sch.season_mask:
                        rates_seen[label] = sch.rate
                        break
        items: list[BillLineItem] = []
        for label, kwh in totals_kwh.items():
            rate = rates_seen.get(label, 0.0)
            items.append(
                BillLineItem(
                    kind="energy",
                    label=f"TOU {label}",
                    quantity=kwh,
                    unit="kWh",
                    rate=rate,
                    amount=round(kwh * rate, 4),
                )
            )
        return items


# ---------------------------------------------------------------------------
# Tiered energy
# ---------------------------------------------------------------------------


@dataclass
class TieredEnergyRate:
    """Inclining-block energy rate, applied monthly.

    Maps to URDB ``energyratestructure`` with multiple tiers per period (when
    ``energyweekdayschedule`` references a single period).

    tiers: list of (threshold_kwh, rate_$/kwh). Sentinel ``float('inf')`` for last.
    """

    tiers: list[tuple[float, float]]

    def compute(self, trace: MeterTrace, period: BillingPeriod) -> list[BillLineItem]:
        total_kwh = 0.0
        for ts, imp in zip(trace.timestamps, trace.import_kwh):
            if period.start <= ts < period.end:
                total_kwh += imp
        items: list[BillLineItem] = []
        prev_threshold = 0.0
        remaining = total_kwh
        for i, (threshold, rate) in enumerate(self.tiers):
            tier_size = threshold - prev_threshold
            consumed = min(remaining, tier_size) if tier_size != float("inf") else remaining
            if consumed <= 0:
                break
            items.append(
                BillLineItem(
                    kind="tier",
                    label=f"Tier {i + 1} (<= {threshold} kWh)",
                    quantity=round(consumed, 4),
                    unit="kWh",
                    rate=rate,
                    amount=round(consumed * rate, 4),
                )
            )
            remaining -= consumed
            prev_threshold = threshold
            if remaining <= 0:
                break
        return items


# ---------------------------------------------------------------------------
# Demand charges
# ---------------------------------------------------------------------------

DemandWindow = Literal["monthly_max", "on_peak_max", "coincident"]


@dataclass
class DemandCharge:
    """Demand charge ($/kW) over a window.

    Maps to URDB ``demandratestructure`` / ``flatdemandstructure``.

    rate: $/kW.
    window:
        - 'monthly_max' : peak interval over the entire billing period.
        - 'on_peak_max' : peak only within `on_peak_hours` (local).
        - 'coincident'  : peak only within `coincident_hours` of a peak day.
    ratchet_pct: 0..1. Floor = ratchet_pct * max(prior_peaks_kw[id]).
    component_id: stable id used to look up prior peaks for ratchet.
    """

    rate: float
    window: DemandWindow = "monthly_max"
    ratchet_pct: float = 0.0
    on_peak_hours: tuple[int, int] = (16, 21)
    on_peak_weekdays_only: bool = True
    coincident_hours: tuple[int, int] = (17, 20)
    component_id: str = "demand"
    label: str | None = None

    def _in_window(self, dt_local: datetime) -> bool:
        if self.window == "monthly_max":
            return True
        if self.window == "on_peak_max":
            if self.on_peak_weekdays_only and is_weekend_or_holiday(dt_local):
                return False
            s, e = self.on_peak_hours
            h = dt_local.hour
            return s <= h < e if s < e else (h >= s or h < e)
        # coincident
        s, e = self.coincident_hours
        h = dt_local.hour
        return s <= h < e if s < e else (h >= s or h < e)

    def compute(self, trace: MeterTrace, period: BillingPeriod) -> list[BillLineItem]:
        peak_kw = 0.0
        for _i, dt_local, imp, _exp in trace.iter_with_local():
            ts_utc = dt_local.astimezone(timezone.utc)
            if not (period.start <= ts_utc < period.end):
                continue
            if not self._in_window(dt_local):
                continue
            kw = imp / trace.interval_hours
            if kw > peak_kw:
                peak_kw = kw

        # Ratchet floor
        ratcheted = False
        floor_kw = 0.0
        priors = period.prior_peaks_kw.get(self.component_id, [])
        if self.ratchet_pct > 0 and priors:
            floor_kw = self.ratchet_pct * max(priors)
            if floor_kw > peak_kw:
                ratcheted = True
                billed_kw = floor_kw
            else:
                billed_kw = peak_kw
        else:
            billed_kw = peak_kw

        label = self.label or f"Demand ({self.window})"
        return [
            BillLineItem(
                kind="demand",
                label=label,
                quantity=round(billed_kw, 4),
                unit="kW",
                rate=self.rate,
                amount=round(billed_kw * self.rate, 4),
                meta={
                    "actual_peak_kw": round(peak_kw, 4),
                    "ratchet_floor_kw": round(floor_kw, 4),
                    "ratcheted": ratcheted,
                    "component_id": self.component_id,
                },
            )
        ]


# ---------------------------------------------------------------------------
# Fixed and minimum
# ---------------------------------------------------------------------------


@dataclass
class FixedCharge:
    """Flat customer charge.

    Maps to URDB ``fixedchargefirstmeter`` (frequency = 'monthly') or daily.
    """

    amount: float
    frequency: Literal["daily", "monthly"] = "monthly"
    label: str = "Fixed charge"

    def compute(self, trace: MeterTrace, period: BillingPeriod) -> list[BillLineItem]:
        if self.frequency == "daily":
            qty = float(period.days)
            unit = "day"
        else:
            qty = 1.0
            unit = "month"
        return [
            BillLineItem(
                kind="fixed",
                label=self.label,
                quantity=qty,
                unit=unit,
                rate=self.amount,
                amount=round(qty * self.amount, 4),
            )
        ]


@dataclass
class AdderRate:
    """Surcharge / adder line item driven off a primary subtotal.

    Examples include state climate-credit surcharges, public-purpose
    program fees, franchise fees, and per-kWh non-bypassable charges.

    Parameters
    ----------
    name : str
        Line-item label.
    rate : float
        ``percent`` basis: fraction (0.015 == 1.5%).
        ``per_kwh`` basis: $/kWh.
    basis : 'percent' | 'per_kwh'
        How ``rate`` is interpreted.
    applies_to : 'subtotal' | 'energy' | 'demand'
        Which running total to multiply against. ``subtotal`` = sum of all
        primary line-items computed BEFORE adders/taxes run.

    Note
    ----
    Adders run AFTER all primary components (energy, demand, fixed,
    tier, minimum) inside :meth:`Tariff.bill`. They do NOT compound with
    each other — every adder is computed against the same primary
    subtotal/energy/demand snapshot. Taxes then run after all adders.
    """

    name: str
    rate: float
    basis: Literal["percent", "per_kwh"] = "percent"
    applies_to: Literal["subtotal", "energy", "demand"] = "subtotal"

    def compute(
        self, trace: MeterTrace, period: BillingPeriod
    ) -> list[BillLineItem]:  # pragma: no cover - marker
        # Marker only; Tariff applies adders against running subtotal.
        return []


@dataclass
class TaxRate:
    """Sales / use / utility tax line item.

    Multiple ``TaxRate`` instances stack ADDITIVELY against the SAME
    pre-tax base — they do not compound. This matches how state and local
    sales taxes typically appear on US utility bills.

    Parameters
    ----------
    name : str
        Line-item label, e.g. "California sales tax".
    rate : float
        Fractional rate (0.05 == 5%).
    jurisdiction : str
        Free-form identifier (state, county, city).
    applies_to : 'subtotal' | 'energy'
        Whether the tax is on the post-adder subtotal or only on energy
        line items. ``subtotal`` is the most common.
    """

    name: str
    rate: float
    jurisdiction: str = ""
    applies_to: Literal["subtotal", "energy"] = "subtotal"

    def compute(
        self, trace: MeterTrace, period: BillingPeriod
    ) -> list[BillLineItem]:  # pragma: no cover - marker
        return []


@dataclass
class MinimumBill:
    """Floor on the total bill.

    Maps to URDB ``mincharge``. Applied as a make-up adjustment computed
    after all other components in :class:`Tariff`.
    """

    amount: float
    label: str = "Minimum bill make-up"

    def compute(self, trace: MeterTrace, period: BillingPeriod) -> list[BillLineItem]:
        # Marker only; Tariff handles the make-up calculation.
        return []
