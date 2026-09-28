"""Net-metering (export compensation) for tariff bills.

A :class:`~vpp.tariffs.tariff.Tariff` bills *imports*; what exported energy
earns depends on the customer's interconnection regime. This module turns an
export trace into a single ``credit`` :class:`BillLineItem` for one billing
window. It is shared by the bill simulator (``POST /api/v1/tariffs/.../simulate``)
and the customer-portal bill so both credit exports identically.

Regimes
-------
``none``
    Exports earn nothing.
``nem2``
    Net Energy Metering 2.0: each exported kWh is credited at the TOU period
    rate in effect when it was exported -- the period's URDB ``sell`` rate when
    the tariff defines one, otherwise the retail import rate. Tariffs without
    TOU periods (tiered-only) fall back to the bill's blended energy rate.
``nem3``
    Net Billing Tariff (CA "NEM 3.0"): exports are credited at an hourly
    avoided-cost vector (``$/kWh`` indexed by *local* time), see
    :func:`avoided_cost_at` for the accepted shapes.
``net_billing``
    Generic net billing (URDB ``dgrules`` = "Net Billing ..." / "Buy All Sell
    All"): exports are credited only at explicit URDB ``sell`` rates; intervals
    whose TOU period has no ``sell`` rate are left uncredited.

Where the regime comes from
---------------------------
:func:`nem_config_from_urdb` derives it from the tariff's URDB JSON:

1. the extension key ``"nem"`` (``"none" | "nem2" | "nem3" | "net_billing"``)
   plus optional ``"nem3_avoided_cost"`` (list of $/kWh), mirroring how this
   codebase already extends URDB with ``adders`` and ``taxes``;
2. otherwise URDB's own ``dgrules`` field ("Net Metering" -> ``nem2``,
   "Net Billing Instantaneous" / "Net Billing Hourly" / "Buy All Sell All" ->
   ``net_billing``, or ``nem3`` when an avoided-cost vector is present);
3. otherwise ``none``.

Avoided-cost shapes
-------------------
``nem3_avoided_cost`` may be (all indexed by *local* wall-clock time):

* 1 value -- flat rate;
* 24 values -- hour of day, repeated every day;
* 12 x 24 -- month x hour of day (CPUC ACC style), given either as 12 lists
  of 24 or flattened month-major to 288 values;
* 8760 values -- hour of year (365 days); on Feb 29 of a leap year the
  Feb 28 values are reused;
* 8784 values -- hour of year on a leap-year calendar (366 days); in a
  non-leap year the Feb 29 block is skipped.

Any other shape, or a non-finite value, is rejected
(:func:`normalize_avoided_cost`). On DST days the repeated autumn hour uses
the same entry twice and the skipped spring hour is unused.

Known simplifications: credits are applied within a single billing window
(no month-to-month roll-over or annual true-up), and a credit may take a bill
below the tariff's minimum charge.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Any

from .components import BillLineItem, TimeOfUseRate

if TYPE_CHECKING:
    from .meter import MeterTrace
    from .tariff import Bill, Tariff

logger = logging.getLogger(__name__)

NEM_REGIMES = ("none", "nem2", "nem3", "net_billing")

_DGRULES_MAP = {
    "net metering": "nem2",
    "net billing instantaneous": "net_billing",
    "net billing hourly": "net_billing",
    "buy all sell all": "net_billing",
}


class NEMConfigError(ValueError):
    """The requested NEM regime cannot be evaluated (e.g. nem3 without avoided cost)."""


@dataclass(frozen=True)
class NEMConfig:
    regime: str = "none"
    avoided_cost: tuple[float, ...] = ()
    # 'request' | 'tariff' | 'urdb_dgrules' | 'default'
    source: str = "default"


@dataclass
class ExportCredit:
    """Result of crediting one billing window's exports."""

    regime: str
    export_kwh: float
    credited_kwh: float
    amount: float  # positive dollars credited
    line_item: BillLineItem | None
    note: str = ""


def normalize_regime(value: str | None) -> str:
    compact = "".join(ch for ch in (value or "none").lower() if ch not in " ._-")
    aliases = {
        "": "none",
        "none": "none",
        "nem2": "nem2",
        "nem20": "nem2",
        "nem3": "nem3",
        "nem30": "nem3",
        "nbt": "nem3",
        "netbilling": "net_billing",
    }
    regime = aliases.get(compact, compact)
    if regime not in NEM_REGIMES:
        raise NEMConfigError(f"Unknown NEM regime {value!r}; expected one of {NEM_REGIMES}")
    return regime


#: Accepted flat lengths of an avoided-cost vector -> what they index.
AVOIDED_COST_SHAPES: dict[int, str] = {
    1: "flat",
    24: "hour of day",
    12 * 24: "month x hour of day",
    8760: "hour of year",
    8784: "hour of leap year",
}


def normalize_avoided_cost(raw: Any) -> tuple[float, ...]:
    """Validate an avoided-cost vector and flatten it to a tuple of floats.

    Accepts a flat list of one of the lengths in :data:`AVOIDED_COST_SHAPES`
    or a 12 x 24 nested list (month x hour). Raises :class:`NEMConfigError`
    otherwise. ``None`` / empty gives ``()`` (no vector).
    """
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise NEMConfigError("nem3_avoided_cost must be a list of $/kWh values")
    if not raw:
        return ()
    if all(isinstance(row, (list, tuple)) for row in raw):
        if len(raw) != 12 or any(len(row) != 24 for row in raw):
            raise NEMConfigError(
                "nested nem3_avoided_cost must be 12 months x 24 hours "
                f"(got {len(raw)} rows of lengths {sorted({len(r) for r in raw})})"
            )
        raw = [v for row in raw for v in row]
    try:
        values = tuple(float(v) for v in raw)
    except (TypeError, ValueError) as exc:
        raise NEMConfigError("nem3_avoided_cost values must be numbers ($/kWh)") from exc
    if not all(math.isfinite(v) for v in values):
        raise NEMConfigError("nem3_avoided_cost values must be finite")
    if len(values) not in AVOIDED_COST_SHAPES:
        shapes = ", ".join(f"{n} ({what})" for n, what in AVOIDED_COST_SHAPES.items())
        raise NEMConfigError(
            f"nem3_avoided_cost has {len(values)} values; expected one of: {shapes}"
        )
    return values


def _hour_of_year(dt_local: datetime, *, leap_calendar: bool) -> int:
    """0-based hour of year of ``dt_local`` on a fixed 365- or 366-day calendar."""
    month, day = dt_local.month, dt_local.day
    if leap_calendar:
        ref = date(2024, month, day)  # any leap year
    else:
        ref = date(2023, month, min(day, 28) if month == 2 else day)  # Feb 29 -> Feb 28
    return (ref.timetuple().tm_yday - 1) * 24 + dt_local.hour


def avoided_cost_at(avoided_cost: tuple[float, ...] | list[float], dt_local: datetime) -> float:
    """Return the $/kWh avoided cost for local time ``dt_local`` (see module doc)."""
    n = len(avoided_cost)
    if n == 1:
        idx = 0
    elif n == 24:
        idx = dt_local.hour
    elif n == 12 * 24:
        idx = (dt_local.month - 1) * 24 + dt_local.hour
    elif n in (8760, 8784):
        idx = _hour_of_year(dt_local, leap_calendar=n == 8784)
    else:
        raise NEMConfigError(f"unsupported nem3_avoided_cost length {n}")
    return float(avoided_cost[idx])


def _avoided_cost(raw: Any) -> tuple[float, ...]:
    try:
        return normalize_avoided_cost(raw)
    except NEMConfigError as exc:
        logger.warning("Ignoring invalid tariff nem3_avoided_cost: %s", exc)
        return ()


def nem_config_from_urdb(data: dict[str, Any] | None) -> NEMConfig:
    """Derive the tariff's default NEM regime from its URDB JSON (see module doc)."""
    data = data or {}
    avoided = _avoided_cost(data.get("nem3_avoided_cost"))
    if data.get("nem") is not None:
        try:
            return NEMConfig(normalize_regime(str(data["nem"])), avoided, "tariff")
        except NEMConfigError:
            logger.warning("Ignoring unknown tariff 'nem' value %r", data.get("nem"))
    dg = str(data.get("dgrules") or "").strip().lower()
    if dg in _DGRULES_MAP:
        regime = _DGRULES_MAP[dg]
        if regime == "net_billing" and avoided:
            regime = "nem3"
        return NEMConfig(regime, avoided, "urdb_dgrules")
    return NEMConfig("none", avoided, "default")


def compute_export_credit(
    tariff: Tariff,
    trace: MeterTrace,
    start: datetime,
    end: datetime,
    *,
    regime: str,
    avoided_cost: list[float] | list[list[float]] | tuple[float, ...] | None = None,
    bill: Bill | None = None,
) -> ExportCredit:
    """Credit the exports of ``trace`` inside ``[start, end)`` under ``regime``.

    ``bill`` (the import bill for the same window) is only needed for the
    NEM 2.0 blended-rate fallback on tariffs without TOU periods.

    Raises :class:`NEMConfigError` for ``nem3`` without an avoided-cost vector.
    """
    regime = normalize_regime(regime)
    in_window = [
        (dt_local, exp)
        for _i, dt_local, _imp, exp in trace.iter_with_local()
        if exp > 0 and start <= dt_local.astimezone(timezone.utc) < end
    ]
    export_kwh = round(sum(exp for _dt, exp in in_window), 4)
    if regime == "none" or export_kwh <= 0:
        return ExportCredit(regime, export_kwh, 0.0, 0.0, None)

    credit = 0.0
    credited_kwh = 0.0
    note = ""
    tou = [c for c in tariff.components if isinstance(c, TimeOfUseRate)]

    if regime == "nem3":
        acc = normalize_avoided_cost(list(avoided_cost or []))
        if not acc:
            raise NEMConfigError(
                "nem3 requires an avoided-cost vector (nem3_avoided_cost, $/kWh by local time)"
            )
        for dt_local, exp in in_window:
            credit += exp * avoided_cost_at(acc, dt_local)
            credited_kwh += exp
    elif regime == "nem2" and not tou:
        # No TOU periods to be period-accurate about (flat/tiered-only tariff):
        # blended retail-rate proxy from the window's energy line items.
        items = bill.line_items if bill is not None else []
        e_amt = sum(li.amount for li in items if li.kind in {"energy", "tier"})
        e_kwh = sum(li.quantity for li in items if li.kind in {"energy", "tier"})
        avg_rate = (e_amt / e_kwh) if e_kwh > 0 else 0.0
        credit = export_kwh * avg_rate
        credited_kwh = export_kwh if avg_rate > 0 else 0.0
        note = "blended retail energy rate (tariff has no TOU periods)"
    else:  # nem2 with TOU, or net_billing
        for dt_local, exp in in_window:
            rate: float | None = None
            for comp in tou:
                if regime == "nem2":
                    rate = comp.export_rate(dt_local)
                else:
                    rate = _sell_rate(comp, dt_local)
                if rate is not None:
                    break
            if rate is None:
                continue
            credit += exp * rate
            credited_kwh += exp
        uncredited = export_kwh - credited_kwh
        if uncredited > 1e-9:
            note = (
                f"{uncredited:.3f} kWh exported in periods without a "
                + ("TOU rate" if regime == "nem2" else "URDB sell rate")
                + " left uncredited"
            )
            logger.warning("Export credit (%s) for %r: %s", regime, tariff.name, note)

    amount = round(credit, 4)
    if amount <= 0:
        return ExportCredit(regime, export_kwh, round(credited_kwh, 4), 0.0, None, note)
    label = {
        "nem2": "NEM2 export credit",
        "nem3": "NEM3 export credit",
        "net_billing": "Net billing export credit",
    }[regime]
    line = BillLineItem(
        kind="credit",
        label=label,
        quantity=export_kwh,
        unit="kWh",
        rate=round(amount / export_kwh, 6),
        amount=-amount,
        meta={"nem": regime, "credited_kwh": round(credited_kwh, 4)},
    )
    return ExportCredit(regime, export_kwh, round(credited_kwh, 4), amount, line, note)


def _sell_rate(tou: TimeOfUseRate, dt_local: datetime) -> float | None:
    label = tou._classify(dt_local)
    if label is None:
        return None
    for sch in tou.periods[label]:
        if dt_local.month in sch.season_mask:
            return sch.sell_rate
    return None
