"""Customer bills computed by the tariff engine from site meter data.

For a customer and a calendar month the bill is::

    Tariff(customer.tariff_id).bill(MeterTrace(meter_readings of owned sites), month)

* **Month boundaries** are local midnight in the timezone of the customer's
  primary (oldest) site, so TOU periods and the billing window line up with
  the utility's local clock. Customers without a site bill in UTC.
* **Meter data** is the revenue-meter interval data ingested per site
  (``POST /api/v1/sites/{id}/meter-readings``). Multiple sites are summed per
  interval; if they report at different interval lengths everything is
  resampled to hourly (demand charges then see hourly peaks).
* **Coverage** (intervals present / intervals expected) is reported in
  ``metadata`` so a bill built on partial data is never mistaken for a
  complete one.
* **Baseline**: when the profile declares ``baseline_kwh_per_month``, the
  same tariff is applied to that energy spread flat across the month. This
  is a declared-typical-usage comparison, not a measured counterfactual;
  ``metadata.baseline_method`` says so.
* **Exports** (``metadata.export_kwh``) are credited under the NEM regime
  of the assigned tariff, derived from its URDB JSON by
  :func:`vpp.tariffs.nem.nem_config_from_urdb` (extension key ``nem`` /
  ``nem3_avoided_cost``, else URDB ``dgrules``). The regime is therefore a
  property of the tariff a customer is on: customers on different NEM
  vintages are assigned different tariff rows. Staff may pass an explicit
  ``nem_override`` (what-if); customers cannot. ``metadata`` reports
  ``nem_regime``, ``nem_source``, ``export_credit`` and whether a credit was
  applied; a regime that cannot be evaluated (nem3 without an avoided-cost
  vector) is reported in ``export_credit_note`` rather than guessed. Credits
  are per month (no roll-over / annual true-up).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from sqlalchemy import select

from vpp.db.models import CustomerProfileModel, MeterReadingModel, SiteModel
from vpp.db.repositories import TariffRepository
from vpp.portal.access import owned_sites
from vpp.portal.telemetry import as_utc
from vpp.tariffs import Bill, BillingPeriod, MeterTrace, load_urdb_json
from vpp.tariffs.nem import NEMConfigError, compute_export_credit, nem_config_from_urdb

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_MONTH_RE = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")


class BillingError(Exception):
    """A bill cannot be produced; ``status`` is the HTTP status to surface."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def parse_month(
    month: str | None, tz: ZoneInfo, *, now: datetime | None = None
) -> tuple[int, int]:
    if month is None:
        local_now = (now or datetime.now(timezone.utc)).astimezone(tz)
        return local_now.year, local_now.month
    m = _MONTH_RE.match(month)
    if not m:
        raise BillingError(422, "month must be formatted YYYY-MM")
    return int(m.group(1)), int(m.group(2))


def month_bounds(year: int, month: int, tz: ZoneInfo) -> tuple[datetime, datetime]:
    start = datetime(year, month, 1, tzinfo=tz)
    end = datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=tz)
    return start, end


def _prev_month(year: int, month: int) -> tuple[int, int]:
    return (year - 1, 12) if month == 1 else (year, month - 1)


@dataclass
class _Series:
    timestamps: list[datetime]
    import_kwh: list[float]
    export_kwh: list[float]
    interval_minutes: int


async def _load_series(
    session: AsyncSession, site_ids: list[str], start: datetime, end: datetime
) -> _Series:
    if not site_ids:
        return _Series([], [], [], 60)
    rows = (
        await session.execute(
            select(
                MeterReadingModel.timestamp,
                MeterReadingModel.interval_minutes,
                MeterReadingModel.import_kwh,
                MeterReadingModel.export_kwh,
            )
            .where(
                MeterReadingModel.site_id.in_(site_ids),
                MeterReadingModel.timestamp >= as_utc(start),
                MeterReadingModel.timestamp < as_utc(end),
            )
            .order_by(MeterReadingModel.timestamp)
        )
    ).all()
    intervals = {r[1] for r in rows}
    mixed = len(intervals) > 1
    interval = next(iter(intervals)) if len(intervals) == 1 else 60
    agg: dict[datetime, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for ts, _iv, imp, exp in rows:
        ts = as_utc(ts)
        if mixed:  # resample to hourly (all allowed intervals divide 60)
            ts = ts.replace(minute=0, second=0, microsecond=0)
        agg[ts][0] += imp
        agg[ts][1] += exp
    keys = sorted(agg)
    return _Series(
        timestamps=keys,
        import_kwh=[agg[k][0] for k in keys],
        export_kwh=[agg[k][1] for k in keys],
        interval_minutes=interval,
    )


def _bill_dict(
    bill, *, tariff_id: str, start: datetime, end: datetime, metadata: dict
) -> dict[str, Any]:
    return {
        "total": round(bill.total, 2),
        "currency": "USD",  # URDB tariffs are US utility rates
        "tariff_id": tariff_id,
        "line_items": [
            {
                "kind": li.kind,
                "name": li.label,
                "amount": round(li.amount, 2),
                "quantity": li.quantity,
                "unit": li.unit,
                "rate": li.rate,
            }
            for li in bill.line_items
        ],
        "period": {"start": start, "end": end},
        "metadata": {"tariff_name": bill.tariff_name, **metadata},
    }


async def compute_customer_bill(
    session: AsyncSession,
    user_id: str,
    month: str | None = None,
    *,
    nem_override: str | None = None,
) -> dict[str, Any]:
    profile = (
        await session.execute(
            select(CustomerProfileModel).where(CustomerProfileModel.user_id == user_id)
        )
    ).scalar_one_or_none()
    if profile is None or not profile.tariff_id:
        raise BillingError(409, "No tariff is assigned to this account yet")
    row = await TariffRepository.get(session, profile.tariff_id)
    if row is None:
        raise BillingError(409, "The tariff assigned to this account no longer exists")
    urdb_json = json.loads(row.urdb_json) if row.urdb_json else {}
    try:
        tariff = load_urdb_json(urdb_json)
    except Exception as exc:  # surface as a data error, not a 500
        raise BillingError(409, f"The assigned tariff cannot be evaluated: {exc}") from exc

    sites: list[SiteModel] = await owned_sites(session, user_id)
    tz_name = (sites[0].timezone if sites else None) or "UTC"
    tz = ZoneInfo(tz_name)
    year, mon = parse_month(month, tz)
    start, end = month_bounds(year, mon, tz)
    site_ids = [s.id for s in sites]

    series = await _load_series(session, site_ids, start, end)
    trace = MeterTrace(
        timestamps=series.timestamps,
        import_kwh=series.import_kwh,
        export_kwh=series.export_kwh,
        interval_minutes=series.interval_minutes,
        tz=tz,
    )
    period = BillingPeriod(start=start, end=end)
    bill = tariff.bill(trace, period)

    nem_cfg = nem_config_from_urdb(urdb_json)
    regime = nem_override or nem_cfg.regime
    nem_source = "override" if nem_override else nem_cfg.source
    credit_note = ""
    credit_amount = 0.0
    try:
        credit = compute_export_credit(
            tariff,
            trace,
            start,
            end,
            regime=regime,
            avoided_cost=nem_cfg.avoided_cost,
            bill=bill,
        )
        credit_note = credit.note
    except NEMConfigError as exc:
        credit = None
        credit_note = str(exc)
    if credit is not None and credit.line_item is not None:
        credit_amount = credit.amount
        bill = Bill(
            line_items=[*bill.line_items, credit.line_item],
            total=round(bill.total - credit.amount, 4),
            period=bill.period,
            tariff_name=bill.tariff_name,
        )

    hours = (end - start).total_seconds() / 3600.0
    expected = round(hours * 60 / series.interval_minutes)
    this_kwh = sum(series.import_kwh)
    metadata = {
        "month": f"{year:04d}-{mon:02d}",
        "timezone": tz_name,
        "site_ids": site_ids,
        "interval_minutes": series.interval_minutes,
        "meter_intervals": len(series.timestamps),
        "expected_intervals": expected,
        "data_coverage": round(len(series.timestamps) / expected, 4) if expected else 0.0,
        "import_kwh": round(this_kwh, 3),
        "export_kwh": round(sum(series.export_kwh), 3),
        "export_credit_applied": credit_amount > 0,
        "export_credit": round(credit_amount, 2),
        "nem_regime": regime,
        "nem_source": nem_source,
    }
    if credit_note:
        metadata["export_credit_note"] = credit_note
    bill_out = _bill_dict(bill, tariff_id=row.id, start=start, end=end, metadata=metadata)

    py, pm = _prev_month(year, mon)
    p_start, p_end = month_bounds(py, pm, tz)
    last_kwh = sum((await _load_series(session, site_ids, p_start, p_end)).import_kwh)

    baseline_out = None
    savings = None
    if profile.baseline_kwh_per_month:
        step = timedelta(minutes=60)
        n = round(hours)
        per = profile.baseline_kwh_per_month / n
        base_trace = MeterTrace(
            timestamps=[as_utc(start) + i * step for i in range(n)],
            import_kwh=[per] * n,
            export_kwh=[0.0] * n,
            interval_minutes=60,
            tz=tz,
        )
        base_bill = tariff.bill(base_trace, period)
        baseline_out = _bill_dict(
            base_bill,
            tariff_id=row.id,
            start=start,
            end=end,
            metadata={
                "baseline_method": "declared_monthly_kwh_flat_profile",
                "import_kwh": round(profile.baseline_kwh_per_month, 3),
            },
        )
        savings = round(baseline_out["total"] - bill_out["total"], 2)

    return {
        "bill": bill_out,
        "baseline": baseline_out,
        "savings": savings,
        "this_month_kwh": round(this_kwh, 3),
        "last_month_kwh": round(last_kwh, 3),
    }
