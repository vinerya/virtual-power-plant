"""Tests for the M1 tariff engine."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from vpp.tariffs import (
    Bill,
    BillingPeriod,
    DemandCharge,
    FixedCharge,
    MeterTrace,
    MinimumBill,
    Tariff,
    TieredEnergyRate,
    TimeOfUseRate,
    TOUSchedule,
    load_urdb_json,
)

PRESETS = Path(__file__).resolve().parents[1] / "src" / "vpp" / "tariffs" / "presets"

ALL_DAYS = (True,) * 7
WEEKDAYS = (True, True, True, True, True, False, False)
ALL_MONTHS = frozenset(range(1, 13))
SUMMER = frozenset({6, 7, 8, 9})
WINTER = frozenset(set(range(1, 13)) - SUMMER)


def _trace_30day(kw: float = 1.0, start: datetime | None = None) -> MeterTrace:
    if start is None:
        start = datetime(2024, 7, 1, tzinfo=timezone.utc)
    return MeterTrace.constant_load(kw=kw, start=start, days=30, interval_minutes=60)


def _period(trace: MeterTrace) -> BillingPeriod:
    return BillingPeriod(
        start=trace.timestamps[0],
        end=trace.timestamps[-1] + (trace.timestamps[1] - trace.timestamps[0]),
    )


def test_tou_simple():
    """24h flat 1 kW load through a 2-period TOU yields the right total."""
    # 12h on-peak @ $0.40, 12h off-peak @ $0.10. 24 kWh total.
    # Expect 12 kWh @ 0.40 + 12 kWh @ 0.10 = $4.80 + $1.20 = $6.00.
    tou = TimeOfUseRate(
        periods={
            "off": [TOUSchedule(ALL_DAYS, (0, 12), ALL_MONTHS, 0.10)],
            "on": [TOUSchedule(ALL_DAYS, (12, 24), ALL_MONTHS, 0.40)],
        }
    )
    tariff = Tariff(name="test-tou", components=[tou])
    trace = MeterTrace.constant_load(
        kw=1.0, start=datetime(2024, 7, 1, tzinfo=timezone.utc), days=1
    )
    bill = tariff.bill(trace, _period(trace))
    assert bill.total == pytest.approx(6.00, abs=1e-4)
    labels = {li.label for li in bill.line_items}
    assert "TOU on" in labels and "TOU off" in labels


def test_demand_charge_picks_max():
    """Single peak hour at 5 kW drives demand charge."""
    start = datetime(2024, 7, 1, tzinfo=timezone.utc)
    timestamps = [start.replace(hour=h) for h in range(24)]
    imp = [1.0] * 24
    imp[18] = 5.0  # 6 PM peak
    trace = MeterTrace(timestamps=timestamps, import_kwh=imp, interval_minutes=60, tz=timezone.utc)
    dc = DemandCharge(rate=20.0, window="monthly_max")
    tariff = Tariff(name="dc-test", components=[dc])
    bill = tariff.bill(trace, _period(trace))
    assert bill.total == pytest.approx(100.0)  # 5 kW * $20
    assert bill.line_items[0].quantity == pytest.approx(5.0)


def test_demand_charge_ratchet():
    """Current-month demand below 75% of prior 11-month max -> ratchet floor applies."""
    start = datetime(2024, 7, 1, tzinfo=timezone.utc)
    timestamps = [start.replace(hour=h) for h in range(24)]
    imp = [1.0] * 24  # peak = 1 kW this month
    trace = MeterTrace(timestamps=timestamps, import_kwh=imp, interval_minutes=60, tz=timezone.utc)
    dc = DemandCharge(rate=10.0, window="monthly_max", ratchet_pct=0.75, component_id="d")
    period = _period(trace)
    period.prior_peaks_kw["d"] = [10.0, 8.0, 9.5]  # max 10
    tariff = Tariff(name="ratchet-test", components=[dc])
    bill = tariff.bill(trace, period)
    # Floor = 0.75 * 10 = 7.5 kW > actual 1 kW => billed at 7.5
    assert bill.line_items[0].quantity == pytest.approx(7.5)
    assert bill.line_items[0].meta["ratcheted"] is True
    assert bill.total == pytest.approx(75.0)


def test_tiered_energy():
    """700 kWh through {350: 0.10, inf: 0.20} -> $35 + $70 = $105."""
    tier = TieredEnergyRate(tiers=[(350.0, 0.10), (float("inf"), 0.20)])
    # 700 kWh total over period: 1 hour @ 700 kW (artificial, but charge logic only sums kWh)
    start = datetime(2024, 7, 1, tzinfo=timezone.utc)
    trace = MeterTrace(
        timestamps=[start],
        import_kwh=[700.0],
        interval_minutes=60,
        tz=timezone.utc,
    )
    period = BillingPeriod(start=start, end=start.replace(hour=2))
    tariff = Tariff(name="tier-test", components=[tier])
    bill = tariff.bill(trace, period)
    assert bill.total == pytest.approx(105.0)
    assert len(bill.line_items) == 2
    assert bill.line_items[0].amount == pytest.approx(35.0)
    assert bill.line_items[1].amount == pytest.approx(70.0)


def test_urdb_roundtrip():
    """Load PG&E E-TOU-C preset and bill a synthetic trace."""
    tariff = load_urdb_json(PRESETS / "pge_etouc.json")
    assert tariff.name == "PG&E E-TOU-C"
    trace = _trace_30day(kw=1.0)
    bill = tariff.bill(trace, _period(trace))
    assert isinstance(bill, Bill)
    assert bill.total > 0
    kinds = {li.kind for li in bill.line_items}
    # PG&E E-TOU-C has no fixed charge (>$0) but does have a minimum.
    assert "energy" in kinds
    # Each line item structure
    for li in bill.line_items:
        assert li.kind and li.label and li.unit
        assert li.amount >= 0


def test_seasons_switch():
    """Same load in summer vs. winter shows different totals when tariff is seasonal."""
    tou = TimeOfUseRate(
        periods={
            "summer_flat": [TOUSchedule(ALL_DAYS, (0, 24), SUMMER, 0.50)],
            "winter_flat": [TOUSchedule(ALL_DAYS, (0, 24), WINTER, 0.10)],
        }
    )
    tariff = Tariff(name="seasonal", components=[tou])

    # July (summer)
    summer_trace = MeterTrace.constant_load(
        kw=1.0, start=datetime(2024, 7, 1, tzinfo=timezone.utc), days=10
    )
    summer_bill = tariff.bill(summer_trace, _period(summer_trace))

    # January (winter)
    winter_trace = MeterTrace.constant_load(
        kw=1.0, start=datetime(2024, 1, 1, tzinfo=timezone.utc), days=10
    )
    winter_bill = tariff.bill(winter_trace, _period(winter_trace))

    assert summer_bill.total > winter_bill.total
    # 10 days * 24 kWh = 240 kWh; 240 * 0.50 = 120; 240 * 0.10 = 24
    assert summer_bill.total == pytest.approx(120.0, rel=1e-3)
    assert winter_bill.total == pytest.approx(24.0, rel=1e-3)


def test_minimum_bill():
    """Bill below minimum is rounded up; line item shows the make-up."""
    tou = TimeOfUseRate(periods={"flat": [TOUSchedule(ALL_DAYS, (0, 24), ALL_MONTHS, 0.10)]})
    minimum = MinimumBill(amount=50.0)
    tariff = Tariff(name="min-test", components=[tou, minimum])
    trace = MeterTrace.constant_load(
        kw=1.0, start=datetime(2024, 7, 1, tzinfo=timezone.utc), days=1
    )
    # 24 kWh * $0.10 = $2.40 < $50 minimum => $47.60 make-up
    bill = tariff.bill(trace, _period(trace))
    assert bill.total == pytest.approx(50.0)
    makeups = [li for li in bill.line_items if li.kind == "minimum"]
    assert len(makeups) == 1
    assert makeups[0].amount == pytest.approx(47.60, abs=1e-2)


def test_fixed_charge_daily_vs_monthly():
    fc_daily = FixedCharge(amount=0.50, frequency="daily")
    fc_monthly = FixedCharge(amount=10.0, frequency="monthly")
    trace = _trace_30day(kw=1.0)
    period = _period(trace)
    [li_d] = fc_daily.compute(trace, period)
    [li_m] = fc_monthly.compute(trace, period)
    assert li_d.amount == pytest.approx(0.50 * period.days)
    assert li_m.amount == pytest.approx(10.0)


def test_pretty_bill_renders():
    tou = TimeOfUseRate(periods={"flat": [TOUSchedule(ALL_DAYS, (0, 24), ALL_MONTHS, 0.10)]})
    tariff = Tariff(name="pretty", components=[tou])
    trace = _trace_30day()
    bill = tariff.bill(trace, _period(trace))
    out = bill.pretty()
    assert "TOTAL" in out
    assert "pretty" in out


def test_load_sce_preset():
    tariff = load_urdb_json(PRESETS / "sce_toudprime.json")
    assert "SCE" in tariff.name
    trace = _trace_30day()
    bill = tariff.bill(trace, _period(trace))
    assert bill.total > 0
    # daily fixed charge expected
    assert any(li.kind == "fixed" for li in bill.line_items)


# ---------------------------------------------------------------------------
# NEM export rate (sell)
# ---------------------------------------------------------------------------


def test_tou_schedule_sell_rate_defaults_to_none():
    sch = TOUSchedule(ALL_DAYS, (0, 24), ALL_MONTHS, 0.20)
    assert sch.sell_rate is None


def test_export_rate_falls_back_to_import_rate_when_sell_unset():
    tou = TimeOfUseRate(periods={"flat": [TOUSchedule(ALL_DAYS, (0, 24), ALL_MONTHS, 0.20)]})
    dt = datetime(2024, 7, 15, 10, tzinfo=timezone.utc)
    assert tou.export_rate(dt) == pytest.approx(0.20)


def test_export_rate_prefers_explicit_sell_rate():
    tou = TimeOfUseRate(
        periods={
            "off": [TOUSchedule(ALL_DAYS, (0, 12), ALL_MONTHS, 0.10, sell_rate=0.08)],
            "on": [TOUSchedule(ALL_DAYS, (12, 24), ALL_MONTHS, 0.40, sell_rate=0.35)],
        }
    )
    off_dt = datetime(2024, 7, 15, 5, tzinfo=timezone.utc)
    on_dt = datetime(2024, 7, 15, 18, tzinfo=timezone.utc)
    assert tou.export_rate(off_dt) == pytest.approx(0.08)
    assert tou.export_rate(on_dt) == pytest.approx(0.35)


def test_export_rate_none_outside_any_period():
    tou = TimeOfUseRate(
        periods={"business_hours": [TOUSchedule(WEEKDAYS, (9, 17), ALL_MONTHS, 0.20)]}
    )
    saturday_evening = datetime(2024, 7, 20, 20, tzinfo=timezone.utc)  # 2024-07-20 is a Sat
    assert tou.export_rate(saturday_evening) is None


def test_urdb_parses_sell_field_into_tou_schedule():
    """energyratestructure[..].sell must populate TOUSchedule.sell_rate."""
    weekday_row = [1 if 16 <= h < 20 else 0 for h in range(24)]
    urdb_json = {
        "name": "Test URDB sell parsing",
        "energyratestructure": [
            [{"rate": 0.10, "sell": 0.08}],
            [{"rate": 0.30, "sell": 0.25}],
        ],
        "energyweekdayschedule": [weekday_row] * 12,
        "energyweekendschedule": [weekday_row] * 12,
    }
    tariff = load_urdb_json(urdb_json)
    tou = next(c for c in tariff.components if isinstance(c, TimeOfUseRate))

    off_peak = datetime(2024, 7, 15, 5, tzinfo=timezone.utc)  # Monday, off-peak
    peak = datetime(2024, 7, 15, 17, tzinfo=timezone.utc)  # Monday, peak (16-20)
    assert tou.export_rate(off_peak) == pytest.approx(0.08)
    assert tou.export_rate(peak) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Combined tiered + TOU
# ---------------------------------------------------------------------------


def test_tou_period_tiers_defaults_to_empty():
    tou = TimeOfUseRate(periods={"flat": [TOUSchedule(ALL_DAYS, (0, 24), ALL_MONTHS, 0.20)]})
    assert tou.period_tiers == {}


def test_tou_compute_applies_per_period_tiers_independently():
    """Off-peak stays flat; peak is tiered against ITS OWN cumulative kWh
    for the billing period, not whole-bill usage."""
    tou = TimeOfUseRate(
        periods={
            "off": [TOUSchedule(ALL_DAYS, (0, 16), ALL_MONTHS, 0.10)],
            "peak": [TOUSchedule(ALL_DAYS, (16, 24), ALL_MONTHS, 0.20)],
        },
        period_tiers={"peak": [(5.0, 0.20), (float("inf"), 0.35)]},
    )
    tariff = Tariff(name="tiered-tou", components=[tou])
    trace = MeterTrace.constant_load(
        kw=1.0, start=datetime(2024, 7, 1, tzinfo=timezone.utc), days=1, interval_minutes=60
    )
    bill = tariff.bill(trace, _period(trace))

    # off: 16 kWh @ 0.10 flat = 1.60
    off_items = [li for li in bill.line_items if li.label == "TOU off"]
    assert len(off_items) == 1
    assert off_items[0].amount == pytest.approx(1.60)

    # peak: 8 kWh total -> tier1 5 kWh @ 0.20 + tier2 3 kWh @ 0.35
    tier_items = sorted(
        (li for li in bill.line_items if li.label.startswith("TOU peak tier")),
        key=lambda li: li.label,
    )
    assert len(tier_items) == 2
    assert tier_items[0].quantity == pytest.approx(5.0)
    assert tier_items[0].amount == pytest.approx(1.0)
    assert tier_items[1].quantity == pytest.approx(3.0)
    assert tier_items[1].amount == pytest.approx(1.05)

    assert bill.total == pytest.approx(1.60 + 1.0 + 1.05)


def test_tou_compute_tiered_period_stays_flat_when_under_first_threshold():
    """When a tiered period's usage never crosses the first threshold, only
    one tier line item appears (no phantom zero-kWh tier rows)."""
    tou = TimeOfUseRate(
        periods={"peak": [TOUSchedule(ALL_DAYS, (0, 24), ALL_MONTHS, 0.20)]},
        period_tiers={"peak": [(100.0, 0.20), (float("inf"), 0.35)]},
    )
    tariff = Tariff(name="under-threshold", components=[tou])
    trace = MeterTrace.constant_load(
        kw=1.0, start=datetime(2024, 7, 1, tzinfo=timezone.utc), days=1, interval_minutes=60
    )
    bill = tariff.bill(trace, _period(trace))
    tier_items = [li for li in bill.line_items if li.label.startswith("TOU peak tier")]
    assert len(tier_items) == 1
    assert tier_items[0].quantity == pytest.approx(24.0)
    assert tier_items[0].amount == pytest.approx(24.0 * 0.20)


def test_urdb_parses_combined_tiered_tou():
    """>1 TOU period referenced AND >1 tier in one of them must produce a
    TimeOfUseRate with period_tiers populated -- not silently collapse to
    just that period's tier-0 rate."""
    weekday_row = [1 if 16 <= h < 24 else 0 for h in range(24)]
    urdb_json = {
        "name": "Test tiered TOU",
        "energyratestructure": [
            [{"rate": 0.10}],  # period 0 ("off"): flat
            [{"rate": 0.20, "max": 5}, {"rate": 0.35}],  # period 1 ("peak"): tiered
        ],
        "energyweekdayschedule": [weekday_row] * 12,
        "energyweekendschedule": [weekday_row] * 12,
    }
    tariff = load_urdb_json(urdb_json)
    tou_components = [c for c in tariff.components if isinstance(c, TimeOfUseRate)]
    assert len(tou_components) == 1
    tou = tou_components[0]
    # No TieredEnergyRate should be added for the multi-period case.
    assert not any(isinstance(c, TieredEnergyRate) for c in tariff.components)

    assert "period_1" in tou.period_tiers
    assert tou.period_tiers["period_1"] == [(5.0, 0.20), (float("inf"), 0.35)]
    assert "period_0" not in tou.period_tiers  # single-tier period stays flat


def test_urdb_single_period_multi_tier_still_uses_tiered_energy_rate():
    """Regression guard: exactly one referenced period with multiple tiers
    must still take the simpler TieredEnergyRate shortcut, unaffected by
    the new per-period-tiers machinery."""
    urdb_json = {
        "name": "Test single-period tiered",
        "energyratestructure": [
            [{"rate": 0.15, "max": 500}, {"rate": 0.25}],
        ],
        "energyweekdayschedule": [[0] * 24] * 12,
        "energyweekendschedule": [[0] * 24] * 12,
    }
    tariff = load_urdb_json(urdb_json)
    assert any(isinstance(c, TieredEnergyRate) for c in tariff.components)
    assert not any(isinstance(c, TimeOfUseRate) for c in tariff.components)


def test_urdb_without_sell_field_export_rate_matches_import_rate():
    """No `sell` in the source JSON -> export_rate falls back to `rate`
    per period (not a single tariff-wide value)."""
    tariff = load_urdb_json(PRESETS / "pge_etouc.json")
    tou = next(c for c in tariff.components if isinstance(c, TimeOfUseRate))
    for scheds in tou.periods.values():
        for sch in scheds:
            assert sch.sell_rate is None
    dt = datetime(2024, 7, 15, 18, tzinfo=timezone.utc)
    label = tou._classify(dt)
    assert label is not None
    assert tou.export_rate(dt) == pytest.approx(tou.periods[label][0].rate)
