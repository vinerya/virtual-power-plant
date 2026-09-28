"""Unit tests for the simulation helpers behind POST /tariffs/{id}/simulate."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from vpp.tariffs import MeterTrace, load_urdb_json
from vpp.tariffs.csv_trace import CSVTraceError, parse_csv_trace
from vpp.tariffs.nem import NEMConfigError, nem_config_from_urdb, normalize_regime
from vpp.tariffs.simulation import billing_cycles, simulate_bill
from vpp.tariffs.synthetic_load import synthetic_trace

LA = ZoneInfo("America/Los_Angeles")
FLAT = {
    "name": "flat",
    "energyratestructure": [[{"rate": 0.2}]],
    "energyweekdayschedule": [[0] * 24] * 12,
    "energyweekendschedule": [[0] * 24] * 12,
    "fixedchargefirstmeter": 10,
}


def test_billing_cycles_clamp_month_end_and_keep_anchor():
    start = datetime(2024, 1, 31, tzinfo=LA)
    end = datetime(2024, 4, 15, tzinfo=LA)
    cycles = billing_cycles(start, end, LA, "auto")
    assert [c[0].date().isoformat() for c in cycles] == [
        "2024-01-31",
        "2024-02-29",
        "2024-03-31",
    ]
    assert cycles[-1][1] == end
    assert billing_cycles(start, start + timedelta(days=31), LA, "auto") == [
        (start, start + timedelta(days=31))
    ]
    with pytest.raises(ValueError):
        billing_cycles(start, end, LA, "weekly")


def test_monthly_cycles_charge_fixed_per_cycle_and_sum():
    tariff = load_urdb_json(FLAT)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    trace = MeterTrace.constant_load(1.0, start, days=60)
    res = simulate_bill(tariff, trace, start, start + timedelta(days=60), cycle_mode="monthly")
    assert len(res.cycles) == 2
    fixed = next(li for li in res.line_items if li.kind == "fixed")
    assert fixed.quantity == 2 and fixed.amount == pytest.approx(20.0)
    assert res.total == pytest.approx(20.0 + 60 * 24 * 0.2)


def test_unsorted_trace_is_billed_correctly():
    tariff = load_urdb_json(FLAT)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ts = [start + timedelta(hours=h) for h in (3, 1, 2, 0)]
    trace = MeterTrace(timestamps=ts, import_kwh=[1.0] * 4)
    res = simulate_bill(tariff, trace, start, start + timedelta(days=1))
    energy = next(li for li in res.line_items if li.kind == "energy")
    assert energy.quantity == pytest.approx(4.0)


def test_synthetic_pv_exports_only_in_daylight_and_is_deterministic():
    start = datetime(2024, 6, 1, tzinfo=LA)
    a = synthetic_trace(start=start, days=7, tz=LA, pv_kw=8)
    b = synthetic_trace(start=start, days=7, tz=LA, pv_kw=8)
    assert a.import_kwh == b.import_kwh and a.export_kwh == b.export_kwh
    assert sum(a.export_kwh) > 0
    for i, exp in enumerate(a.export_kwh):
        if exp > 0:
            assert 7 <= a.local_time(i).hour <= 17
    base = synthetic_trace(start=start, days=7, tz=LA, avg_kw=2.0)
    assert sum(base.import_kwh) / (7 * 24) == pytest.approx(2.0, rel=0.05)
    with pytest.raises(ValueError):
        synthetic_trace(start=start, days=1, tz=LA, profile="industrial")


def test_csv_kw_and_kwh_layouts():
    kw = parse_csv_trace("timestamp,kw\n2024-01-01T00:30,-2\n2024-01-01T00:00,4\n", tz=LA)
    assert kw.interval_minutes == 30
    assert kw.timestamps[0] == datetime(2024, 1, 1, 0, 0, tzinfo=LA)  # sorted, local
    assert kw.import_kwh == [2.0, 0.0] and kw.export_kwh == [0.0, 1.0]
    kwh = parse_csv_trace(
        "Timestamp,import_kwh,export_kwh\n2024-01-01T00:00Z,1,0.5\n2024-01-01T01:00Z,2,0\n",
        tz=LA,
    )
    assert kwh.interval_minutes == 60 and kwh.export_kwh == [0.5, 0.0]
    with pytest.raises(CSVTraceError, match="single row"):
        parse_csv_trace("timestamp,kw\n2024-01-01T00:00,1\n", tz=LA)


def test_nem_config_derivation():
    assert nem_config_from_urdb({}).regime == "none"
    assert nem_config_from_urdb({"dgrules": "Net Metering"}).regime == "nem2"
    assert nem_config_from_urdb({"dgrules": "Net Billing Hourly"}).regime == "net_billing"
    cfg = nem_config_from_urdb({"dgrules": "Net Billing Hourly", "nem3_avoided_cost": [0.1]})
    assert cfg.regime == "nem3" and cfg.avoided_cost == (0.1,)
    ext = nem_config_from_urdb({"nem": "NEM3", "dgrules": "Net Metering"})
    assert ext.regime == "nem3" and ext.source == "tariff"
    assert normalize_regime("NEM-2") == "nem2"
    with pytest.raises(NEMConfigError):
        normalize_regime("nem9")


def test_net_billing_credits_only_explicit_sell_rates():
    urdb = {
        "name": "two-period",
        "energyratestructure": [[{"rate": 0.10, "sell": 0.04}], [{"rate": 0.30}]],
        "energyweekdayschedule": [[1 if 16 <= h < 20 else 0 for h in range(24)]] * 12,
        "energyweekendschedule": [[1 if 16 <= h < 20 else 0 for h in range(24)]] * 12,
    }
    tariff = load_urdb_json(urdb)
    start = datetime(2024, 7, 1, tzinfo=timezone.utc)
    exports = [0.0] * 24
    exports[12] = 10.0  # period 0, sell 0.04
    exports[17] = 10.0  # period 1, no sell rate
    trace = MeterTrace(
        timestamps=[start + timedelta(hours=h) for h in range(24)],
        import_kwh=[0.0] * 24,
        export_kwh=exports,
    )
    res = simulate_bill(tariff, trace, start, start + timedelta(days=1), nem_regime="net_billing")
    assert res.export_credit == pytest.approx(0.4)
    assert any("left uncredited" in n for n in res.notes)
    nem2 = simulate_bill(tariff, trace, start, start + timedelta(days=1), nem_regime="nem2")
    assert nem2.export_credit == pytest.approx(0.4 + 3.0)  # retail fallback for period 1
