"""Tests for the M2 tariff -> optimizer adapter (vpp.tariffs.optimization)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from vpp.optimization.solvers import PyomoPlugin

pyomo_available = PyomoPlugin().is_available()
pytestmark = pytest.mark.skipif(not pyomo_available, reason="pyomo + HiGHS not installed")

if pyomo_available:
    import pyomo.environ as pyo

    from vpp.optimization.formulations.dispatch import build_battery_dispatch_model
    from vpp.tariffs import (
        BillingPeriod,
        DemandCharge,
        MeterTrace,
        Tariff,
        TimeOfUseRate,
        TOUSchedule,
        load_urdb_json,
    )
    from vpp.tariffs.optimization import (
        build_tariff_hooks,
        load_nem3_avoided_cost_2024,
        tariff_to_opt_params,
    )

PRESETS = Path(__file__).resolve().parents[1] / "src" / "vpp" / "tariffs" / "presets"

ALL_DAYS = (True,) * 7
ALL_MONTHS = frozenset(range(1, 13))


def _battery_params(prices: list[float], **extras) -> dict[str, Any]:
    p = {
        "battery_capacity_kwh": 20.0,
        "max_charge_kw": 5.0,
        "max_discharge_kw": 5.0,
        "soc_init": 0.5,
        "soc_min": 0.1,
        "soc_max": 0.95,
        "eta_charge": 0.97,
        "eta_discharge": 0.97,
        "prices": prices,
        "dt_hours": 1.0,
    }
    p.update(extras)
    return p


def _solve(model, time_limit_s: float = 30.0):
    from pyomo.contrib.appsi.solvers.highs import Highs

    s = Highs()
    s.config.time_limit = time_limit_s
    return s.solve(model)


def _flat_tariff(rate: float = 0.30) -> Tariff:
    tou = TimeOfUseRate(
        periods={
            "flat": [TOUSchedule(ALL_DAYS, (0, 24), ALL_MONTHS, rate)],
        }
    )
    return Tariff(name="flat", components=[tou])


def _pge_etouc() -> Tariff:
    return load_urdb_json(PRESETS / "pge_etouc.json")


# ---------------------------------------------------------------------------
# 1. Flat tariff parity
# ---------------------------------------------------------------------------


def test_tariff_energy_cost_matches_baseline_when_flat():
    """A flat tariff via the tariff hook should produce the same dispatch
    cost as the M1 raw-price path with the same constant price (within
    the small wedge from import/export mutex; here load=solar=0 so the
    LP collapses)."""
    T = 24
    rate = 0.20
    prices = [rate] * T
    base_params = _battery_params(prices)
    m_base = build_battery_dispatch_model(base_params)
    _solve(m_base)

    # Build tariff version: flat tariff, NEM2 (sell == buy).
    tariff = _flat_tariff(rate)
    horizon_start = datetime(2024, 7, 15, 0, 0, tzinfo=timezone.utc)
    opt = tariff_to_opt_params(
        tariff, horizon_start, horizon_hours=T, interval_minutes=60, nem="nem2"
    )
    assert all(b == pytest.approx(rate) for b in opt.energy_buy_per_kwh)

    obj_terms, con_builders = build_tariff_hooks(opt)
    tariff_params = _battery_params(
        [0.0] * T,  # base term suppressed below
        disable_base_energy_cost=True,
    )
    m_tar = build_battery_dispatch_model(
        tariff_params,
        objective_terms=obj_terms,
        constraint_builders=con_builders,
    )
    _solve(m_tar)

    # With load=solar=0 and sell==buy, optimizer is indifferent to charge/
    # discharge size beyond the constant flat rate. Both costs should be
    # near zero (no arbitrage).
    assert pyo.value(m_base.cost) == pytest.approx(0.0, abs=1e-3)
    assert pyo.value(m_tar.cost) == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# 2. PG&E TOU dispatch shape
# ---------------------------------------------------------------------------


def test_tou_dispatch_pattern():
    """PG&E E-TOU-C produces discharge during 4-9 PM peak window
    on a battery + flat-load problem."""
    tariff = _pge_etouc()
    # July (summer) -> period_3 (peak) is 4-9 PM at $0.46, off-peak $0.36.
    horizon_start = datetime(2024, 7, 17, 0, 0, tzinfo=timezone.utc)  # Wed
    T = 24
    opt = tariff_to_opt_params(
        tariff, horizon_start, horizon_hours=T, interval_minutes=60, nem="none"
    )
    # Sanity: peak hours have higher rate than off-peak hours.
    assert opt.energy_buy_per_kwh[18] > opt.energy_buy_per_kwh[3] + 0.01

    obj_terms, con_builders = build_tariff_hooks(opt)
    params = _battery_params(
        [0.0] * T,
        disable_base_energy_cost=True,
        load=[2.0] * T,  # 2 kW flat load
    )
    m = build_battery_dispatch_model(
        params, objective_terms=obj_terms, constraint_builders=con_builders
    )
    _solve(m)

    discharge = [pyo.value(m.p_discharge[t]) for t in m.T]
    charge = [pyo.value(m.p_charge[t]) for t in m.T]
    # Most discharge mass should land in 16:00-21:00 window.
    peak_dis = sum(discharge[16:21])
    off_dis = sum(discharge[:16]) + sum(discharge[21:])
    assert peak_dis > off_dis, (peak_dis, off_dis)
    # Charging happens off peak.
    assert sum(charge[16:21]) <= sum(charge[:16]) + 1e-6


# ---------------------------------------------------------------------------
# 3. Demand-charge caps import
# ---------------------------------------------------------------------------


def test_demand_charge_caps_import():
    """A $25/kW non-coincident demand charge causes the optimizer to
    pre-charge in off-peak and discharge during high load to flatten
    grid import. Compared with no demand charge, peak import drops."""
    T = 24
    horizon_start = datetime(2024, 7, 17, 0, 0, tzinfo=timezone.utc)
    # Spiky load: 3 kW base, 8 kW spike at hour 18.
    load = [3.0] * T
    load[18] = 8.0
    load[19] = 8.0

    flat = _flat_tariff(0.20)

    # Without demand charge.
    opt_no = tariff_to_opt_params(flat, horizon_start, T, 60, nem="none")
    obj_terms, cb = build_tariff_hooks(opt_no)
    p = _battery_params([0.0] * T, disable_base_energy_cost=True, load=load)
    m_no = build_battery_dispatch_model(p, obj_terms, cb)
    _solve(m_no)
    peak_no = max(pyo.value(m_no.p_import[t]) for t in m_no.T)

    # With demand charge.
    flat_with_dc = Tariff(
        name="flat+dc",
        components=[
            *flat.components,
            DemandCharge(rate=25.0, window="monthly_max", component_id="dc"),
        ],
    )
    opt_dc = tariff_to_opt_params(flat_with_dc, horizon_start, T, 60, nem="none")
    assert opt_dc.demand_charge_per_kw == 25.0
    assert all(opt_dc.demand_charge_window)  # monthly_max -> always True
    obj_terms2, cb2 = build_tariff_hooks(opt_dc)
    m_dc = build_battery_dispatch_model(p, obj_terms2, cb2)
    _solve(m_dc)
    peak_dc = max(pyo.value(m_dc.p_import[t]) for t in m_dc.T)

    assert peak_dc < peak_no - 1e-3, (peak_dc, peak_no)
    # Peak demand variable is below or equal to the actual peak import.
    assert pyo.value(m_dc.peak_demand_kw) == pytest.approx(peak_dc, abs=1e-3)


# ---------------------------------------------------------------------------
# 4. Ratchet floor binds
# ---------------------------------------------------------------------------


def test_ratchet_floor_binds():
    """prior_demand_max=50, ratchet 75% -> floor 37.5 kW; var must be >=37.5."""
    T = 24
    horizon_start = datetime(2024, 7, 17, 0, 0, tzinfo=timezone.utc)
    flat = _flat_tariff(0.20)
    tariff = Tariff(
        name="ratchet",
        components=[
            *flat.components,
            DemandCharge(
                rate=15.0,
                window="monthly_max",
                ratchet_pct=0.75,
                component_id="dc",
            ),
        ],
    )
    opt = tariff_to_opt_params(
        tariff,
        horizon_start,
        T,
        60,
        nem="none",
        prior_demand_max_kw=50.0,
    )
    assert opt.demand_ratchet_floor_kw == pytest.approx(37.5)

    obj_terms, cb = build_tariff_hooks(opt)
    # Tiny load so unconstrained peak would be ~0.
    params = _battery_params([0.0] * T, disable_base_energy_cost=True, load=[0.5] * T)
    m = build_battery_dispatch_model(params, obj_terms, cb)
    _solve(m)
    assert pyo.value(m.peak_demand_kw) >= 37.5 - 1e-6


# ---------------------------------------------------------------------------
# 5. NEM 2.0 export credit
# ---------------------------------------------------------------------------


def test_nem2_export_credited_at_buy_price():
    """Excess solar exported under NEM 2.0 earns the buy price."""
    T = 24
    horizon_start = datetime(2024, 7, 17, 0, 0, tzinfo=timezone.utc)
    tariff = _pge_etouc()
    opt = tariff_to_opt_params(tariff, horizon_start, T, 60, nem="nem2")
    # Sell == buy under NEM 2.0.
    assert opt.energy_sell_per_kwh == opt.energy_buy_per_kwh

    # Big midday solar, small load -> excess export.
    load = [0.5] * T
    solar = [0.0] * 8 + [6.0] * 8 + [0.0] * 8  # 8 AM - 4 PM solar
    obj_terms, cb = build_tariff_hooks(opt)
    p = _battery_params([0.0] * T, disable_base_energy_cost=True, load=load, solar=solar)
    m = build_battery_dispatch_model(p, obj_terms, cb)
    _solve(m)

    exports = [pyo.value(m.p_export[t]) for t in m.T]
    imports = [pyo.value(m.p_import[t]) for t in m.T]
    assert sum(exports) > 1.0  # actually exporting

    # Validate the energy revenue/cost equality with the model objective.
    energy_term_value = sum(
        opt.energy_buy_per_kwh[t] * imports[t] - opt.energy_sell_per_kwh[t] * exports[t]
        for t in range(T)
    )
    # peak_demand contribution is 0 (no DC component in PG&E E-TOU-C).
    assert pyo.value(m.cost) == pytest.approx(energy_term_value, abs=1e-3)


def test_nem2_uses_explicit_urdb_sell_rate_when_present():
    """When a TOU period defines a distinct sell rate, NEM 2.0 must use it
    instead of assuming exports earn the same as imports."""
    T = 24
    horizon_start = datetime(2024, 7, 17, 0, 0, tzinfo=timezone.utc)
    tou = TimeOfUseRate(
        periods={
            "off": [TOUSchedule(ALL_DAYS, (0, 16), ALL_MONTHS, 0.10, sell_rate=0.08)],
            "peak": [TOUSchedule(ALL_DAYS, (16, 21), ALL_MONTHS, 0.30, sell_rate=0.25)],
            "off2": [TOUSchedule(ALL_DAYS, (21, 24), ALL_MONTHS, 0.10, sell_rate=0.08)],
        }
    )
    tariff = Tariff(name="sell-aware", components=[tou])
    opt = tariff_to_opt_params(tariff, horizon_start, T, 60, nem="nem2")

    assert opt.energy_sell_per_kwh != opt.energy_buy_per_kwh
    for h in range(T):
        expected_sell = 0.25 if 16 <= h < 21 else 0.08
        expected_buy = 0.30 if 16 <= h < 21 else 0.10
        assert opt.energy_buy_per_kwh[h] == pytest.approx(expected_buy)
        assert opt.energy_sell_per_kwh[h] == pytest.approx(expected_sell)


# ---------------------------------------------------------------------------
# 6. NEM 3.0 avoided cost shifts behavior to self-consumption
# ---------------------------------------------------------------------------


def test_nem3_export_avoided_cost():
    """Under NEM 3.0, exports earn far less than retail; the optimizer
    self-consumes more (less export) than under NEM 2.0."""
    T = 24
    horizon_start = datetime(2024, 7, 17, 0, 0, tzinfo=timezone.utc)
    tariff = _pge_etouc()
    load = [0.5] * T
    solar = [0.0] * 8 + [6.0] * 8 + [0.0] * 8

    opt_nem2 = tariff_to_opt_params(tariff, horizon_start, T, 60, nem="nem2")
    obj2, cb2 = build_tariff_hooks(opt_nem2)
    m2 = build_battery_dispatch_model(
        _battery_params([0.0] * T, disable_base_energy_cost=True, load=load, solar=solar),
        obj2,
        cb2,
    )
    _solve(m2)

    ac = load_nem3_avoided_cost_2024()
    opt_nem3 = tariff_to_opt_params(tariff, horizon_start, T, 60, nem="nem3", nem3_avoided_cost=ac)
    # Verify sell < buy at most hours.
    assert all(
        s <= b + 1e-9
        for s, b in zip(opt_nem3.energy_sell_per_kwh, opt_nem3.energy_buy_per_kwh, strict=True)
    )
    obj3, cb3 = build_tariff_hooks(opt_nem3)
    m3 = build_battery_dispatch_model(
        _battery_params([0.0] * T, disable_base_energy_cost=True, load=load, solar=solar),
        obj3,
        cb3,
    )
    _solve(m3)

    exp_nem2 = sum(pyo.value(m2.p_export[t]) for t in m2.T)
    exp_nem3 = sum(pyo.value(m3.p_export[t]) for t in m3.T)
    # Under NEM 3.0 export earns less, so battery should soak up more solar
    # (less midday export).
    assert exp_nem3 <= exp_nem2 + 1e-3


# ---------------------------------------------------------------------------
# 7. Full-month PG&E E-TOU-C bill consistency
# ---------------------------------------------------------------------------


def test_pge_etouc_full_month():
    """Run a 24-hour synthetic optimization, recompute the bill via
    Tariff.bill() on the realized import/export trace, and assert the
    energy-charge total matches the optimizer's energy term."""
    T = 24
    horizon_start = datetime(2024, 7, 17, 0, 0, tzinfo=timezone.utc)
    tariff = _pge_etouc()
    load = [1.0] * T
    solar = [0.0] * 8 + [3.0] * 8 + [0.0] * 8

    opt = tariff_to_opt_params(tariff, horizon_start, T, 60, nem="nem2")
    obj_terms, cb = build_tariff_hooks(opt)
    params = _battery_params([0.0] * T, disable_base_energy_cost=True, load=load, solar=solar)
    m = build_battery_dispatch_model(params, obj_terms, cb)
    _solve(m)

    imports_kwh = [pyo.value(m.p_import[t]) * 1.0 for t in m.T]  # dt=1h
    exports_kwh = [pyo.value(m.p_export[t]) * 1.0 for t in m.T]
    timestamps = [horizon_start + timedelta(hours=t) for t in range(T)]
    trace = MeterTrace(
        timestamps=timestamps,
        import_kwh=imports_kwh,
        export_kwh=exports_kwh,
        interval_minutes=60,
        tz=timezone.utc,
    )
    period = BillingPeriod(start=timestamps[0], end=timestamps[-1] + timedelta(hours=1))
    bill = tariff.bill(trace, period)

    # Energy line items only (PG&E E-TOU-C has no demand component).
    energy_total = sum(li.amount for li in bill.line_items if li.kind == "energy")
    # Optimizer energy cost (positive = cost). With NEM 2.0 the symmetric
    # buy = sell yields cost = sum buy*(import - export). The Bill object
    # only counts imports (it ignores export compensation in M1) so the
    # two will agree only on the import side: build that quantity from the
    # opt buy vector.
    opt_import_cost = sum(opt.energy_buy_per_kwh[t] * imports_kwh[t] for t in range(T))
    assert energy_total == pytest.approx(opt_import_cost, abs=1e-2)
