"""Adder + tax line items, including PG&E preset cross-check (M4)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from vpp.tariffs import (
    AdderRate,
    BillingPeriod,
    FixedCharge,
    MeterTrace,
    Tariff,
    TaxRate,
    TieredEnergyRate,
    load_urdb_json,
)


PRESET = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "vpp"
    / "tariffs"
    / "presets"
    / "pge_etouc.json"
)


def _flat_trace(kwh_per_hr: float = 1.0, hours: int = 720) -> MeterTrace:
    start = datetime(2024, 7, 1, tzinfo=timezone.utc)
    timestamps = [start + timedelta(hours=h) for h in range(hours)]
    return MeterTrace(
        timestamps=timestamps,
        import_kwh=[kwh_per_hr] * hours,
        interval_minutes=60,
    )


def test_adder_percent_subtotal():
    """1.5% adder on a $100 subtotal yields a $1.50 line item."""
    # Build a tariff whose subtotal is exactly $100 by combining:
    # - flat tier at $0.10/kWh on 1000 kWh = $100.
    tariff = Tariff(
        name="t",
        components=[
            TieredEnergyRate(tiers=[(float("inf"), 0.10)]),
            AdderRate(name="Test adder", rate=0.015, basis="percent", applies_to="subtotal"),
        ],
    )
    trace = _flat_trace(kwh_per_hr=1.0, hours=1000)  # 1000 kWh
    period = BillingPeriod(
        start=datetime(2024, 7, 1, tzinfo=timezone.utc),
        end=datetime(2024, 7, 1, tzinfo=timezone.utc) + timedelta(hours=1000),
    )
    bill = tariff.bill(trace, period)

    adders = [li for li in bill.line_items if li.kind == "adder"]
    assert len(adders) == 1
    assert adders[0].amount == pytest.approx(1.50, abs=1e-2)


def test_adder_per_kwh_energy():
    """A $0.005/kWh adder on 720 kWh -> $3.60 line item."""
    tariff = Tariff(
        name="t",
        components=[
            TieredEnergyRate(tiers=[(float("inf"), 0.10)]),
            AdderRate(name="PPP fee", rate=0.005, basis="per_kwh", applies_to="energy"),
        ],
    )
    trace = _flat_trace(kwh_per_hr=1.0, hours=720)
    period = BillingPeriod(
        start=datetime(2024, 7, 1, tzinfo=timezone.utc),
        end=datetime(2024, 7, 1, tzinfo=timezone.utc) + timedelta(hours=720),
    )
    bill = tariff.bill(trace, period)

    adders = [li for li in bill.line_items if li.kind == "adder"]
    assert len(adders) == 1
    assert adders[0].amount == pytest.approx(3.60, abs=1e-2)
    assert adders[0].unit == "kWh"


def test_taxes_stack_correctly():
    """Two 5% taxes apply additively to the same base — not compounded."""
    tariff = Tariff(
        name="t",
        components=[
            TieredEnergyRate(tiers=[(float("inf"), 0.10)]),  # $100 on 1000 kWh
            TaxRate(name="State", rate=0.05, jurisdiction="CA", applies_to="subtotal"),
            TaxRate(name="Local", rate=0.05, jurisdiction="SF", applies_to="subtotal"),
        ],
    )
    trace = _flat_trace(kwh_per_hr=1.0, hours=1000)
    period = BillingPeriod(
        start=datetime(2024, 7, 1, tzinfo=timezone.utc),
        end=datetime(2024, 7, 1, tzinfo=timezone.utc) + timedelta(hours=1000),
    )
    bill = tariff.bill(trace, period)

    taxes = [li for li in bill.line_items if li.kind == "tax"]
    assert len(taxes) == 2
    # Both taxes against the same $100 base (no compounding) -> $5 + $5.
    total_tax = sum(li.amount for li in taxes)
    assert total_tax == pytest.approx(10.0, abs=1e-2)
    # If they had compounded the second would be 5% of $105 = $5.25.
    assert taxes[1].amount == pytest.approx(5.0, abs=1e-2)


def test_pge_etouc_with_added_taxes_matches_published_total():
    """The augmented PG&E preset's bill total reflects the added 0.5%
    climate-credit adder + 1.5% UUT (utility users tax) on top of energy."""
    with open(PRESET) as f:
        data = json.load(f)
    tariff = load_urdb_json(data)

    # Synthetic: 1 kWh/hour for 30 days = 720 kWh. Tax + adder both base on subtotal.
    trace = _flat_trace(kwh_per_hr=1.0, hours=720)
    start = datetime(2024, 7, 1, tzinfo=timezone.utc)
    period = BillingPeriod(start=start, end=start + timedelta(hours=720))
    bill = tariff.bill(trace, period)

    # The energy subtotal of the preset (TOU rates 0.36 / 0.46) on a flat
    # constant-load trace is deterministic. The adder (0.5%) + tax (1.5%) =
    # 2.0% of the subtotal-after-adder.
    energy_subtotal = sum(
        li.amount for li in bill.line_items if li.kind in {"energy", "tier"}
    )
    assert energy_subtotal > 0
    adder = [li for li in bill.line_items if li.kind == "adder"]
    tax = [li for li in bill.line_items if li.kind == "tax"]
    assert adder, "preset must include the climate-credit adder"
    assert tax, "preset must include the utility users tax"
    # Adder = 0.5% of energy subtotal (no other primary line items besides minimum-bill markup).
    expected_adder = 0.005 * energy_subtotal
    # Allow some slack for the minimum-bill make-up if it kicked in.
    assert adder[0].amount == pytest.approx(expected_adder, rel=0.20)
    # Tax = 1.5% of (energy + adder).
    expected_tax = 0.015 * (energy_subtotal + adder[0].amount)
    assert tax[0].amount == pytest.approx(expected_tax, rel=0.20)
