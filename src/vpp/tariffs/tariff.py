"""Tariff orchestrator: composes rate components into a Bill."""

from __future__ import annotations

from dataclasses import dataclass, field

from .components import (
    AdderRate,
    BillingPeriod,
    BillLineItem,
    Component,
    MinimumBill,
    TaxRate,
)
from .meter import MeterTrace


@dataclass
class Bill:
    """Computed bill: line items plus a total."""

    line_items: list[BillLineItem]
    total: float
    period: BillingPeriod | None = None
    tariff_name: str = ""

    def pretty(self) -> str:
        lines: list[str] = []
        header = f"=== {self.tariff_name or 'Tariff'} bill ==="
        lines.append(header)
        if self.period is not None:
            lines.append(
                f"Period: {self.period.start.isoformat()} -> {self.period.end.isoformat()}"
                f"  ({self.period.days} days)"
            )
        lines.append("-" * len(header))
        lines.append(
            f"{'Kind':<8} {'Label':<32} {'Qty':>10} {'Unit':<5} {'Rate':>9} {'Amount':>10}"
        )
        for li in self.line_items:
            lines.append(
                f"{li.kind:<8} {li.label[:32]:<32} {li.quantity:>10.3f} "
                f"{li.unit:<5} {li.rate:>9.4f} ${li.amount:>9.2f}"
            )
        lines.append("-" * len(header))
        lines.append(f"{'TOTAL':<57} ${self.total:>9.2f}")
        return "\n".join(lines)


@dataclass
class Tariff:
    """A tariff is an ordered list of rate components plus metadata.

    Composition: iterate components, collect line items, sum, then apply
    ``MinimumBill`` make-up if present.
    """

    name: str
    components: list[Component] = field(default_factory=list)
    utility: str = ""
    sector: str = "Residential"
    source: str = ""
    effective_date: str = ""

    def bill(self, trace: MeterTrace, period: BillingPeriod) -> Bill:
        """Compute the bill.

        Ordering (M4):

        1. Primary components (energy, demand, fixed, tier).
        2. Minimum-bill make-up (so adders/taxes apply to the floored amount).
        3. AdderRate components — applied additively against the snapshot of
           the primary subtotal / energy / demand. Adders do NOT compound.
        4. TaxRate components — applied additively against the post-adder
           snapshot. Multiple taxes stack additively, not multiplicatively.
        5. Credit line items (e.g. NEM3 export credits) are NOT applied here;
           they are appended by the API simulator after :meth:`bill`.
        """
        items: list[BillLineItem] = []
        minimum: MinimumBill | None = None
        adders: list[AdderRate] = []
        taxes: list[TaxRate] = []
        for comp in self.components:
            if isinstance(comp, MinimumBill):
                minimum = comp
                continue
            if isinstance(comp, AdderRate):
                adders.append(comp)
                continue
            if isinstance(comp, TaxRate):
                taxes.append(comp)
                continue
            items.extend(comp.compute(trace, period))

        primary_subtotal = round(sum(li.amount for li in items), 4)

        # Minimum bill make-up.
        if minimum is not None and primary_subtotal < minimum.amount:
            makeup = round(minimum.amount - primary_subtotal, 4)
            items.append(
                BillLineItem(
                    kind="minimum",
                    label=minimum.label,
                    quantity=1.0,
                    unit="$",
                    rate=makeup,
                    amount=makeup,
                )
            )
            primary_subtotal = round(minimum.amount, 4)

        # Snapshots BEFORE adders run, so adders don't compound with each other.
        energy_snapshot = round(sum(li.amount for li in items if li.kind in {"energy", "tier"}), 4)
        demand_snapshot = round(sum(li.amount for li in items if li.kind == "demand"), 4)

        # Adders.
        for ad in adders:
            qty: float
            unit: str
            if ad.basis == "per_kwh":
                # Per-kWh adders apply to total imported kWh in the period.
                kwh = 0.0
                for ts, imp in zip(trace.timestamps, trace.import_kwh):
                    if period.start <= ts < period.end:
                        kwh += imp
                qty = round(kwh, 4)
                unit = "kWh"
                amount = round(qty * ad.rate, 4)
            else:  # percent
                if ad.applies_to == "energy":
                    base = energy_snapshot
                elif ad.applies_to == "demand":
                    base = demand_snapshot
                else:
                    base = primary_subtotal
                qty = base
                unit = "$"
                amount = round(base * ad.rate, 4)
            items.append(
                BillLineItem(
                    kind="adder",
                    label=ad.name,
                    quantity=qty,
                    unit=unit,
                    rate=ad.rate,
                    amount=amount,
                    meta={"basis": ad.basis, "applies_to": ad.applies_to},
                )
            )

        # Post-adder snapshot for taxes.
        post_adder_subtotal = round(sum(li.amount for li in items), 4)
        post_adder_energy = round(
            sum(li.amount for li in items if li.kind in {"energy", "tier", "adder"}),
            4,
        )

        # Taxes (additive, not compounding).
        for tx in taxes:
            base = post_adder_energy if tx.applies_to == "energy" else post_adder_subtotal
            amount = round(base * tx.rate, 4)
            items.append(
                BillLineItem(
                    kind="tax",
                    label=tx.name,
                    quantity=base,
                    unit="$",
                    rate=tx.rate,
                    amount=amount,
                    meta={"jurisdiction": tx.jurisdiction, "applies_to": tx.applies_to},
                )
            )

        total = round(sum(li.amount for li in items), 4)
        return Bill(line_items=items, total=total, period=period, tariff_name=self.name)
