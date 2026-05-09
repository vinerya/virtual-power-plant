"""Tariff orchestrator: composes rate components into a Bill."""
from __future__ import annotations

from dataclasses import dataclass, field

from .components import (
    BillingPeriod,
    BillLineItem,
    Component,
    MinimumBill,
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
        lines.append(f"{'Kind':<8} {'Label':<32} {'Qty':>10} {'Unit':<5} {'Rate':>9} {'Amount':>10}")
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
        items: list[BillLineItem] = []
        minimum: MinimumBill | None = None
        for comp in self.components:
            if isinstance(comp, MinimumBill):
                minimum = comp
                continue
            items.extend(comp.compute(trace, period))
        subtotal = round(sum(li.amount for li in items), 4)
        if minimum is not None and subtotal < minimum.amount:
            makeup = round(minimum.amount - subtotal, 4)
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
            subtotal = round(minimum.amount, 4)
        return Bill(line_items=items, total=subtotal, period=period, tariff_name=self.name)
