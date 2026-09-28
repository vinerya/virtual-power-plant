import type { TariffComponent } from "@/lib/api/tariffs";

export const NEM_LABELS: Record<string, string> = {
  none: "No export credit",
  nem2: "NEM 2.0 (retail / TOU)",
  nem3: "NEM 3.0 (avoided cost)",
  net_billing: "Net billing (sell rates)",
};

export function formatRate(value: number | null | undefined, unit: string): string {
  if (value == null) return "—";
  if (unit === "fraction") return `${(value * 100).toFixed(2)}%`;
  if (unit === "$/kWh") return `$${value.toFixed(4)}/kWh`;
  if (unit.startsWith("$/")) return `$${value.toFixed(2)}${unit.slice(1)}`;
  return `${value} ${unit}`;
}

export function componentValue(c: TariffComponent): string {
  if (c.tiers && c.tiers.length > 0) {
    return c.tiers
      .map((t) =>
        t.max_kwh == null
          ? `${formatRate(t.rate, c.unit)} above`
          : `${formatRate(t.rate, c.unit)} ≤ ${t.max_kwh} kWh`,
      )
      .join(" · ");
  }
  return formatRate(c.rate, c.unit);
}

export function money(v: number): string {
  return `${v < 0 ? "−" : ""}$${Math.abs(v).toFixed(2)}`;
}
