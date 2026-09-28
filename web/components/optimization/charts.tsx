"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const tooltipStyle = {
  background: "hsl(var(--card))",
  border: "1px solid hsl(var(--border))",
  borderRadius: 8,
  fontSize: 12,
};

const PRICE_COLOR = "#d97706"; // amber-600
const CHARGE_COLOR = "hsl(var(--primary))";
const DISCHARGE_COLOR = "#dc2626"; // red-600
export const SERIES_COLORS = [
  "hsl(var(--primary))",
  "#059669",
  "#7c3aed",
  "#d97706",
  "#db2777",
  "#0891b2",
];

function stepLabel(intervalMinutes: number) {
  return (step: number) => {
    const mins = step * intervalMinutes;
    const h = Math.floor(mins / 60);
    const m = mins % 60;
    return `+${h}:${String(m).padStart(2, "0")}`;
  };
}

/**
 * Fleet power per step (bars; + charging, − discharging) against the price
 * signal (line, right axis).
 */
export function PowerPriceChart({
  power,
  prices,
  intervalMinutes,
  priceUnit = "per kWh",
}: {
  power: number[];
  prices: number[];
  intervalMinutes: number;
  priceUnit?: string;
}) {
  const data = power.map((p, i) => ({ step: i, power: p, price: prices[i] }));
  const fmt = stepLabel(intervalMinutes);
  return (
    <div
      className="h-64 w-full"
      role="img"
      aria-label={`Fleet power and price over ${power.length} steps of ${intervalMinutes} minutes. Positive power is charging.`}
      data-testid="power-price-chart"
    >
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid strokeOpacity={0.15} vertical={false} />
          <XAxis
            dataKey="step"
            tickFormatter={fmt}
            stroke="currentColor"
            strokeOpacity={0.4}
            fontSize={11}
            minTickGap={24}
          />
          <YAxis
            yAxisId="kw"
            stroke="currentColor"
            strokeOpacity={0.4}
            fontSize={11}
            width={52}
            tickFormatter={(v: number) => `${v.toFixed(0)}`}
            label={{ value: "kW", angle: -90, position: "insideLeft", fontSize: 11 }}
          />
          <YAxis
            yAxisId="price"
            orientation="right"
            stroke={PRICE_COLOR}
            fontSize={11}
            width={48}
            tickFormatter={(v: number) => v.toFixed(2)}
          />
          <ReferenceLine yAxisId="kw" y={0} stroke="currentColor" strokeOpacity={0.3} />
          <Tooltip
            contentStyle={tooltipStyle}
            labelFormatter={(s) => `Step ${s} (${fmt(s as number)})`}
            formatter={(v: number, name: string) =>
              name === "price"
                ? [`${v.toFixed(4)} ${priceUnit}`, "Price"]
                : [`${v.toFixed(1)} kW`, v >= 0 ? "Charging" : "Discharging"]
            }
          />
          <Legend
            wrapperStyle={{ fontSize: 12 }}
            formatter={(v: string) => (v === "price" ? "Price" : "Power (+ charge / − discharge)")}
          />
          <Bar yAxisId="kw" dataKey="power" isAnimationActive={false}>
            {data.map((d) => (
              <Cell key={d.step} fill={d.power >= 0 ? CHARGE_COLOR : DISCHARGE_COLOR} />
            ))}
          </Bar>
          <Line
            yAxisId="price"
            type="stepAfter"
            dataKey="price"
            stroke={PRICE_COLOR}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

/** State of charge (0-1 → %) per battery at the end of each step. */
export function SocChart({
  series,
  intervalMinutes,
}: {
  series: { name: string; soc: number[] }[];
  intervalMinutes: number;
}) {
  const n = Math.max(0, ...series.map((s) => s.soc.length));
  const data = Array.from({ length: n }, (_, i) => {
    const row: Record<string, number> = { step: i };
    series.forEach((s, j) => {
      if (s.soc[i] != null) row[`s${j}`] = s.soc[i] * 100;
    });
    return row;
  });
  const fmt = stepLabel(intervalMinutes);
  return (
    <div
      className="h-56 w-full"
      role="img"
      aria-label={`State of charge for ${series.length} batter${series.length === 1 ? "y" : "ies"} over ${n} steps`}
      data-testid="soc-chart"
    >
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
          <CartesianGrid strokeOpacity={0.15} vertical={false} />
          <XAxis
            dataKey="step"
            tickFormatter={fmt}
            stroke="currentColor"
            strokeOpacity={0.4}
            fontSize={11}
            minTickGap={24}
          />
          <YAxis
            domain={[0, 100]}
            stroke="currentColor"
            strokeOpacity={0.4}
            fontSize={11}
            width={40}
            tickFormatter={(v: number) => `${v}%`}
          />
          <Tooltip
            contentStyle={tooltipStyle}
            labelFormatter={(s) => `End of step ${s}`}
            formatter={(v: number, key: string) => {
              const idx = Number(key.slice(1));
              return [`${v.toFixed(1)}%`, series[idx]?.name ?? key];
            }}
          />
          {series.length > 1 && (
            <Legend
              wrapperStyle={{ fontSize: 12 }}
              formatter={(key: string) => series[Number(key.slice(1))]?.name ?? key}
            />
          )}
          {series.map((s, j) => (
            <Line
              key={s.name + j}
              type="monotone"
              dataKey={`s${j}`}
              stroke={SERIES_COLORS[j % SERIES_COLORS.length]}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Horizontal bars of total cost per policy (lower is better). */
export function CostBars({
  rows,
}: {
  rows: { name: string; cost: number; highlight?: boolean }[];
}) {
  return (
    <div
      className="h-48 w-full"
      role="img"
      aria-label={`Cost comparison: ${rows.map((r) => `${r.name} ${r.cost.toFixed(2)}`).join(", ")}. Lower is better.`}
      data-testid="cost-bars"
    >
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 24, bottom: 0, left: 8 }}>
          <CartesianGrid strokeOpacity={0.15} horizontal={false} />
          <XAxis
            type="number"
            stroke="currentColor"
            strokeOpacity={0.4}
            fontSize={11}
            tickFormatter={(v: number) => v.toFixed(0)}
          />
          <YAxis
            type="category"
            dataKey="name"
            stroke="currentColor"
            strokeOpacity={0.4}
            fontSize={11}
            width={132}
          />
          <ReferenceLine x={0} stroke="currentColor" strokeOpacity={0.4} />
          <Tooltip
            contentStyle={tooltipStyle}
            formatter={(v: number) => [v.toFixed(2), "Cost"]}
          />
          <Bar dataKey="cost" isAnimationActive={false}>
            {rows.map((r) => (
              <Cell
                key={r.name}
                fill={r.highlight ? "hsl(var(--primary))" : "hsl(var(--muted-foreground))"}
                fillOpacity={r.highlight ? 1 : 0.5}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
