"use client";

import {
  CartesianGrid,
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

function timeTick(t: number) {
  return new Date(t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

/** Live last-price trace for one market (points received since page load). */
export function PriceTrace({
  data,
  unit,
  label,
}: {
  data: { t: number; price: number }[];
  unit: string;
  label: string;
}) {
  return (
    <div className="h-40 w-full" role="img" aria-label={label}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid strokeOpacity={0.15} vertical={false} />
          <XAxis
            dataKey="t"
            type="number"
            domain={["dataMin", "dataMax"]}
            tickFormatter={timeTick}
            stroke="currentColor"
            strokeOpacity={0.4}
            fontSize={11}
            minTickGap={40}
          />
          <YAxis
            domain={["auto", "auto"]}
            stroke="currentColor"
            strokeOpacity={0.4}
            fontSize={11}
            width={48}
            tickFormatter={(v: number) => v.toFixed(1)}
          />
          <Tooltip
            contentStyle={tooltipStyle}
            labelFormatter={(t) => new Date(t as number).toLocaleTimeString()}
            formatter={(v: number) => [`${v.toFixed(2)} ${unit}`, "Last"]}
          />
          <Line
            type="stepAfter"
            dataKey="price"
            stroke="hsl(var(--primary))"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Backtest equity curve with the starting capital as a reference line. */
export function EquityCurve({
  data,
  initial,
}: {
  data: { t: number; equity: number }[];
  initial: number;
}) {
  return (
    <div
      className="h-64 w-full"
      role="img"
      aria-label={`Equity curve over ${data.length} bars, starting at ${initial.toFixed(0)}`}
      data-testid="equity-curve"
    >
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 8 }}>
          <CartesianGrid strokeOpacity={0.15} vertical={false} />
          <XAxis
            dataKey="t"
            type="number"
            domain={["dataMin", "dataMax"]}
            tickFormatter={(t: number) =>
              new Date(t).toLocaleDateString([], { month: "short", day: "numeric" })
            }
            stroke="currentColor"
            strokeOpacity={0.4}
            fontSize={11}
            minTickGap={40}
          />
          <YAxis
            domain={["auto", "auto"]}
            stroke="currentColor"
            strokeOpacity={0.4}
            fontSize={11}
            width={64}
            tickFormatter={(v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          />
          <ReferenceLine
            y={initial}
            stroke="currentColor"
            strokeOpacity={0.4}
            strokeDasharray="4 4"
          />
          <Tooltip
            contentStyle={tooltipStyle}
            labelFormatter={(t) => new Date(t as number).toLocaleString()}
            formatter={(v: number) => [
              v.toLocaleString(undefined, { maximumFractionDigits: 2 }),
              "Equity",
            ]}
          />
          <Line
            type="monotone"
            dataKey="equity"
            stroke="hsl(var(--primary))"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
