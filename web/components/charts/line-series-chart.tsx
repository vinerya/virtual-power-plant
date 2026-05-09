"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export interface SeriesPoint {
  timestamp: string;
  value: number;
}

export function LineSeriesChart({
  data,
  unit = "kW",
  color = "hsl(var(--primary))",
}: {
  data: SeriesPoint[];
  unit?: string;
  color?: string;
}) {
  const formatted = data.map((p) => ({
    ...p,
    t: new Date(p.timestamp).getTime(),
  }));
  return (
    <div
      className="h-64 w-full"
      role="img"
      aria-label={`Time series chart with ${data.length} points in ${unit}`}
    >
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={formatted}
          margin={{ top: 8, right: 16, bottom: 8, left: 8 }}
        >
          <CartesianGrid strokeOpacity={0.15} vertical={false} />
          <XAxis
            dataKey="t"
            type="number"
            domain={["dataMin", "dataMax"]}
            tickFormatter={(t: number) =>
              new Date(t).toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              })
            }
            stroke="currentColor"
            strokeOpacity={0.4}
            fontSize={11}
          />
          <YAxis
            tickFormatter={(v: number) => `${v.toFixed(0)}`}
            stroke="currentColor"
            strokeOpacity={0.4}
            fontSize={11}
            width={40}
          />
          <Tooltip
            contentStyle={{
              background: "hsl(var(--card))",
              border: "1px solid hsl(var(--border))",
              borderRadius: 8,
              fontSize: 12,
            }}
            labelFormatter={(t) =>
              new Date(t as number).toLocaleString()
            }
            formatter={(v: number) => [`${v.toFixed(2)} ${unit}`, ""]}
          />
          <Line
            type="monotone"
            dataKey="value"
            stroke={color}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
