"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ExplainerRun } from "@/lib/api/types";

const COLORS: Record<string, string> = {
  actual: "hsl(var(--primary))",
  no_action: "#9ca3af",
  price_naive: "#a78bfa",
};

function netPower(s: { charge?: number; discharge?: number; power?: number }) {
  if (typeof s.power === "number") return s.power;
  return (s.charge ?? 0) - (s.discharge ?? 0);
}

export function PerStepComparisonChart({
  actual,
  counterfactuals,
}: {
  actual: ExplainerRun;
  counterfactuals: ExplainerRun[];
}) {
  const all = [actual, ...counterfactuals];
  const len = Math.max(...all.map((r) => r.per_step.length));
  const merged: Record<string, number | string>[] = [];
  for (let i = 0; i < len; i++) {
    const row: Record<string, number | string> = { step: i };
    for (const run of all) {
      const s = run.per_step[i];
      if (s) row[run.name] = netPower(s);
    }
    merged.push(row);
  }

  return (
    <div
      className="h-64 w-full"
      role="img"
      aria-label={`Per-step net-power overlay over ${len} steps`}
      data-testid="per-step-comparison-chart"
    >
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={merged} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid strokeOpacity={0.15} vertical={false} />
          <XAxis
            dataKey="step"
            stroke="currentColor"
            strokeOpacity={0.5}
            fontSize={11}
          />
          <YAxis
            tickFormatter={(v: number) => `${v.toFixed(0)}`}
            stroke="currentColor"
            strokeOpacity={0.5}
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
            formatter={(v: number) => `${v.toFixed(2)} kW`}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {all.map((run) => (
            <Line
              key={run.name}
              type="monotone"
              dataKey={run.name}
              stroke={COLORS[run.name] ?? "#10b981"}
              strokeWidth={run.name === "actual" ? 2.25 : 1.25}
              strokeDasharray={run.name === "actual" ? undefined : "4 3"}
              dot={false}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
