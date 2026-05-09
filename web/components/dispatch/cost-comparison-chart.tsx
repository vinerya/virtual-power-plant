"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ExplainerRun } from "@/lib/api/types";

const PALETTE = [
  "hsl(var(--primary))",
  "hsl(var(--muted-foreground))",
  "#a78bfa",
  "#f59e0b",
  "#10b981",
];

export function CostComparisonChart({
  actual,
  counterfactuals,
}: {
  actual: ExplainerRun;
  counterfactuals: ExplainerRun[];
}) {
  const baseline = counterfactuals.find((c) => c.name === "no_action");
  const data = [actual, ...counterfactuals].map((r, i) => ({
    name: r.name,
    cost: r.total_cost,
    delta: baseline ? r.total_cost - baseline.total_cost : 0,
    fill: i === 0 ? PALETTE[0] : PALETTE[(i + 1) % PALETTE.length],
  }));
  return (
    <div
      className="h-56 w-full"
      role="img"
      aria-label={`Total-cost bar chart for ${data.length} runs`}
      data-testid="cost-comparison-chart"
    >
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          margin={{ top: 16, right: 16, bottom: 8, left: 8 }}
        >
          <CartesianGrid strokeOpacity={0.15} vertical={false} />
          <XAxis
            dataKey="name"
            stroke="currentColor"
            strokeOpacity={0.5}
            fontSize={11}
          />
          <YAxis
            tickFormatter={(v: number) => `$${v.toFixed(0)}`}
            stroke="currentColor"
            strokeOpacity={0.5}
            fontSize={11}
            width={48}
          />
          <Tooltip
            contentStyle={{
              background: "hsl(var(--card))",
              border: "1px solid hsl(var(--border))",
              borderRadius: 8,
              fontSize: 12,
            }}
            formatter={(v: number) => [`$${v.toFixed(2)}`, "total"]}
          />
          <Bar dataKey="cost" radius={[4, 4, 0, 0]}>
            {data.map((d, i) => (
              <Cell key={i} fill={d.fill} />
            ))}
            <LabelList
              dataKey="cost"
              position="top"
              formatter={(v: number) => `$${v.toFixed(0)}`}
              fontSize={11}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
