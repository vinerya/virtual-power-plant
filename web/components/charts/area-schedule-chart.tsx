"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export interface SchedulePoint {
  step: number;
  charge?: number;
  discharge?: number;
  power?: number;
}

export function AreaScheduleChart({
  data,
}: {
  data: SchedulePoint[];
}) {
  return (
    <div
      className="h-48 w-full"
      role="img"
      aria-label={`Per-timestep dispatch schedule, ${data.length} steps`}
    >
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={data}
          margin={{ top: 8, right: 16, bottom: 8, left: 8 }}
        >
          <CartesianGrid strokeOpacity={0.15} vertical={false} />
          <XAxis
            dataKey="step"
            stroke="currentColor"
            strokeOpacity={0.4}
            fontSize={11}
          />
          <YAxis
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
          />
          <Area
            type="monotone"
            dataKey="charge"
            stackId="1"
            stroke="hsl(var(--primary))"
            fill="hsl(var(--primary))"
            fillOpacity={0.3}
            isAnimationActive={false}
          />
          <Area
            type="monotone"
            dataKey="discharge"
            stackId="1"
            stroke="hsl(var(--destructive))"
            fill="hsl(var(--destructive))"
            fillOpacity={0.3}
            isAnimationActive={false}
          />
          <Area
            type="monotone"
            dataKey="power"
            stroke="hsl(var(--success))"
            fill="hsl(var(--success))"
            fillOpacity={0.25}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
