"use client";

import { useMemo } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Alert } from "@/lib/api/types";

const COLORS = {
  info: "#94a3b8",
  warning: "#f59e0b",
  critical: "hsl(var(--destructive))",
} as const;

interface Bucket {
  hour: string;
  hourLabel: string;
  info: number;
  warning: number;
  critical: number;
}

export function SeveritySparkline({ alerts }: { alerts: Alert[] }) {
  const buckets = useMemo<Bucket[]>(() => {
    const now = new Date();
    const start = new Date(now);
    start.setMinutes(0, 0, 0);
    start.setHours(start.getHours() - 23);
    const buckets: Bucket[] = [];
    for (let i = 0; i < 24; i++) {
      const d = new Date(start);
      d.setHours(start.getHours() + i);
      buckets.push({
        hour: d.toISOString(),
        hourLabel: `${String(d.getHours()).padStart(2, "0")}:00`,
        info: 0,
        warning: 0,
        critical: 0,
      });
    }
    for (const a of alerts) {
      const t = new Date(a.timestamp);
      const idx = Math.floor(
        (t.getTime() - start.getTime()) / (60 * 60 * 1000),
      );
      if (idx >= 0 && idx < 24) {
        buckets[idx][a.severity] += 1;
      }
    }
    return buckets;
  }, [alerts]);

  return (
    <div
      className="h-32 w-full"
      role="img"
      aria-label="24-hour alert volume by severity"
      data-testid="severity-sparkline"
    >
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={buckets}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
          <XAxis
            dataKey="hourLabel"
            tick={{ fontSize: 10 }}
            interval={3}
          />
          <YAxis tick={{ fontSize: 10 }} allowDecimals={false} width={20} />
          <Tooltip
            contentStyle={{ fontSize: 12 }}
            labelFormatter={(l) => `Hour ${l}`}
          />
          <Area
            type="monotone"
            dataKey="info"
            stackId="1"
            stroke={COLORS.info}
            fill={COLORS.info}
            fillOpacity={0.5}
          />
          <Area
            type="monotone"
            dataKey="warning"
            stackId="1"
            stroke={COLORS.warning}
            fill={COLORS.warning}
            fillOpacity={0.55}
          />
          <Area
            type="monotone"
            dataKey="critical"
            stackId="1"
            stroke={COLORS.critical}
            fill={COLORS.critical}
            fillOpacity={0.6}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
