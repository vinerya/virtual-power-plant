"use client";

import { useMemo } from "react";

const MONTHS = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
];

/**
 * 12 × 24 calendar heatmap. Cells colored by relative rate.
 * Hand-rendered with CSS grid — keeps Recharts off this page until the
 * Simulate tab is opened.
 */
export function ScheduleHeatmap({
  heatmap,
  unit = "$/kWh",
}: {
  heatmap: number[][];
  unit?: string;
}) {
  const { min, max } = useMemo(() => {
    let mn = Infinity;
    let mx = -Infinity;
    for (const row of heatmap)
      for (const v of row) {
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
    if (!Number.isFinite(mn)) mn = 0;
    if (!Number.isFinite(mx)) mx = 1;
    if (mn === mx) mx = mn + 0.0001;
    return { min: mn, max: mx };
  }, [heatmap]);

  return (
    <figure
      className="space-y-2"
      role="group"
      aria-label="Tariff time-of-use schedule"
      data-testid="schedule-heatmap"
    >
      <div className="grid grid-cols-[2.5rem_repeat(24,minmax(0,1fr))] gap-px text-[10px] tabular-nums">
        <span aria-hidden="true" />
        {Array.from({ length: 24 }, (_, h) => (
          <span key={h} className="text-center text-muted-foreground">
            {h}
          </span>
        ))}
        {heatmap.map((row, m) => (
          <Row
            key={m}
            label={MONTHS[m] ?? `M${m + 1}`}
            row={row}
            min={min}
            max={max}
            unit={unit}
          />
        ))}
      </div>
      <figcaption className="flex items-center gap-2 text-xs text-muted-foreground">
        <span>Low</span>
        <span
          aria-hidden="true"
          className="inline-block h-2 w-24 rounded"
          style={{
            background:
              "linear-gradient(to right, hsl(210 80% 92%), hsl(220 80% 50%), hsl(0 80% 45%))",
          }}
        />
        <span>High</span>
        <span className="ml-2">
          {min.toFixed(3)}–{max.toFixed(3)} {unit}
        </span>
      </figcaption>
    </figure>
  );
}

function Row({
  label,
  row,
  min,
  max,
  unit,
}: {
  label: string;
  row: number[];
  min: number;
  max: number;
  unit: string;
}) {
  return (
    <>
      <span className="pr-1 text-right text-muted-foreground">{label}</span>
      {row.map((v, h) => {
        const t = (v - min) / (max - min);
        const hue = 210 - 210 * t; // blue → red
        const light = 92 - 50 * t;
        return (
          <span
            key={h}
            role="gridcell"
            aria-label={`${label} ${h}:00, ${v.toFixed(3)} ${unit}`}
            title={`${v.toFixed(3)} ${unit}`}
            className="aspect-square w-full"
            style={{ background: `hsl(${hue} 80% ${light}%)` }}
          />
        );
      })}
    </>
  );
}
